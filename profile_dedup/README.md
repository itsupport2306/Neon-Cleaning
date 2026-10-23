# profile_dedup

Safe duplicate detection and merging for the NeonDB `profiles` table.

Runs read-only by default. Nothing is written to the database unless you
pass `--apply`, and even then every change happens inside a transaction that
checks for foreign-key references before it will ever delete a row.

## Why this looks the way it does

Before writing any matching logic, the schema was inspected directly
(`information_schema`, `pg_constraint`, `pg_indexes`) against the live
NeonDB. Three facts from that inspection drove the whole design:

1. **`npi_number` and `email` already have DB-level unique constraints**
   (`profiles_npi_number_key`, `ux_profiles_email_ci`). Two live rows can
   never share a non-empty NPI or email today, so real-world duplicates in
   this table show up through `phone` + `name`, not NPI/email. (NPI/email
   matching is still implemented, defensively, in case that constraint is
   ever relaxed during a bulk import - it's cheap to check and currently
   returns zero groups.)

2. **Phone numbers are heavily shared.** The most common phone values are
   attached to 5,000-18,000 different profiles each - these are
   staffing-agency/front-desk switchboard numbers, not personal numbers.
   Treating "same phone" as a duplicate signal on its own would merge
   thousands of unrelated people. So phone is only ever used **together
   with** a matching or near-matching name, and groups larger than
   `PROFILE_DEDUP_MAX_PHONE_GROUP_SIZE` (default 5) are excluded from
   matching entirely as shared-line noise.

3. **Most foreign keys into `profiles` are `ON DELETE CASCADE`**
   (applications, interviews, offers, saved_jobs, post activity, etc.), and
   two (`outreach_messages`, `submissions`) are `ON DELETE NO ACTION`. The
   table also already has `merged_into`/`merged_at` columns from an earlier
   soft-merge process (1,024 rows). Physically deleting a duplicate that has
   any activity history would silently destroy that history via cascade.
   So **deletion is never the default** - see "What happens to a duplicate
   row" below.

## Match confidence levels

| Level | Signal | Auto-applied with `--apply`? |
|---|---|---|
| `CERTAIN` | Matching NPI number or matching email (currently always empty - see above) | Yes |
| `HIGH` | Same normalized phone number (in a small, non-shared-line group) **and** exactly matching normalized name | Yes |
| `MEDIUM` | Same phone number, name is *similar but not identical* (typo, nickname, missing middle name) | No - always written to the report for manual review, even with `--apply` (use `--apply-medium` to override, not recommended) |

**Hard veto:** if two rows have a matching/similar name and phone but carry
*different*, non-empty NPI numbers, they are never clustered at any
confidence level - a disagreeing NPI is treated as proof they're different
entities (e.g. two facilities sharing a switchboard and a generic org name).
This was caught on a real dry run against production data (see
`profile_dedup/reports/` history) before it shipped.

Matching first and last names alone, with no phone/email/NPI corroboration,
is **never** treated as a duplicate signal.

## Merge rules

Each mergeable column (`profile_dedup/merge_logic.py::MERGE_RULES`) has a
strategy:

- **`prefer_non_empty`** (most text/professional fields): fill blanks from
  whichever record has the value; if both have different non-empty values,
  keep the primary's and record a conflict for manual review - never guess.
- **`prefer_verified_contact`** (phone, email): a contact value backed by
  `contact_updated_by_user_id`/`contact_updated_by_email` (i.e. user
  confirmed) always wins over an unverified one. If neither/both are
  verified and they disagree, keep the primary's and flag a conflict -
  contact details are never auto-overwritten on a guess.
- **`prefer_most_recent`** (`open_to_work`, `job_type_prefs`,
  `pay_min_hourly`, `available_date`): these are user-stated current
  preferences, so whichever row has the more recent `updated_at` wins.
- **`prefer_max`** (`years_experience`, `completion_score`): numeric,
  larger is strictly more informative.
- **`prefer_non_empty_json`** (`resume_sections`, `education`): the
  array/object with more entries wins; if both are non-empty and differ,
  the longer-or-equal one is kept and a conflict is logged (arrays are not
  silently deep-merged/deduplicated - that's a separate, higher-risk
  problem left for manual review).

`NULL`, `''`, whitespace-only, `0` (for `years_experience`/
`pay_min_hourly` specifically, where 0 is a placeholder, not a fact),
`false` (never treated as "empty" - `open_to_work=false` is a real answer),
and `[]`/`{}` are all handled per-field via `normalize.is_empty_value()`.

When a primary profile has multiple duplicates, each duplicate is folded in
**in order against the running merged state**, so information from several
duplicates combines (one has a resume, another has NPI + address, a third
has years of experience) without any of it getting discarded. See
`tests/test_profile_dedup.py::test_merge_combines_info_from_multiple_duplicates`.

## Primary profile selection

The survivor keeps its own `profile_id`. Scored by (`profile_dedup/primary.py`):
NPI present > verified email > verified phone > resume present > license
present > photo present > `completion_score` > `is_listable`. Ties broken by
most-recently-updated, then oldest account, then `profile_id` for full
determinism.

## What happens to a duplicate row

For every duplicate folded into a primary:

1. Its data is merged into the primary (above).
2. All 19 tables with a foreign key into `profiles.profile_id` are checked
   for rows referencing that duplicate's `profile_id`
   (`db.py::FK_CHILD_TABLES`, matching the live schema).
3. **If zero rows reference it anywhere** → it is safe to delete, and it
   is physically `DELETE`d.
4. **If anything references it** (an application, a saved job, a post,
   etc.) → it is **soft-merged**: `merged_into = <primary_id>`,
   `merged_at = now()`. The row stays, its activity history stays intact,
   and it's excluded from all future matching (`WHERE merged_into IS NULL`).
   This reuses the exact pattern already present on 1,024 rows in the table.

Either way, nothing is ever deleted when identity is uncertain - only
`CERTAIN`/`HIGH` clusters reach this step at all under default settings.

## Transactions, idempotency, and recovery

- Each cluster is processed in its own transaction. Rows are re-fetched with
  `FOR UPDATE` immediately before writing, and if any member was already
  merged elsewhere by a concurrent run, the whole cluster is skipped (no
  partial writes).
- Every `profile_id` in a candidate cluster is first resolved through any
  existing `merged_into` chain, so re-running the script is idempotent -
  already-merged profiles are recognized and skipped rather than reprocessed.
- A `profile_merge_audit` table is created on first `--apply` run
  (`db.py::ensure_audit_table`) and gets one row per (primary, duplicate)
  pair: the full original JSON of both rows, the final merged values, which
  source each field came from, every conflict, and the action taken
  (`soft_merge` or `deleted`). This is enough to manually reconstruct a
  deleted row if ever needed.
- Any exception inside a cluster's transaction rolls back that cluster only
  (`psycopg2`'s `with conn:` block) and is recorded under `errors` in the
  run's report; other clusters in the same run are unaffected.

## Optional: local Qwen second opinion

If `LOCAL_MODEL_BASE_URL`/`LOCAL_MODEL_ENABLED` are configured (same env
vars `profile_enrichment` already uses for its local Qwen server), `MEDIUM`
confidence clusters get an additional AI opinion
(`same_person`/`confidence`/`reason`) attached to the report purely as a
hint for the human reviewer. It never influences whether something is
auto-applied - `MEDIUM` is always manual-review-only regardless of what the
model says. Disable with `PROFILE_DEDUP_USE_QWEN=false`.

## Usage

```bash
# Dry run (default) - writes a report, touches nothing in the DB.
python -m profile_dedup.run_dedup

# Dry run on a small sample while testing.
python -m profile_dedup.run_dedup --limit-groups 50

# Apply CERTAIN/HIGH clusters. Prompts for a typed "APPLY" confirmation.
python -m profile_dedup.run_dedup --apply

# Also apply MEDIUM clusters (discouraged - review the report by hand instead).
python -m profile_dedup.run_dedup --apply --apply-medium
```

Reports land in `profile_dedup/reports/` as matching `.json` (full detail,
every field decision) and `.csv` (one row per primary/duplicate pair, for a
quick skim) files, named `dedup_<dryrun|apply>_<run_id>_<timestamp>.*`.

### Config (env vars, all optional - see `profile_dedup/config.py`)

| Variable | Default | Meaning |
|---|---|---|
| `NEON_DB_URL` / `DATABASE_URL` | - | Reused from the existing `.env` (required) |
| `NEON_TABLE_NAME` | `profiles` | Reused from the existing `.env` |
| `PROFILE_DEDUP_AUDIT_TABLE` | `profile_merge_audit` | Audit table name |
| `PROFILE_DEDUP_MAX_PHONE_GROUP_SIZE` | `5` | Phone groups larger than this are shared-line noise, excluded entirely |
| `PROFILE_DEDUP_MIN_PHONE_DIGITS` | `10` | Minimum digits for a phone to be considered real |
| `PROFILE_DEDUP_NAME_FUZZY_RATIO` | `0.82` | `difflib` ratio threshold for a MEDIUM name match |
| `PROFILE_DEDUP_GROUPS_PER_BATCH` | `500` | Candidate groups fetched per identifier type per run |
| `PROFILE_DEDUP_REPORT_DIR` | `profile_dedup/reports/` | Where reports are written |
| `PROFILE_DEDUP_USE_QWEN` | `true` | Toggle the optional AI second opinion on MEDIUM clusters |

## Tests

```bash
pytest tests/test_profile_dedup.py -v
```

All matching/merge/primary-selection logic is pure Python (no DB
connection needed to test it) - see `tests/test_profile_dedup.py`, including
a regression test for the NPI-conflict veto that was caught live against
production data.

## Scale note

The live table has ~11.9M rows; a shared-phone-group scan across the whole
table currently returns on the order of 1M candidate rows across ~2.3M
small groups. The phone query is paginated server-side by `--limit-groups`
(default batch size `PROFILE_DEDUP_GROUPS_PER_BATCH`), so a full run should
be done in batches (re-running is idempotent - already-merged profiles are
skipped automatically) rather than as a single unbounded pass.
