#!/usr/bin/env python
"""Diagnostic script to test text extraction from downloaded files."""

from profile_enrichment.downloaders import download_file_bytes
from profile_enrichment.extractors import extract_text_layers_from_bytes, parse_pdf_bytes
from profile_enrichment.neon_db import fetch_profiles_for_reprocessing
import io

# Get a few test profiles
profiles = fetch_profiles_for_reprocessing(limit=3)

if not profiles:
    print("No profiles found to test")
    exit(1)

for profile in profiles[:2]:
    profile_id = profile.get("profile_id")
    file_ref = (
        profile.get("resume_url")
        or profile.get("resume_path")
        or profile.get("cloudflare_file")
        or profile.get("file_path")
        or profile.get("document_path")
    )

    if not file_ref:
        print(f"SKIP {profile_id}: No file reference")
        continue

    print(f"\n{'='*60}")
    print(f"Testing: {profile_id}")
    print(f"File ref: {file_ref}")

    try:
        # Download bytes
        file_data = download_file_bytes(file_ref)
        if not file_data:
            print("❌ Download returned None")
            continue

        print(f"✓ Downloaded {len(file_data)} bytes")

        # Extract extension
        import os
        url_path = file_ref.split("?")[0] if "?" in file_ref else file_ref
        _, ext = os.path.splitext(url_path.lower())
        ext = ext or ".bin"
        print(f"  File extension: {ext}")

        # Try extraction with OCR disabled
        text_layers = extract_text_layers_from_bytes(file_data, ext, ocr_enabled=False)

        native_len = len(text_layers.get("native_text", "").strip())
        ocr_len = len(text_layers.get("ocr_text", "").strip())
        combined_len = len(text_layers.get("combined_text", "").strip())

        print(f"  Native text: {native_len} chars")
        print(f"  OCR text: {ocr_len} chars")
        print(f"  Combined: {combined_len} chars")

        if combined_len == 0:
            print("  ⚠️  NO TEXT EXTRACTED")
            # Try to see what parse_pdf_bytes returns
            if ext == ".pdf":
                try:
                    raw_pdf_text = parse_pdf_bytes(file_data)
                    print(f"    Direct PDF parse: {len(raw_pdf_text.strip())} chars")
                except Exception as e:
                    print(f"    Direct PDF parse failed: {e}")
        else:
            print(f"  ✓ Text extracted successfully")
            print(f"  Sample: {text_layers['combined_text'][:100]}...")

    except Exception as exc:
        print(f"❌ Error: {exc}")

print(f"\n{'='*60}")
print("Diagnostic complete")
