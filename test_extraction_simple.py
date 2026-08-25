#!/usr/bin/env python
"""Test if parse_pdf_bytes function works correctly."""

from profile_enrichment.extractors import parse_pdf_bytes
import io

# Create a simple test PDF with text
try:
    from pypdf import PdfWriter

    writer = PdfWriter()
    page = writer.add_blank_page(width=200, height=200)

    # Try to add text (this might not work with pypdf - it's for writing PDFs)
    print("pypdf can create PDFs but adding text requires additional libraries")
except Exception as e:
    print(f"Error creating test PDF: {e}")

# Instead, let's test the extraction logic itself
print("\nTesting extract_text_layers_from_bytes function...")

from profile_enrichment.extractors import extract_text_layers_from_bytes

# Test with a non-PDF file (should fail gracefully)
test_data = b"This is just text, not a PDF"

try:
    result = extract_text_layers_from_bytes(test_data, ".txt", ocr_enabled=False)
    print(f"✓ TXT extraction worked")
    print(f"  Combined text length: {len(result['combined_text'])}")
    print(f"  Content: {result['combined_text'][:50]}")
except Exception as e:
    print(f"✗ TXT extraction failed: {e}")

print("\nNow checking PDF parsing...")

# For a PDF, let's check what happens with invalid data
try:
    result = extract_text_layers_from_bytes(b"%PDF-1.4\n%invalid", ".pdf", ocr_enabled=False)
    print(f"✓ Invalid PDF handling worked (returned {len(result['combined_text'])} chars)")
except Exception as e:
    print(f"✗ Invalid PDF handling failed: {e}")

print("\nChecking if files can be properly detected...")
import os

test_file_refs = [
    "/files/resumes/import/51/51cb8d5d47a9e35c5d2aeb5262a3dbae7239706b5f6f1e65fbed21cd2f0ae39c.pdf",
    "/files/resumes/import/23/23769c3ee2f06fb771919f3302d505000ec5a355ddddf1bf760b87cd98ba245d.docx",
    "document.pdf?v=123",
]

for ref in test_file_refs:
    url_path = ref.split("?")[0] if "?" in ref else ref
    _, ext = os.path.splitext(url_path.lower())
    ext = ext or ".bin"
    print(f"  {ref:70} -> {ext}")
