from pathlib import Path
from pdf2image import convert_from_path
from datetime import datetime

source_dir = Path("/Users/jodut/Projects/parsing_engine/data/Training Docs/Sydney Notes/Preped for Training")
output_dir = Path("/Users/jodut/Projects/parsing_engine/data/Training Docs/Sydney Notes/images")
output_dir.mkdir(parents=True, exist_ok=True)

pdfs = list(source_dir.rglob("*.pdf"))
print(f"📄 Found {len(pdfs)} PDFs to convert...")

count = 0
for pdf in pdfs:
    if pdf.suffix.lower() != ".pdf" or pdf.name.startswith("._"):
        print(f"⏩ Skipping system or non-PDF: {pdf.name}")
        continue
    print(f"→ Processing: {pdf}")
    try:
        pages = convert_from_path(str(pdf), dpi=300)
        for i, page in enumerate(pages):
            out_path = output_dir / f"{pdf.stem}_page{i+1}.png"
            page.save(out_path)
            count += 1
    except Exception as e:
        print(f"⚠️ Failed to process {pdf.name}: {e}")

print(f"✅ Extracted {count} pages to {output_dir}")