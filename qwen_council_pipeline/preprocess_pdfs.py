"""
Standalone PDF preprocessor — use if you skipped conversion in download_papers.py.

Converts all PDFs in papers/ to Markdown using marker-pdf, preserving:
  - Mathematical notation (LaTeX)
  - Tables and structured data
  - Figures/images (saved to papers_markdown/<paper_id>/images/)
  - Proper cross-references

This is called automatically by download_papers.py unless --no-convert is set.
Run this manually if you added new PDFs after the initial download.

Usage:
    python preprocess_pdfs.py                        # process all PDFs
    python preprocess_pdfs.py --input papers --output papers_markdown
"""
import os
import subprocess
import argparse
from pathlib import Path


def process_pdfs_with_marker(input_dir: str = "papers", output_dir: str = "papers_markdown"):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    pdfs = sorted(input_path.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {input_dir}/")
        return

    print(f"\nProcessing {len(pdfs)} PDFs with marker-pdf...\n")

    for i, pdf in enumerate(pdfs, 1):
        paper_id = pdf.stem
        paper_out = output_path / paper_id

        # Skip if already converted
        if paper_out.exists() and list(paper_out.glob("*.md")):
            print(f"  [{i}/{len(pdfs)}] {paper_id} — cached, skipping")
            continue

        print(f"  [{i}/{len(pdfs)}] {paper_id} — converting...", end=" ", flush=True)

        try:
            result = subprocess.run(
                ["marker_single", str(pdf), "--output_dir", str(output_path)],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0:
                md_files = list(paper_out.glob("*.md")) if paper_out.exists() else []
                if md_files:
                    # Count extracted images
                    img_dir = paper_out / "images"
                    n_imgs = len(list(img_dir.iterdir())) if img_dir.exists() else 0
                    print(f"OK ({md_files[0].name}, {n_imgs} images)")
                else:
                    print(f"OK (no .md found in output)")
            else:
                print(f"FAILED: {result.stderr[:150]}")
        except subprocess.TimeoutExpired:
            print("TIMEOUT (>5 min)")
        except FileNotFoundError:
            print("ERROR: 'marker_single' not found. Install: pip install marker-pdf")
            return

    print(f"\nDone. Markdown output in {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PDFs to Markdown using marker-pdf")
    parser.add_argument("--input", default="papers", help="Input directory with PDFs")
    parser.add_argument("--output", default="papers_markdown", help="Output directory for Markdown")
    args = parser.parse_args()
    process_pdfs_with_marker(args.input, args.output)
