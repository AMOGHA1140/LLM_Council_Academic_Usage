#!/usr/bin/env python3
"""
ICLR 2021 Paper Pipeline — Download, Convert, and Label
========================================================

Downloads PDFs from OpenReview, converts them to high-fidelity Markdown
(with images) using marker-pdf, and builds a labeled dataset for the
LLM Council evaluation pipeline.

Usage:
    python download_papers.py                         # process all papers below
    python download_papers.py --skip-download         # skip download, use existing PDFs
    python download_papers.py --no-convert            # download only, skip markdown conversion

Paper IDs are defined in PAPERS below. Add/remove entries as needed.
The OpenReview ID is the last segment of the URL:
    https://openreview.net/forum?id=<THIS_PART>
"""

import json
import os
import subprocess
import sys
import time
import argparse
import requests
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# PAPER REGISTRY — Edit these lists to change which papers are processed.
#
# Format: (openreview_id, short_title)
# The ID is from the URL: https://openreview.net/forum?id=<ID>
#
# ⚡ INSTRUCTIONS: Browse https://openreview.net/group?id=ICLR.cc/2021/Conference
#    Click on papers → copy the ID from the URL → paste here.
#    Accepted papers are under the "Accept" tabs.
#    Rejected papers are under "Withdrawn/Rejected Submissions".
# ═══════════════════════════════════════════════════════════════════════════════

ACCEPTED_PAPERS = [
    # Verified ICLR 2021 accepted papers (Oral/Spotlight)
    ("zv-typ1gPxA",  "Retrieval-Augmented Generation for Code Summarization via Hybrid GNN"),
    ("iAX0l6Cz8ub", "Geometry-aware Instance-reweighted Adversarial Training"),
    ("rcQdycl0zyk", "Beyond Fully-Connected Layers with Quaternions: Parameterization of Hypercomplex Multiplications with 1/n Parameters"),
    ("Mos9F9kDwkz", "Complex Query Answering with Neural Link Predictors"),
    ("NzTU59SYbNq",  "EigenGame: PCA as a Nash Equilibrium"),
    ("rsf1z-JSj87", "End-to-end Adversarial Text-to-Speech "),
    ("uCY5MuAxcxU", "Why Are Convolutional Nets More Sample-Efficient than Fully-Connected Nets?"),
    ("Pd_oMxH8IlF", "Iterated learning for emergent systematicity in VQA"),
    ("m5Qsh0kBQG", "Deep symbolic regression: Recovering mathematical expressions from data via risk-seeking policy gradients"),
    ("xppLmXCbOw1", "Self-supervised Visual Reinforcement Learning with Object-centric Representations"),
    ("PUkhWz65dy5", "Discovering a set of policies for the worst case reward"),
    ("LmUJqB1Cz8", "Winning the L2RPN Challenge: Power Grid Management via Semi-Markov Afterstate Actor-Critic"),
    ("3UDSdyIcBDA", "RMSprop converges with proper hyper-parameter"),
    ("ZPa2SyGcbwh", "Learning with Feature-Dependent Label Noise: A Progressive Approach"),
    ("opHLcXxYTC_", "Influence Estimation for Generative Adversarial Networks"),
]

REJECTED_PAPERS = [
    ("kB8DkEKSDH", "Hellinger Distance Constrained Regression"),
    ("-aThAo4b1zn", "A Theory of Self-Supervised Framework for Few-Shot Learning"),
    ("j0uePNuoBho", "Learned Threshold Pruning"),
    ("N5Zacze7uru", "Neural Lyapunov Model Predictive Control"),
    ("trPMYEn1FCX", "GENERATIVE MODEL-ENHANCED HUMAN MOTION PREDICTION"),
    ("oev4KdikGjy", "FMix: Enhancing Mixed Sample Data Augmentation"),
    ("xfOVXyO_cwJ", "Empirical Frequentist Coverage of Deep Learning Uncertainty Quantification Procedures"),
    ("UiLl8yjh57", "Deep Reinforcement Learning For Wireless Scheduling with Multiclass Services"),
    ("hbzCPZEIUU", "Connecting Sphere Manifolds Hierarchically for Regularization"),
    ("F8xpAPm_ZKS", "Model-Free Counterfactual Credit Assignment"),
    ("ijVgDcvLmZ", "FSV: Learning to Factorize Soft Value Function for Cooperative Multi-Agent Reinforcement Learning"),
    ("aJLjjpi0Vty", "Collaborative Filtering with Smooth Reconstruction of the Preference Function"),
    ("MhTgnultR1K", "A Real-time Contribution Measurement Method for Participants in Federated Learning"),
    ("C4-QQ1EHNcI", "Expressive yet Tractable Bayesian Deep Learning via Subnetwork Inference"),
    ("LvJ8hLSusrv", "Gradient-based tuning of Hamiltonian Monte Carlo hyperparameters")
]

# ═══════════════════════════════════════════════════════════════════════════════

PDF_URL_TEMPLATE   = "https://openreview.net/pdf?id={paper_id}"
FORUM_URL_TEMPLATE = "https://openreview.net/forum?id={paper_id}"
PDF_DIR            = Path("papers")
MARKDOWN_DIR       = Path("papers_markdown")
DATASET_FILE       = Path("dataset.jsonl")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept": "application/pdf,*/*",
    "Referer": "https://openreview.net/",
}


# ─── PDF Download ─────────────────────────────────────────────────────────────

def download_pdf(paper_id: str, output_path: Path) -> bool:
    """Download a paper PDF from OpenReview. Returns True on success."""
    url = PDF_URL_TEMPLATE.format(paper_id=paper_id)
    try:
        r = requests.get(url, headers=HEADERS, stream=True, timeout=60)
        if r.status_code == 200 and r.headers.get("Content-Type", "").startswith("application/pdf"):
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            size = output_path.stat().st_size
            if size > 10_000:  # sanity check — a real PDF is > 10KB
                return True
            else:
                output_path.unlink(missing_ok=True)
    except requests.RequestException as e:
        print(f"         └─ Network error: {e}")
    return False


# ─── PDF → Markdown Conversion ────────────────────────────────────────────────

def convert_pdf_to_markdown(pdf_path: Path, output_dir: Path) -> Path | None:
    """
    Convert a PDF to Markdown using marker-pdf.
    
    Marker creates a subfolder per paper containing:
      - <name>.md         — The full paper in Markdown
      - images/           — Extracted figures with relative links in the .md
      - meta.json         — Marker's metadata (page count, etc.)
    
    Returns the path to the .md file, or None on failure.
    """
    try:
        result = subprocess.run(
            ["marker_single", str(pdf_path), "--output_dir", str(output_dir)],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            print(f"         └─ marker error: {result.stderr[:200]}")
            return None

        # Marker creates a subfolder named after the PDF (without extension)
        paper_name = pdf_path.stem
        paper_dir = output_dir / paper_name

        # Find the .md file inside the output folder
        md_files = list(paper_dir.glob("*.md")) if paper_dir.exists() else []
        if not md_files:
            # Some marker versions put it directly
            md_files = list(output_dir.glob(f"{paper_name}*.md"))

        return md_files[0] if md_files else None

    except subprocess.TimeoutExpired:
        print(f"         └─ marker timed out (>5min)")
        return None
    except FileNotFoundError:
        print(f"         └─ 'marker_single' not found. Install: pip install marker-pdf")
        return None


# ─── Main Pipeline ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ICLR 2021 Paper Pipeline: Download → Markdown → Dataset"
    )
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip PDF download, use existing files in papers/")
    parser.add_argument("--no-convert", action="store_true",
                        help="Download PDFs only, skip Markdown conversion")
    args = parser.parse_args()

    PDF_DIR.mkdir(exist_ok=True)
    MARKDOWN_DIR.mkdir(exist_ok=True)

    all_papers = (
        [(pid, title, "Accept") for pid, title in ACCEPTED_PAPERS] +
        [(pid, title, "Reject") for pid, title in REJECTED_PAPERS]
    )

    print()
    print("=" * 70)
    print("  ICLR 2021 Paper Pipeline")
    print(f"  {len(ACCEPTED_PAPERS)} accepted + {len(REJECTED_PAPERS)} rejected papers")
    print("=" * 70)

    dataset = []
    download_failures = []

    for i, (paper_id, title, decision) in enumerate(all_papers, 1):
        short = title[:55] + ("..." if len(title) > 55 else "")
        label = "ACCEPT" if decision == "Accept" else "REJECT"
        pdf_path = PDF_DIR / f"{paper_id}.pdf"
        forum_url = FORUM_URL_TEMPLATE.format(paper_id=paper_id)

        print(f"\n  [{i:2d}/{len(all_papers)}] [{label:6s}] {short}")
        print(f"         Forum: {forum_url}")

        # ── Step 1: PDF Download ────────────────────────────────────────
        if pdf_path.exists() and pdf_path.stat().st_size > 10_000:
            size_mb = pdf_path.stat().st_size / (1024 * 1024)
            print(f"         └─ PDF: cached ({size_mb:.1f} MB)")
        elif args.skip_download:
            print(f"         └─ PDF: NOT FOUND (--skip-download)")
            download_failures.append((paper_id, title))
            continue
        else:
            print(f"         └─ PDF: downloading...", end=" ", flush=True)
            if download_pdf(paper_id, pdf_path):
                size_mb = pdf_path.stat().st_size / (1024 * 1024)
                print(f"OK ({size_mb:.1f} MB)")
            else:
                print(f"FAILED")
                print(f"         └─ Manual download: https://openreview.net/pdf?id={paper_id}")
                download_failures.append((paper_id, title))
                continue
            time.sleep(1)  # be polite

        # ── Step 2: PDF → Markdown ──────────────────────────────────────
        md_path = None
        if not args.no_convert:
            # Check if already converted
            paper_md_dir = MARKDOWN_DIR / paper_id
            existing_md = list(paper_md_dir.glob("*.md")) if paper_md_dir.exists() else []
            if existing_md:
                md_path = existing_md[0]
                print(f"         └─ Markdown: cached ({md_path.name})")
            else:
                print(f"         └─ Markdown: converting with marker...", end=" ", flush=True)
                md_path = convert_pdf_to_markdown(pdf_path, MARKDOWN_DIR)
                if md_path:
                    print(f"OK ({md_path.name})")
                    # Check for images
                    img_dir = md_path.parent / "images"
                    if img_dir.exists():
                        n_imgs = len(list(img_dir.iterdir()))
                        print(f"         └─ Images: {n_imgs} figures extracted")
                else:
                    print(f"FAILED")

        # ── Build dataset entry ─────────────────────────────────────────
        entry = {
            "paper_id":    paper_id,
            "title":       title,
            "decision":    decision,
            "label":       label,
            "forum_url":   forum_url,
            "pdf_url":     PDF_URL_TEMPLATE.format(paper_id=paper_id),
            "pdf_path":    str(pdf_path),
            "markdown_path": str(md_path) if md_path else None,
        }
        dataset.append(entry)

    # ── Write dataset manifest ──────────────────────────────────────────────
    with open(DATASET_FILE, "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")

    # ── Summary ─────────────────────────────────────────────────────────────
    n_accepted = sum(1 for e in dataset if e["label"] == "ACCEPT")
    n_rejected = sum(1 for e in dataset if e["label"] == "REJECT")
    n_with_md  = sum(1 for e in dataset if e["markdown_path"])

    print()
    print("=" * 70)
    print(f"  Pipeline complete!")
    print(f"  PDFs:      {len(dataset)} saved to {PDF_DIR}/")
    print(f"  Markdown:  {n_with_md} converted to {MARKDOWN_DIR}/")
    print(f"  Manifest:  {DATASET_FILE}")
    print(f"  Accepted:  {n_accepted}  |  Rejected: {n_rejected}")

    if download_failures:
        print()
        print("  ⚠  The following papers could not be downloaded.")
        print("     Download them manually and place in papers/<id>.pdf,")
        print("     then re-run with: python download_papers.py --skip-download")
        print()
        for pid, t in download_failures:
            print(f"     curl -o papers/{pid}.pdf 'https://openreview.net/pdf?id={pid}'")

    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
