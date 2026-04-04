"""
Baseline evaluation: Zero-shot "Area Chair" review using Gemma 4 via Ollama.

Reads papers from dataset.jsonl, sends each to the LLM for a single-shot
review, and compares the LLM's decision against the ground-truth label.

Usage:
    python baseline_eval.py                          # evaluate all papers
    python baseline_eval.py --paper-id qO3lALCVMF    # evaluate a single paper
    python baseline_eval.py --model gemma4:31b        # use a different model
"""
import json
import os
import argparse
import base64
from pathlib import Path
from openai import OpenAI

# Connect to the local Ollama server (OpenAI-compatible endpoint)
client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

SYSTEM_PROMPT = (
    "You are the Area Chair for a top-tier Deep Learning conference (ICLR). "
    "Read the following paper provided in Markdown. Evaluate its:\n"
    "1. Mathematical soundness and theoretical contributions\n"
    "2. Empirical methodology and experimental rigor\n"
    "3. Novelty and significance to the field\n"
    "4. Clarity and quality of presentation\n\n"
    "Conclude your review with a final decision formatted exactly as:\n"
    "`DECISION: Accept` or `DECISION: Reject`"
)


def load_dataset(dataset_path: str = "dataset.jsonl") -> list[dict]:
    """Load the paper dataset manifest."""
    entries = []
    with open(dataset_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry.get("markdown_path") and Path(entry["markdown_path"]).exists():
                entries.append(entry)
    return entries


def review_paper(paper_text: str, image_paths: list[Path], model: str) -> str:
    """Send a paper to the LLM for review."""
    content = [{"type": "text", "text": f"Review this paper:\n\n{paper_text[:60000]}"}]
    for img_path in image_paths:
        mime_type = "image/jpeg" if img_path.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
        b64_img = base64.b64encode(img_path.read_bytes()).decode('utf-8')
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{b64_img}"}
        })

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ],
        temperature=1.0,
        top_p=0.95,
    )
    return response.choices[0].message.content


def extract_decision(review_text: str) -> str | None:
    """Extract the DECISION: Accept/Reject from the review text."""
    for line in reversed(review_text.split("\n")):
        line_upper = line.strip().upper()
        if "DECISION:" in line_upper:
            if "ACCEPT" in line_upper:
                return "Accept"
            elif "REJECT" in line_upper:
                return "Reject"
    return None


def main():
    parser = argparse.ArgumentParser(description="Baseline LLM paper review")
    parser.add_argument("--model", default="gemma4:26b", help="Ollama model name")
    parser.add_argument("--paper-id", default=None, help="Evaluate a single paper by ID")
    parser.add_argument("--dataset", default="dataset.jsonl", help="Path to dataset manifest")
    parser.add_argument("--output", default="results_baseline.jsonl", help="Output results file")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    if args.paper_id:
        dataset = [e for e in dataset if e["paper_id"] == args.paper_id]

    if not dataset:
        print("No papers found. Run download_papers.py first.")
        return

    print(f"\n{'='*70}")
    print(f"  Baseline Evaluation — {args.model}")
    print(f"  Papers: {len(dataset)}")
    print(f"{'='*70}\n")

    results = []
    completed_ids = set()

    if os.path.exists(args.output):
        with open(args.output, "r") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    res = json.loads(line)
                    results.append(res)
                    if "paper_id" in res:
                        completed_ids.add(res["paper_id"])
                except json.JSONDecodeError:
                    pass

    correct = sum(1 for r in results if r.get("correct"))

    for i, entry in enumerate(dataset, 1):
        if entry["paper_id"] in completed_ids:
            print(f"  [{i}/{len(dataset)}] Skipping {entry['paper_id']} (already complete)")
            continue

        title = entry["title"][:50] + ("..." if len(entry["title"]) > 50 else "")
        gt = entry["label"]
        print(f"  [{i}/{len(dataset)}] [{gt:6s}] {title}")
        print(f"         Reviewing...", end=" ", flush=True)

        md_path = Path(entry["markdown_path"])
        paper_text = md_path.read_text(encoding="utf-8")
        image_paths = list(md_path.parent.glob("*.jpeg")) + list(md_path.parent.glob("*.jpg")) + list(md_path.parent.glob("*.png"))
        review = review_paper(paper_text, image_paths, args.model)
        prediction = extract_decision(review)

        match = "✓" if prediction and prediction.upper() == entry["decision"].upper() else "✗"
        if match == "✓":
            correct += 1

        print(f"→ {prediction or 'NO DECISION'} [{match}]")

        result = {
            **entry,
            "model": args.model,
            "prediction": prediction,
            "correct": match == "✓",
            "review": review,
        }
        results.append(result)

        # Write incrementally
        with open(args.output, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    # ── Summary ─────────────────────────────────────────────────────────────
    n = len(results)
    print(f"\n{'='*70}")
    print(f"  Accuracy: {correct}/{n} ({100*correct/n:.0f}%)")
    print(f"  Results:  {args.output}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
