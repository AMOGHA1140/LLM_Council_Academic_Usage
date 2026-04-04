"""
Dynamic Council evaluation — Multi-persona review pipeline using Gemma 4.

For each paper in dataset.jsonl, runs a 3-step pipeline:
  1. Profile: Analyze the paper → generate 3 content-specific reviewer personas
  2. Review:  3 parallel async reviews using those personas
  3. Decide:  Area Chair consolidates reviews → final DECISION

Usage:
    python dynamic_council.py                          # evaluate all papers
    python dynamic_council.py --paper-id qO3lALCVMF    # single paper
    python dynamic_council.py --model gemma4:31b        # different model
"""
import json
import os
import asyncio
import argparse
import base64
from pathlib import Path
from openai import AsyncOpenAI

# Connect to the local Ollama server (OpenAI-compatible endpoint)
client = AsyncOpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)


# ─── Step 1: Profile ──────────────────────────────────────────────────────────

async def profile_paper(paper_text: str, model: str) -> list[dict]:
    """Analyze paper content → generate 3 tailored reviewer personas."""
    sys_msg = (
        "You are an intelligent Academic Profiler. Read the paper and return "
        "exactly a JSON array of 3 reviewer personas best suited to evaluate "
        "this exact methodology."
    )
    prompt = (
        "Read this paper excerpt and identify the 3 most critical dimensions "
        "it must be evaluated on.\n"
        "Output 3 specific Reviewer Personas tailored to this exact paper, "
        "including a detailed instruction role for each.\n"
        "Output strict JSON format:\n"
        '[{"persona": "...", "instruction": "..."}, ...]\n\n'
        f"Paper:\n{paper_text[:20000]}"
    )

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
    )
    content = response.choices[0].message.content
    try:
        clean = content.replace("```json", "").replace("```", "").strip()
        # Handle case where model wraps in extra text
        start = clean.index("[")
        end = clean.rindex("]") + 1
        return json.loads(clean[start:end])
    except Exception as e:
        print(f"         └─ Parse error ({e}), using defaults")
        return [
            {"persona": "Methodologist", "instruction": "Evaluate the empirical methodology and experimental design."},
            {"persona": "Theorist", "instruction": "Assess mathematical soundness and theoretical contributions."},
            {"persona": "Domain Expert", "instruction": "Evaluate novelty, significance, and real-world applicability."},
        ]


# ─── Step 2: Review ───────────────────────────────────────────────────────────

async def draft_review(persona_data: dict, paper_text: str, image_paths: list[Path], model: str) -> str:
    """Generate a single expert review from a persona's perspective."""
    persona = persona_data["persona"]
    instruction = persona_data["instruction"]

    content = [{"type": "text", "text": f"Provide your detailed critique of this paper:\n\n{paper_text[:60000]}"}]
    for img_path in image_paths:
        mime_type = "image/jpeg" if img_path.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
        b64_img = base64.b64encode(img_path.read_bytes()).decode('utf-8')
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{b64_img}"}
        })

    print(f"         └─ Agent '{persona}' starting review...", flush=True)
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"You are acting as: {persona}. {instruction}"},
            {"role": "user", "content": content}
        ],
        temperature=1.0,
        top_p=0.95,
    )
    print(f"         └─ Agent '{persona}' finished review.", flush=True)
    return f"--- Review by {persona} ---\n{response.choices[0].message.content}"


# ─── Step 3: Consolidate ──────────────────────────────────────────────────────

async def consolidate_reviews(reviews: list[str], model: str) -> str:
    """Area Chair synthesizes all reviews into a final decision."""
    sys_msg = (
        "You are the Area Chair for ICLR. Read the 3 expert reviews below. "
        "Synthesize their arguments, weigh the strengths and weaknesses, and "
        "output a final verdict formatted exactly as:\n"
        "`DECISION: Accept` or `DECISION: Reject`"
    )
    combined = "\n\n".join(reviews)

    print(f"         └─ Area Chair starting consolidation...", flush=True)
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": f"Expert reviews:\n\n{combined}"}
        ],
        temperature=1.0,
        top_p=0.95,
    )
    print(f"         └─ Area Chair finished consolidation.", flush=True)
    return response.choices[0].message.content


# ─── Full Pipeline ────────────────────────────────────────────────────────────

async def run_council(paper_text: str, image_paths: list[Path], model: str) -> tuple[str, list[str], str]:
    """Run the full 3-step council pipeline. Returns (decision_text, reviews, consolidation)."""

    # Step 1
    personas = await profile_paper(paper_text, model)
    persona_names = [p["persona"] for p in personas]
    print(f"→ Personas: {persona_names}")

    print(f"         Step 2: Drafting parallel reviews...")
    # Step 2 — parallel reviews
    reviews = await asyncio.gather(*(draft_review(p, paper_text, image_paths, model) for p in personas))
    print(f"         Step 2: Reviews complete ({len(reviews)} reviews)")

    print(f"         Step 3: Consolidating...")
    # Step 3 — consolidation
    consolidation = await consolidate_reviews(list(reviews), model)

    return consolidation, list(reviews), persona_names


def extract_decision(text: str) -> str | None:
    """Extract DECISION: Accept/Reject from text."""
    for line in reversed(text.split("\n")):
        line_upper = line.strip().upper()
        if "DECISION:" in line_upper:
            if "ACCEPT" in line_upper:
                return "Accept"
            elif "REJECT" in line_upper:
                return "Reject"
    return None


def load_dataset(path: str = "dataset.jsonl") -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry.get("markdown_path") and Path(entry["markdown_path"]).exists():
                entries.append(entry)
    return entries


async def main():
    parser = argparse.ArgumentParser(description="Dynamic Council LLM paper review")
    parser.add_argument("--model", default="gemma4:26b", help="Ollama model name")
    parser.add_argument("--paper-id", default=None, help="Evaluate a single paper by ID")
    parser.add_argument("--dataset", default="dataset.jsonl", help="Path to dataset manifest")
    parser.add_argument("--output", default="results_council.jsonl", help="Output results file")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    if args.paper_id:
        dataset = [e for e in dataset if e["paper_id"] == args.paper_id]

    if not dataset:
        print("No papers found. Run download_papers.py first.")
        return

    print(f"\n{'='*70}")
    print(f"  Dynamic Council Evaluation — {args.model}")
    print(f"  Papers: {len(dataset)} | Personas: 3 per paper")
    print(f"{'='*70}")

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
            print(f"\n  [{i}/{len(dataset)}] Skipping {entry['paper_id']} (already complete)")
            continue
        title = entry["title"][:50] + ("..." if len(entry["title"]) > 50 else "")
        gt = entry["label"]

        print(f"\n  [{i}/{len(dataset)}] [{gt:6s}] {title}")

        md_path = Path(entry["markdown_path"])
        paper_text = md_path.read_text(encoding="utf-8")
        image_paths = list(md_path.parent.glob("*.jpeg")) + list(md_path.parent.glob("*.jpg")) + list(md_path.parent.glob("*.png"))

        print(f"         Step 1: Profiling paper...", end=" ", flush=True)
        try:
            consolidation, reviews, personas = await run_council(paper_text, image_paths, args.model)

            prediction = extract_decision(consolidation)
            match = "✓" if prediction and prediction.upper() == entry["decision"].upper() else "✗"
            if match == "✓":
                correct += 1
            print(f"→ {prediction or 'NO DECISION'} [{match}]")

        except Exception as e:
            print(f"ERROR: {e}")
            consolidation, reviews, personas, prediction, match = str(e), [], [], None, "✗"

        result = {
            **entry,
            "model": args.model,
            "method": "dynamic_council",
            "personas": personas,
            "prediction": prediction,
            "correct": match == "✓",
            "reviews": reviews,
            "consolidation": consolidation,
        }
        results.append(result)

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
    asyncio.run(main())
