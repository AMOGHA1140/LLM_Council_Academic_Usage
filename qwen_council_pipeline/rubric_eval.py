"""
Calibrated Rubric Council — ICLR-style multi-reviewer evaluation pipeline.

For each paper in dataset.jsonl, runs a 6-phase pipeline:
  Phase 0: Preprocess  — strip references/appendix, clean markdown
  Phase 1: Profile     — generate 3 domain-specific reviewer personas
  Phase 2: Review      — independent rubric reviews with ICLR verbal anchors
  Phase 3: Discuss     — reviewers see each other's reviews, revise scores
  Phase 4: Extract     — parse verbal anchors → numeric scores (pure Python)
  Phase 5: Aggregate   — confidence-weighted average → threshold decision (pure Python)

The LLM never makes the Accept/Reject decision — it only scores on described
scales. The final decision is a mechanical threshold on aggregated scores.

Usage:
    python rubric_eval.py                              # evaluate all papers
    python rubric_eval.py --paper-id zv-typ1gPxA       # single paper
    python rubric_eval.py --threshold 5.0              # adjust decision threshold
    python rubric_eval.py --output results_rubric.jsonl
"""
import json
import os
import re
import asyncio
import argparse
import textwrap
from pathlib import Path
from openai import AsyncOpenAI

# ─── Model constants ──────────────────────────────────────────────────────────
PROFILER_MODEL = "gemma4:26b"
REVIEWER_MODELS = ["gemma4:26b", "qwen3.5:27b", "gemma4:26b"]

# ─── Ollama client ────────────────────────────────────────────────────────────
client = AsyncOpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def section(title: str, width: int = 72) -> str:
    bar = "─" * width
    return f"\n{bar}\n  {title}\n{bar}"


def log_block(label: str, content: str, indent: int = 4) -> None:
    pad = " " * indent
    wrapped = textwrap.indent(content.strip(), pad + "│ ")
    print(f"\n{pad}┌── {label} {'─'*(60 - len(label))}")
    print(wrapped)
    print(f"{pad}└──{'─'*63}")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 0 — Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_paper(raw_markdown: str) -> str:
    """Strip references, appendix, image tags; collapse whitespace."""
    text = raw_markdown

    # Remove everything after References/Bibliography section
    text = re.split(r'^#+\s*(References|Bibliography)\b', text, maxsplit=1, flags=re.MULTILINE | re.IGNORECASE)[0]

    # Remove everything after Appendix/Supplementary section
    text = re.split(r'^#+\s*(Appendix|Supplementary)\b', text, maxsplit=1, flags=re.MULTILINE | re.IGNORECASE)[0]

    # Remove image markdown tags but keep figure captions
    # Match ![caption](path) — keep caption text if it looks like a caption
    text = re.sub(r'!\[([^\]]*)\]\([^)]*\)', r'\1', text)

    # Collapse repeated blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Dynamic Profiling
# ═══════════════════════════════════════════════════════════════════════════════

async def profile_paper(paper_text: str) -> list[dict]:
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
        '[{"persona": "...", "instruction": "...", "focus_area": "soundness|contribution|clarity"}, ...]\n\n'
        f"Paper:\n{paper_text[:8000]}"
    )

    response = await client.chat.completions.create(
        model=PROFILER_MODEL,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
    )
    content = response.choices[0].message.content
    try:
        clean = content.replace("```json", "").replace("```", "").strip()
        start = clean.index("[")
        end = clean.rindex("]") + 1
        return json.loads(clean[start:end])
    except Exception as e:
        print(f"         └─ Profile parse error ({e}), using defaults")
        return [
            {"persona": "Methodologist", "instruction": "Evaluate the empirical methodology and experimental design.", "focus_area": "soundness"},
            {"persona": "Theorist", "instruction": "Assess mathematical soundness and theoretical contributions.", "focus_area": "contribution"},
            {"persona": "Domain Expert", "instruction": "Evaluate novelty, significance, and clarity of presentation.", "focus_area": "clarity"},
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Independent Rubric Review
# ═══════════════════════════════════════════════════════════════════════════════

RUBRIC_PROMPT = """\
Review this paper carefully. For each dimension below, first provide your detailed
analysis and reasoning, then select the SINGLE most appropriate verbal anchor.

== OVERALL RECOMMENDATION ==
Select exactly one:
- "10: Top 5% of accepted papers, seminal paper"
- "8: Top 15-20% of accepted papers, clear accept"
- "6: Marginally above acceptance threshold"
- "5: Marginally below acceptance threshold"
- "3: Clear reject, significant technical or conceptual issues"
- "1: Trivially wrong, irrelevant, or already known"

== SOUNDNESS (technical correctness) ==
Select exactly one:
- "4: Excellent — claims well-supported, proofs correct, experiments rigorous"
- "3: Good — minor issues but overall technically sound"
- "2: Fair — some concerns about correctness of claims or methodology"
- "1: Poor — major flaws in reasoning, proofs, or experimental design"

== CONTRIBUTION (novelty and significance) ==
Select exactly one:
- "4: Excellent — significant advance, likely to influence future work"
- "3: Good — interesting contribution with clear value"
- "2: Fair — incremental or limited contribution"
- "1: Poor — no significant contribution or already well-known"

== CLARITY (quality of writing and presentation) ==
Select exactly one:
- "4: Excellent — well-written, easy to follow, well-organized"
- "3: Good — generally clear with minor issues"
- "2: Fair — important parts hard to follow or poorly explained"
- "1: Poor — very difficult to understand"

== CONFIDENCE ==
Select exactly one:
- "5: Absolutely certain, familiar with all related work"
- "4: Confident, but not absolutely certain"
- "3: Fairly confident"
- "2: Willing to defend, but possible I missed something"
- "1: Not confident at all, just an educated guess"

After your full analysis, output your selections on separate lines in this EXACT format:
OVERALL: <your selected verbal anchor>
SOUNDNESS: <your selected verbal anchor>
CONTRIBUTION: <your selected verbal anchor>
CLARITY: <your selected verbal anchor>
CONFIDENCE: <your selected verbal anchor>

Paper:
{paper_text}
"""


async def rubric_review(persona_data: dict, paper_text: str, model: str, reviewer_idx: int) -> dict:
    """Generate an independent rubric review from one persona."""
    persona = persona_data["persona"]
    instruction = persona_data["instruction"]

    sys_msg = (
        f"You are acting as: {persona}. {instruction}\n\n"
        "You are reviewing a paper submitted to ICLR, a top machine learning conference. "
        "Evaluate the paper on the ICLR rubric dimensions using the EXACT verbal anchors provided. "
        "You MUST select one of the provided options verbatim for each dimension."
    )
    user_msg = RUBRIC_PROMPT.format(paper_text=paper_text[:40000])

    print(f"  │  ├─ [R{reviewer_idx+1}] {persona} reviewing ({model})...", flush=True)
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.7,
        top_p=0.95,
    )
    raw = response.choices[0].message.content
    print(f"  │  ├─ [R{reviewer_idx+1}] {persona} finished review.", flush=True)

    scores = extract_scores(raw)
    return {
        "persona": persona,
        "model": model,
        "raw_review": raw,
        "scores": scores,
        "parse_success": scores is not None and len(scores) >= 4,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3 — Discussion Round (Score Revision)
# ═══════════════════════════════════════════════════════════════════════════════

def format_review_summary(review: dict) -> str:
    """Format a review for inclusion in the discussion prompt."""
    scores = review.get("scores") or {}
    score_lines = "\n".join(f"  {k.upper()}: {v}" for k, v in scores.items())
    # Take last ~2000 chars of the review to get the reasoning + scores section
    raw = review["raw_review"]
    # Find the scoring section
    summary_start = raw.rfind("OVERALL:")
    if summary_start > 0:
        reasoning = raw[:summary_start].strip()[-1500:]
        scoring = raw[summary_start:]
    else:
        reasoning = raw[-2000:]
        scoring = ""

    return f"Key arguments:\n{reasoning}\n\nScores:\n{score_lines}\n{scoring}"


async def discussion_round(
    reviewer_idx: int,
    persona_data: dict,
    own_review: dict,
    other_reviews: list[dict],
    model: str,
) -> dict:
    """One reviewer reads all reviews and revises their scores."""
    persona = persona_data["persona"]
    instruction = persona_data["instruction"]

    # Build the other reviews section
    other_sections = []
    for i, rev in enumerate(other_reviews):
        other_sections.append(
            f"=== Reviewer: {rev['persona']} ===\n"
            f"{format_review_summary(rev)}"
        )
    others_text = "\n\n".join(other_sections)

    own_summary = format_review_summary(own_review)

    sys_msg = (
        f"You are acting as: {persona}. {instruction}\n\n"
        "You are participating in the reviewer discussion phase for an ICLR submission. "
        "You have already submitted your initial review. Now you are reading the other "
        "reviewers' assessments and may revise your scores."
    )
    user_msg = (
        f"YOUR initial review and scores:\n{own_summary}\n\n"
        f"OTHER REVIEWERS' assessments:\n\n{others_text}\n\n"
        "Now, considering their arguments:\n"
        "1. Identify any points they raised that you missed or underweighted.\n"
        "2. Identify any points where you disagree with their assessment and explain why.\n"
        "3. State your REVISED scores (or confirm your original scores if unchanged).\n\n"
        "For each score you change, briefly explain why.\n\n"
        "Output your FINAL scores in this EXACT format:\n"
        "OVERALL: <verbal anchor from the ICLR scale>\n"
        "SOUNDNESS: <verbal anchor>\n"
        "CONTRIBUTION: <verbal anchor>\n"
        "CLARITY: <verbal anchor>\n"
        "CONFIDENCE: <verbal anchor>"
    )

    print(f"  │  ├─ [R{reviewer_idx+1}] {persona} discussing ({model})...", flush=True)
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.7,
        top_p=0.95,
    )
    raw = response.choices[0].message.content
    print(f"  │  ├─ [R{reviewer_idx+1}] {persona} finished discussion.", flush=True)

    revised_scores = extract_scores(raw)
    original_scores = own_review.get("scores") or {}

    # Compute score changes
    score_changes = {}
    if revised_scores and original_scores:
        for key in revised_scores:
            if key in original_scores and revised_scores[key] != original_scores[key]:
                score_changes[key] = revised_scores[key] - original_scores[key]

    return {
        "persona": persona,
        "model": model,
        "raw_discussion": raw,
        "revised_scores": revised_scores,
        "score_changes": score_changes,
        "parse_success": revised_scores is not None and len(revised_scores) >= 4,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4 — Score Extraction (Pure Python)
# ═══════════════════════════════════════════════════════════════════════════════

# Fallback mappings: verbal anchor text → numeric score
OVERALL_ANCHORS = {
    "top 5%": 10, "seminal": 10,
    "top 15": 8, "clear accept": 8,
    "marginally above": 6, "above acceptance": 6,
    "marginally below": 5, "below acceptance": 5,
    "clear reject": 3, "significant technical": 3, "significant conceptual": 3,
    "trivially wrong": 1, "irrelevant": 1, "already known": 1,
}

FOUR_SCALE_ANCHORS = {
    "excellent": 4, "good": 3, "fair": 2, "poor": 1,
}

CONFIDENCE_ANCHORS = {
    "absolutely certain": 5, "confident, but not": 4, "fairly confident": 3,
    "willing to defend": 2, "not confident": 1, "educated guess": 1,
}


def extract_scores(review_text: str) -> dict | None:
    """Extract numeric scores from verbal anchor responses."""
    if not review_text:
        return None

    scores = {}

    # Primary: regex for "DIMENSION: <number>:" pattern
    patterns = {
        'overall':      r'OVERALL:\s*"?(\d+)\s*:',
        'soundness':    r'SOUNDNESS:\s*"?(\d+)\s*:',
        'contribution': r'CONTRIBUTION:\s*"?(\d+)\s*:',
        'clarity':      r'CLARITY:\s*"?(\d+)\s*:',
        'confidence':   r'CONFIDENCE:\s*"?(\d+)\s*:',
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, review_text, re.IGNORECASE)
        if match:
            scores[key] = int(match.group(1))

    # Fallback: search for verbal anchor text near the dimension label
    if 'overall' not in scores:
        # Search for anchor text after "OVERALL"
        overall_match = re.search(r'OVERALL:?\s*(.*)', review_text, re.IGNORECASE)
        if overall_match:
            line = overall_match.group(1).lower()
            for anchor, val in OVERALL_ANCHORS.items():
                if anchor in line:
                    scores['overall'] = val
                    break

    for dim in ['soundness', 'contribution', 'clarity']:
        if dim not in scores:
            dim_match = re.search(rf'{dim}:?\s*(.*)', review_text, re.IGNORECASE)
            if dim_match:
                line = dim_match.group(1).lower()
                for anchor, val in FOUR_SCALE_ANCHORS.items():
                    if anchor in line:
                        scores[dim] = val
                        break

    if 'confidence' not in scores:
        conf_match = re.search(r'CONFIDENCE:?\s*(.*)', review_text, re.IGNORECASE)
        if conf_match:
            line = conf_match.group(1).lower()
            for anchor, val in CONFIDENCE_ANCHORS.items():
                if anchor in line:
                    scores['confidence'] = val
                    break

    return scores if scores else None


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5 — Aggregation & Decision (Pure Python)
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_and_decide(reviewer_scores: list[dict], threshold: float = 5.5) -> dict:
    """Aggregate scores from multiple reviewers via confidence-weighted average."""
    valid_scores = [s for s in reviewer_scores if s]
    if not valid_scores:
        return {"decision": None, "error": "No valid scores to aggregate"}

    weights = [s.get('confidence', 3) for s in valid_scores]
    total_weight = sum(weights)

    dims = {}
    for dim in ['overall', 'soundness', 'contribution', 'clarity']:
        vals = [(s.get(dim, 5 if dim == 'overall' else 2), w) for s, w in zip(valid_scores, weights)]
        dims[f'avg_{dim}'] = round(sum(v * w for v, w in vals) / total_weight, 2)

    # Decision: accept if overall >= threshold AND no critical dimension below 2.0
    decision = "Accept" if (
        dims['avg_overall'] >= threshold
        and dims['avg_soundness'] >= 2.0
        and dims['avg_contribution'] >= 2.0
    ) else "Reject"

    return {
        **dims,
        "decision": decision,
        "threshold": threshold,
        "weights": weights,
        "individual_scores": valid_scores,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Full Pipeline Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

async def run_rubric_council(paper_text: str, threshold: float) -> dict:
    """Run the full 6-phase rubric council pipeline on a single paper."""

    # Phase 0: Preprocess
    print(section("Phase 0 │ Preprocessing"))
    cleaned = preprocess_paper(paper_text)
    print(f"  Original: {len(paper_text)} chars → Cleaned: {len(cleaned)} chars")

    # Phase 1: Profile
    print(section("Phase 1 │ Dynamic Profiling"))
    personas = await profile_paper(cleaned)
    for idx, p in enumerate(personas):
        print(f"  R{idx+1}: {p['persona']} — {p.get('instruction', '')[:60]}...")

    # Phase 2: Independent rubric reviews
    print(section("Phase 2 │ Independent Rubric Reviews"))
    models = REVIEWER_MODELS[:len(personas)]
    # Pad if fewer than 3 personas
    while len(models) < len(personas):
        models.append(REVIEWER_MODELS[0])

    review_tasks = [
        rubric_review(persona, cleaned, model, idx)
        for idx, (persona, model) in enumerate(zip(personas, models))
    ]
    initial_reviews = list(await asyncio.gather(*review_tasks))

    for rev in initial_reviews:
        scores_str = json.dumps(rev.get("scores") or {})
        status = "OK" if rev["parse_success"] else "PARSE FAILED"
        print(f"  │  └─ {rev['persona']}: {scores_str} [{status}]")

    # Phase 3: Discussion round
    print(section("Phase 3 │ Reviewer Discussion"))
    discussion_tasks = []
    for idx, (persona, model, own_review) in enumerate(zip(personas, models, initial_reviews)):
        others = [r for j, r in enumerate(initial_reviews) if j != idx]
        discussion_tasks.append(
            discussion_round(idx, persona, own_review, others, model)
        )
    discussion_reviews = list(await asyncio.gather(*discussion_tasks))

    for disc in discussion_reviews:
        changes = disc.get("score_changes", {})
        if changes:
            changes_str = ", ".join(f"{k}: {v:+d}" for k, v in changes.items())
            print(f"  │  └─ {disc['persona']}: revised ({changes_str})")
        else:
            print(f"  │  └─ {disc['persona']}: no changes")

    # Phase 4: Extract final scores
    print(section("Phase 4 │ Score Extraction"))
    final_scores = []
    for disc, init_rev in zip(discussion_reviews, initial_reviews):
        # Prefer post-discussion scores; fall back to initial if discussion parse failed
        if disc["parse_success"] and disc["revised_scores"]:
            final_scores.append(disc["revised_scores"])
            print(f"  Using post-discussion scores for {disc['persona']}")
        elif init_rev["parse_success"] and init_rev["scores"]:
            final_scores.append(init_rev["scores"])
            print(f"  Falling back to initial scores for {disc['persona']} (discussion parse failed)")
        else:
            print(f"  WARNING: No valid scores for {disc['persona']}")

    # Phase 5: Aggregate and decide
    print(section("Phase 5 │ Aggregation & Decision"))
    aggregation = aggregate_and_decide(final_scores, threshold)
    print(f"  Avg Overall:      {aggregation.get('avg_overall', '?')}")
    print(f"  Avg Soundness:    {aggregation.get('avg_soundness', '?')}")
    print(f"  Avg Contribution: {aggregation.get('avg_contribution', '?')}")
    print(f"  Avg Clarity:      {aggregation.get('avg_clarity', '?')}")
    print(f"  Threshold:        {threshold}")
    print(f"  Decision:         {aggregation.get('decision', '?')}")

    return {
        "personas": [{"persona": p["persona"], "instruction": p.get("instruction", ""), "focus_area": p.get("focus_area", "")} for p in personas],
        "initial_reviews": initial_reviews,
        "discussion_reviews": discussion_reviews,
        "final_scores": final_scores,
        "aggregation": aggregation,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset(path: str = "dataset.jsonl") -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry.get("markdown_path") and Path(entry["markdown_path"]).exists():
                entries.append(entry)
    return entries


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser(
        description="Calibrated Rubric Council — ICLR-style LLM paper review"
    )
    parser.add_argument("--paper-id", default=None,
                        help="Evaluate a single paper by ID")
    parser.add_argument("--dataset", default="dataset.jsonl",
                        help="Path to dataset manifest")
    parser.add_argument("--output", default="results_rubric.jsonl",
                        help="Output results file")
    parser.add_argument("--threshold", type=float, default=5.5,
                        help="Decision threshold for avg overall score (default: 5.5)")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    if args.paper_id:
        dataset = [e for e in dataset if e["paper_id"] == args.paper_id]

    if not dataset:
        print("No papers found. Run download_papers.py first.")
        return

    # ── Resume: skip already-completed papers ─────────────────────────────────
    results: list[dict] = []
    completed_ids: set[str] = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    res = json.loads(line)
                    results.append(res)
                    if "paper_id" in res:
                        completed_ids.add(res["paper_id"])
                except json.JSONDecodeError:
                    pass

    correct = sum(1 for r in results if r.get("correct"))

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"  ██  Calibrated Rubric Council")
    print(f"  ██  Papers: {len(dataset)}  │  Threshold: {args.threshold}")
    print(f"  ██  Reviewer models: {', '.join(REVIEWER_MODELS)}")
    print(f"  ██  Profiler: {PROFILER_MODEL}")
    print(f"{'═'*72}")

    for i, entry in enumerate(dataset, 1):
        paper_id = entry["paper_id"]

        if paper_id in completed_ids:
            print(f"\n  [{i}/{len(dataset)}] SKIP  {paper_id}  (already in output)")
            continue

        title = entry["title"][:60] + ("..." if len(entry["title"]) > 60 else "")
        gt = entry.get("decision", entry.get("label", "?"))

        print(f"\n\n{'▓'*72}")
        print(f"  [{i}/{len(dataset)}]  Paper  : {title}")
        print(f"             ID      : {paper_id}")
        print(f"             Truth   : {gt}")
        print(f"{'▓'*72}")

        md_path = Path(entry["markdown_path"])
        paper_text = md_path.read_text(encoding="utf-8")

        try:
            council_output = await run_rubric_council(paper_text, args.threshold)
            prediction = council_output["aggregation"].get("decision")
            match = "✓" if prediction and prediction.upper() == gt.upper() else "✗"
            if match == "✓":
                correct += 1

            print(f"\n  {'━'*68}")
            print(f"  RESULT │ Prediction: {prediction or 'NONE'} │ Ground Truth: {gt} │ {match}")
            print(f"         │ Avg Overall: {council_output['aggregation'].get('avg_overall', '?')}")
            print(f"  {'━'*68}\n")

        except Exception as e:
            import traceback
            print(f"\n  ✗ ERROR processing {paper_id}:")
            traceback.print_exc()
            council_output = {"error": str(e)}
            prediction, match = None, "✗"

        result = {
            **entry,
            "method": "rubric_council",
            "profiler_model": PROFILER_MODEL,
            "reviewer_models": REVIEWER_MODELS,
            "threshold": args.threshold,
            "prediction": prediction,
            "correct": match == "✓",
            **council_output,
        }
        results.append(result)

        # Incremental write
        with open(args.output, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    # ── Final summary ─────────────────────────────────────────────────────────
    n = len(results)
    print(f"\n{'═'*72}")
    print(f"  Rubric Council Complete")
    print(f"  Accuracy : {correct}/{n} ({100*correct/n:.1f}%)" if n else "  No results.")
    print(f"  Output   : {args.output}")
    print(f"{'═'*72}\n")


if __name__ == "__main__":
    asyncio.run(main())
