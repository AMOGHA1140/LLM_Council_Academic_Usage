"""
Debate Council — Adversarial multi-agent review pipeline.

Architecture:
  Phase 1 │ Dynamic Profiling  → Gemma 4:26b identifies the domain and generates
           │                     two adversarial personas: Proponent + Critic.
  Phase 2 │ Independent Reads  → Proponent (Gemma 4:26b) and Critic (Qwen3:latest)
           │                     each read the full paper and produce structured
           │                     logical analyses independently.
  Phase 3 │ Cross-Examination  → Up to 3 debate rounds. Each agent receives the
           │                     other's analysis and must attack its reasoning.
           │                     Each agent may concede or stand firm.
  Phase 4 │ Area Chair         → Gemma 4:26b acts as judge: reviews the full
           │                     debate transcript, declares winners per claim,
           │                     and issues DECISION: Accept / Reject.

Models:
  Proponent → gemma4:26b   (strong generalist, tends toward validation)
  Critic    → qwen3:latest (aggressive skeptic, different architecture)
  Chair     → gemma4:26b   (strongest available model as final arbiter)

Usage:
    python debate_council.py                          # evaluate all papers
    python debate_council.py --paper-id j0uePNuoBho   # single paper
    python debate_council.py --rounds 2                # fewer debate rounds
    python debate_council.py --output results_debate.jsonl
"""

import json
import os
import asyncio
import argparse
import textwrap
import re
from pathlib import Path
from openai import AsyncOpenAI

# ─── Model constants ──────────────────────────────────────────────────────────
PROPONENT_MODEL = "gemma4:26b"
CRITIC_MODEL    = "qwen3.5:35b"
CHAIR_MODEL     = "gemma4:26b"

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

def subsection(label: str) -> str:
    return f"\n  ┌─ {label}"

def log_block(label: str, content: str, indent: int = 4) -> None:
    """Pretty-print a labelled block of text to stdout."""
    pad = " " * indent
    wrapped = textwrap.indent(content.strip(), pad + "│ ")
    print(f"\n{pad}┌── {label} {'─'*(60 - len(label))}")
    print(wrapped)
    print(f"{pad}└──{'─'*63}")


def extract_json_array(text: str) -> list | None:
    """Best-effort extraction of a JSON array embedded in model output."""
    clean = text.replace("```json", "").replace("```", "").strip()
    try:
        start = clean.index("[")
        end   = clean.rindex("]") + 1
        return json.loads(clean[start:end])
    except Exception:
        return None


def extract_decision(text: str) -> str | None:
    """Extract DECISION: Accept/Reject from any position in text."""
    for line in reversed(text.split("\n")):
        u = line.strip().upper()
        if "DECISION:" in u:
            if "ACCEPT" in u:
                return "Accept"
            if "REJECT" in u:
                return "Reject"
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Dynamic Adversarial Profiling
# ═══════════════════════════════════════════════════════════════════════════════

async def profile_paper(paper_text: str) -> dict:
    """
    Feed the abstract + intro to Gemma. It returns a JSON object with:
      - domain: the precise technical domain (e.g. "Diffusion Models for Text-to-Image")
      - proponent_persona: name + mandate for the advocate role
      - critic_persona:    name + mandate for the adversarial role

    The personas are grounded in the paper's actual methodology so the debate
    stays technically meaningful.
    """
    sys_msg = (
        "You are an Academic Adversarial Setup Specialist. "
        "Your job is NOT to review the paper — it is to design a fair but adversarial "
        "peer-review environment tailored to this specific paper's methodology."
    )
    prompt = (
        "Read the opening sections of this paper carefully.\n\n"
        "Then output a single JSON object (no prose, no markdown fences) with these fields:\n"
        "  \"domain\": a precise 3-10 word technical domain label\n"
        "  \"proponent\": {\n"
        "    \"name\": a specific expert title (e.g. 'Diffusion Model Theorist'),\n"
        "    \"mandate\": a one-sentence instruction telling this reviewer to build\n"
        "                 the STRONGEST possible case for the paper's novelty and soundness\n"
        "  },\n"
        "  \"critic\": {\n"
        "    \"name\": a specific expert title (e.g. 'Evaluation Methodology Auditor'),\n"
        "    \"mandate\": a one-sentence instruction telling this reviewer to find every\n"
        "                 fatal flaw: unproven assumptions, data leakage, missing baselines,\n"
        "                 statistical errors, scope inflation\n"
        "  }\n\n"
        "The titles and mandates must be laser-focused on THIS paper's specific method.\n\n"
        f"Paper (first 6000 chars):\n{paper_text[:6000]}"
    )

    print(subsection("Calling profiler (gemma4:26b)..."), flush=True)
    resp = await client.chat.completions.create(
        model=PROPONENT_MODEL,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.6,
    )
    raw = resp.choices[0].message.content
    try:
        clean = raw.replace("```json","").replace("```","").strip()
        start = clean.index("{")
        end   = clean.rindex("}") + 1
        profile = json.loads(clean[start:end])
        assert "proponent" in profile and "critic" in profile
        return profile
    except Exception as e:
        print(f"  └─ Profile parse error ({e}), using generic adversarial roles.", flush=True)
        return {
            "domain": "Machine Learning Research",
            "proponent": {
                "name": "Domain Proponent",
                "mandate": "Build the strongest possible case for this paper's novelty and technical soundness."
            },
            "critic": {
                "name": "Methodological Critic",
                "mandate": "Identify every fatal flaw, unproven assumption, and methodological weakness in this paper."
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Independent Logical Extraction
# ═══════════════════════════════════════════════════════════════════════════════

EXTRACTION_SYSTEM = """You are {name}.

Your mandate: {mandate}

You are doing a careful, independent first-pass read of a research paper.
Your goal is to extract the key logical claims and your analysis of each.

Output a JSON array. Each element represents one significant claim or aspect you analysed.
Each element must have at minimum:
  "claim"    : the specific assertion the paper is making (quote or paraphrase closely)
  "evidence" : how the authors attempt to back it up (method, experiment, theorem, etc.)
  "verdict"  : your step-by-step reasoning on whether the evidence actually supports the claim,
               ending with a one-word tag: [VALID], [WEAK], or [INVALID]

You may add any additional fields you find useful (e.g. "concern", "strength", "citation_gap").
Aim for 4-7 elements covering the paper's most consequential claims.
Do NOT pad with trivial observations.

Output only the JSON array. No prose before or after.
"""

async def extract_analysis(persona_name: str, mandate: str,
                            paper_text: str, model: str,
                            role_label: str) -> list[dict]:
    """Run Phase 2 independent analysis for one agent."""
    sys_msg = EXTRACTION_SYSTEM.format(name=persona_name, mandate=mandate)
    user_msg = (
        f"Analyse this paper thoroughly.\n\n"
        f"Paper:\n{paper_text}"
    )

    print(f"  │  ├─ [{role_label}] {persona_name} reading paper ({model})...", flush=True)
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.8,
        top_p=0.95,
    )
    raw = resp.choices[0].message.content
    print(f"  │  ├─ [{role_label}] {persona_name} finished extraction.", flush=True)

    parsed = extract_json_array(raw)
    if parsed is None:
        print(f"  │  └─ [{role_label}] WARNING: could not parse JSON, storing raw text.")
        return [{"claim": "PARSE_ERROR", "evidence": "", "verdict": raw}]
    return parsed


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3 — Cross-Examination (Debate Rounds)
# ═══════════════════════════════════════════════════════════════════════════════

DEBATE_SYSTEM = """You are {name} — {mandate}

You are now in Round {round_num} of a formal peer-review debate.

You have just read your opponent's analysis. Your task:
1. For each point your opponent makes, explicitly evaluate the LOGIC of their argument.
   - If the logic is flawed, state exactly why.
   - If you concede the point, say so clearly with "CONCEDE:" prefix.
   - If you rebut the point, say so with "REBUT:" prefix and provide counter-evidence from the paper.
2. After addressing your opponent's points, output a JSON object with:
   "position_changes": a list of claims from YOUR OWN previous analysis
                       where your view changed, as: {{"claim": "...", "original_verdict": "...",
                       "revised_verdict": "...", "reason": "..."}}
   "standing_firm":    a list of your claims you are NOT revising, each with a brief rationale.
   "round_summary":    1-2 sentence summary of how the debate shifted your thinking this round.

Do NOT simply restate your own original analysis. Engage directly with the opponent's text.
"""

async def debate_round(
    round_num:     int,
    agent_name:    str,
    agent_mandate: str,
    agent_model:   str,
    agent_analysis: list[dict],
    opponent_analysis: list[dict],
    role_label:    str,
) -> dict:
    """
    One agent responds to the opponent's analysis.
    Returns a dict with the full rebuttal text and extracted structured response.
    """
    sys_msg = DEBATE_SYSTEM.format(
        name=agent_name,
        mandate=agent_mandate,
        round_num=round_num,
    )
    user_msg = (
        f"YOUR previous analysis:\n"
        f"{json.dumps(agent_analysis, indent=2)}\n\n"
        f"OPPONENT's analysis to rebut:\n"
        f"{json.dumps(opponent_analysis, indent=2)}"
    )

    print(f"  │  ├─ [{role_label}] {agent_name} composing Round {round_num} rebuttal...", flush=True)
    resp = await client.chat.completions.create(
        model=agent_model,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.9,
        top_p=0.95,
    )
    raw = resp.choices[0].message.content
    print(f"  │  ├─ [{role_label}] {agent_name} finished Round {round_num} rebuttal.", flush=True)

    # Try to extract the structured JSON block from the rebuttal
    structured = None
    try:
        clean = raw.replace("```json","").replace("```","")
        start = clean.index("{")    # first { = outermost JSON object
        end   = clean.rindex("}") + 1
        structured = json.loads(clean[start:end])
    except Exception:
        structured = {"round_summary": "Could not parse structured response.", "position_changes": [], "standing_firm": []}

    return {
        "round":      round_num,
        "agent":      agent_name,
        "role":       role_label,
        "rebuttal":   raw,
        "structured": structured,
    }


async def run_debate(
    proponent_name:    str,
    proponent_mandate: str,
    critic_name:       str,
    critic_mandate:    str,
    proponent_analysis: list[dict],
    critic_analysis:    list[dict],
    max_rounds:        int,
) -> list[dict]:
    """
    Run up to max_rounds of cross-examination.
    Each round: both agents respond to the LATEST version of the opponent's analysis.
    Returns the full debate transcript (list of round-result dicts).
    """
    transcript = []

    current_proponent_analysis = proponent_analysis
    current_critic_analysis    = critic_analysis

    for rnd in range(1, max_rounds + 1):
        print(f"\n  │  Round {rnd}/{max_rounds}", flush=True)
        print(f"  │  {'·'*58}", flush=True)

        # Both agents run in parallel — they are responding to the PREVIOUS round's output
        pro_task = debate_round(
            round_num=rnd,
            agent_name=proponent_name,
            agent_mandate=proponent_mandate,
            agent_model=PROPONENT_MODEL,
            agent_analysis=current_proponent_analysis,
            opponent_analysis=current_critic_analysis,
            role_label="PROPONENT",
        )
        crit_task = debate_round(
            round_num=rnd,
            agent_name=critic_name,
            agent_mandate=critic_mandate,
            agent_model=CRITIC_MODEL,
            agent_analysis=current_critic_analysis,
            opponent_analysis=current_proponent_analysis,
            role_label="CRITIC",
        )

        pro_result, crit_result = await asyncio.gather(pro_task, crit_task)
        transcript.append({"proponent": pro_result, "critic": crit_result})

        # ── Check for convergence (both sides conceding heavily) ──────────────
        pro_changes  = len(pro_result["structured"].get("position_changes", []))
        crit_changes = len(crit_result["structured"].get("position_changes", []))
        print(f"  │  └─ Round {rnd} complete — "
              f"Proponent revised {pro_changes} positions, "
              f"Critic revised {crit_changes} positions.", flush=True)

        # Carry forward the rebuttal text as context for the next round
        # (agents reason about their revised stance, not their original JSON)
        current_proponent_analysis = [{"rebuttal_round": rnd, "text": pro_result["rebuttal"]}]
        current_critic_analysis    = [{"rebuttal_round": rnd, "text": crit_result["rebuttal"]}]

        # Early stopping: if both sides made zero position changes in round ≥ 2,
        # the debate has stabilised — no need to continue
        if rnd >= 2 and pro_changes == 0 and crit_changes == 0:
            print(f"  │  ⚑ Debate converged early after round {rnd} (no position changes).", flush=True)
            break

    return transcript


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4 — Area Chair Resolution
# ═══════════════════════════════════════════════════════════════════════════════

CHAIR_SYSTEM = (
    "You are the Area Chair for ICML, the world's premier machine learning conference. "
    "You have just presided over a structured adversarial peer-review debate. "
    "Your role is to act as the ultimate arbiter: cool-headed, evidence-driven, and decisive. "
    "The baseline assumption for any submission is rejection. Only papers in the top 20% of quality are accepted."
)

async def area_chair_resolution(
    proponent_name:    str,
    critic_name:       str,
    proponent_analysis: list[dict],
    critic_analysis:    list[dict],
    debate_transcript:  list[dict],
) -> str:
    """
    Provides the Area Chair with the full debate record and requests a final verdict.
    """
    # ── Build a human-readable debate summary for the chair ──────────────────
    debate_summary_parts = []
    for i, rnd in enumerate(debate_transcript, 1):
        debate_summary_parts.append(
            f"=== ROUND {i} ===\n\n"
            f"[PROPONENT — {proponent_name}]\n"
            f"{rnd['proponent']['rebuttal']}\n\n"
            f"[CRITIC — {critic_name}]\n"
            f"{rnd['critic']['rebuttal']}"
        )
    debate_text = "\n\n".join(debate_summary_parts)

    user_msg = (
        f"INITIAL PROPONENT ANALYSIS (by {proponent_name}):\n"
        f"{json.dumps(proponent_analysis, indent=2)}\n\n"
        f"INITIAL CRITIC ANALYSIS (by {critic_name}):\n"
        f"{json.dumps(critic_analysis, indent=2)}\n\n"
        f"FULL DEBATE TRANSCRIPT:\n"
        f"{debate_text}\n\n"
        "---\n"
        "As Area Chair, perform the following:\n\n"
        "1. CLAIM-BY-CLAIM VERDICT: For each major claim debated, explicitly state which "
        "agent's argument was more logically sound and why. Label each verdict clearly.\n\n"
        "2. OVERALL ASSESSMENT: Which agent won the overall debate? Summarise the decisive "
        "arguments on both sides.\n\n"
        "3. PAPER STRENGTHS: List the genuine strengths that survived the debate.\n\n"
        "4. FATAL WEAKNESSES: List any flaws that the Critic raised and the Proponent "
        "could not adequately refute.\n\n"
        "5. META-REVIEW: Write a concise 3-5 sentence meta-review as if sending it to the "
        "authors.\n\n"
        "6. FINAL DECISION: The default decision for this conference is Reject. You must only "
        "output DECISION: Accept if the Proponent thoroughly dismantled every single methodological "
        "flaw raised by the Critic. If any of the Critic's substantial concerns remain unresolved, "
        "or were only partially addressed by the Proponent, you must output exactly:\n"
        "   DECISION: Reject\n"
        "Otherwise, if the Critic is entirely defeated, output exactly:\n"
        "   DECISION: Accept"
    )

    print(f"  └─ Area Chair deliberating...", flush=True)
    resp = await client.chat.completions.create(
        model=CHAIR_MODEL,
        messages=[
            {"role": "system", "content": CHAIR_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.7,
        top_p=0.95,
    )
    result = resp.choices[0].message.content
    print(f"  └─ Area Chair delivered verdict.", flush=True)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Full Pipeline Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

async def run_debate_council(paper_text: str, max_rounds: int) -> dict:
    """
    Runs the full 4-phase debate council on a single paper.
    Returns a structured dict with all intermediate outputs.
    """

    # ── Phase 1: Profile ──────────────────────────────────────────────────────
    print(section("Phase 1 │ Dynamic Adversarial Profiling"))
    profile = await profile_paper(paper_text)

    domain           = profile.get("domain", "Unknown Domain")
    proponent_name   = profile["proponent"]["name"]
    proponent_mandate= profile["proponent"]["mandate"]
    critic_name      = profile["critic"]["name"]
    critic_mandate   = profile["critic"]["mandate"]

    print(f"\n  Domain    : {domain}")
    print(f"  Proponent : {proponent_name}")
    print(f"  Mandate   : {proponent_mandate}")
    print(f"  Critic    : {critic_name}")
    print(f"  Mandate   : {critic_mandate}")

    # ── Phase 2: Independent Extraction (parallel) ────────────────────────────
    print(section("Phase 2 │ Independent Logical Extraction"))
    print(f"  Proponent ({PROPONENT_MODEL}) and Critic ({CRITIC_MODEL}) reading paper in parallel...\n")

    pro_task  = extract_analysis(proponent_name, proponent_mandate, paper_text, PROPONENT_MODEL, "PROPONENT")
    crit_task = extract_analysis(critic_name,    critic_mandate,    paper_text, CRITIC_MODEL,    "CRITIC")
    proponent_analysis, critic_analysis = await asyncio.gather(pro_task, crit_task)

    print(f"\n  ✓ Proponent extracted {len(proponent_analysis)} claims")
    print(f"  ✓ Critic   extracted  {len(critic_analysis)} claims")
    log_block(f"PROPONENT ANALYSIS — {proponent_name}", json.dumps(proponent_analysis, indent=2))
    log_block(f"CRITIC ANALYSIS — {critic_name}",       json.dumps(critic_analysis,    indent=2))

    # ── Phase 3: Debate ───────────────────────────────────────────────────────
    print(section(f"Phase 3 │ Cross-Examination Debate  (max {max_rounds} rounds)"))
    print(f"  Proponent ({proponent_name}) ↔  Critic ({critic_name})")
    print(f"  Models: {PROPONENT_MODEL} ↔ {CRITIC_MODEL}\n")

    debate_transcript = await run_debate(
        proponent_name=proponent_name,
        proponent_mandate=proponent_mandate,
        critic_name=critic_name,
        critic_mandate=critic_mandate,
        proponent_analysis=proponent_analysis,
        critic_analysis=critic_analysis,
        max_rounds=max_rounds,
    )

    # Print per-round summaries
    for i, rnd in enumerate(debate_transcript, 1):
        print(f"\n  Round {i} Summaries:")
        pro_summary  = rnd["proponent"]["structured"].get("round_summary", "—")
        crit_summary = rnd["critic"]["structured"].get("round_summary", "—")
        log_block(f"Proponent R{i} Summary", pro_summary)
        log_block(f"Critic    R{i} Summary", crit_summary)

    # ── Phase 4: Area Chair ───────────────────────────────────────────────────
    print(section("Phase 4 │ Area Chair Resolution"))
    print(f"  Judge model: {CHAIR_MODEL}\n")

    verdict = await area_chair_resolution(
        proponent_name=proponent_name,
        critic_name=critic_name,
        proponent_analysis=proponent_analysis,
        critic_analysis=critic_analysis,
        debate_transcript=debate_transcript,
    )

    log_block("AREA CHAIR VERDICT", verdict)

    return {
        "domain":              domain,
        "proponent_persona":   {"name": proponent_name,  "mandate": proponent_mandate},
        "critic_persona":      {"name": critic_name,     "mandate": critic_mandate},
        "proponent_analysis":  proponent_analysis,
        "critic_analysis":     critic_analysis,
        "debate_transcript":   debate_transcript,
        "verdict":             verdict,
        "rounds_completed":    len(debate_transcript),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset(path: str = "dataset.jsonl") -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line.strip())
            pid = entry.get("paper_id")
            # Automatically find the markdown file even if it's not marked in the dataset json
            expected_md = Path(f"papers_markdown/{pid}/{pid}.md")
            if expected_md.exists():
                entry["markdown_path"] = str(expected_md)
                entries.append(entry)
            elif entry.get("markdown_path") and Path(entry["markdown_path"]).exists():
                entries.append(entry)
    return entries


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser(
        description="Debate Council — adversarial LLM peer review"
    )
    parser.add_argument("--paper-id", default=None,
                        help="Evaluate a single paper by ID")
    parser.add_argument("--dataset",  default="dataset.jsonl",
                        help="Path to dataset manifest")
    parser.add_argument("--output",   default="results_debate.jsonl",
                        help="Output results file")
    parser.add_argument("--rounds",   type=int, default=3,
                        help="Max debate rounds (default: 3, max enforced: 3)")
    args = parser.parse_args()

    max_rounds = min(args.rounds, 3)   # hard cap at 3

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
    print(f"  ██  Debate Council — Adversarial LLM Peer Review")
    print(f"  ██  Papers: {len(dataset)}  │  Max debate rounds: {max_rounds}")
    print(f"  ██  Proponent : {PROPONENT_MODEL}  │  Critic: {CRITIC_MODEL}")
    print(f"  ██  Chair     : {CHAIR_MODEL}")
    print(f"{'═'*72}")

    for i, entry in enumerate(dataset, 1):
        paper_id = entry["paper_id"]

        if paper_id in completed_ids:
            print(f"\n  [{i}/{len(dataset)}] SKIP  {paper_id}  (already in output)")
            continue

        title = entry["title"][:60] + ("…" if len(entry["title"]) > 60 else "")
        gt    = entry.get("decision", entry.get("label", "?"))

        print(f"\n\n{'▓'*72}")
        print(f"  [{i}/{len(dataset)}]  Paper  : {title}")
        print(f"             ID      : {paper_id}")
        print(f"             Truth   : {gt}")
        print(f"{'▓'*72}")

        md_path    = Path(entry["markdown_path"])
        raw_text   = md_path.read_text(encoding="utf-8")
        # Strip markdown images to save tokens since models only need the text
        paper_text = re.sub(r'!\[.*?\]\(.*?\)', '', raw_text)

        try:
            council_output = await run_debate_council(paper_text, max_rounds)
            prediction     = extract_decision(council_output["verdict"])
            match          = "✓" if prediction and prediction.upper() == gt.upper() else "✗"
            if match == "✓":
                correct += 1

            rounds_done = council_output["rounds_completed"]
            print(f"\n\n  {'━'*68}")
            print(f"  RESULT │ Prediction: {prediction or 'NO DECISION'} │ Ground Truth: {gt} │ {match}")
            print(f"         │ Rounds completed: {rounds_done}")
            print(f"  {'━'*68}\n")

        except Exception as e:
            import traceback
            print(f"\n  ✗ ERROR processing {paper_id}:")
            traceback.print_exc()
            council_output = {"error": str(e)}
            prediction, match = None, "✗"

        result = {
            **entry,
            "method":          "debate_council",
            "proponent_model": PROPONENT_MODEL,
            "critic_model":    CRITIC_MODEL,
            "chair_model":     CHAIR_MODEL,
            "max_rounds":      max_rounds,
            "prediction":      prediction,
            "correct":         match == "✓",
            **council_output,
        }
        results.append(result)

        # Incremental write — safe to interrupt
        with open(args.output, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    # ── Final summary ─────────────────────────────────────────────────────────
    n = len(results)
    print(f"\n{'═'*72}")
    print(f"  Debate Council Complete")
    print(f"  Accuracy : {correct}/{n} ({100*correct/n:.1f}%)" if n else "  No results.")
    print(f"  Output   : {args.output}")
    print(f"{'═'*72}\n")


if __name__ == "__main__":
    asyncio.run(main())
