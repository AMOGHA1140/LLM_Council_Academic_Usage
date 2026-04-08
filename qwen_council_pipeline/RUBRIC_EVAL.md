# Calibrated Rubric Council — Technical Documentation

## Quick Start

**Prerequisites:** Ollama running locally with `gemma4:26b` and `qwen3.5:27b` pulled. Papers downloaded and converted via `download_papers.py`.

```bash
# Evaluate all papers in dataset.jsonl
python rubric_eval.py

# Evaluate a single paper (useful for testing / debugging)
python rubric_eval.py --paper-id zv-typ1gPxA

# Adjust the accept/reject decision threshold (default: 5.5)
#   Lower = more papers accepted, Higher = more papers rejected
python rubric_eval.py --threshold 5.0

# Write results to a custom output file
python rubric_eval.py --output results_rubric_t5.0.jsonl

# Use a different dataset manifest
python rubric_eval.py --dataset my_dataset.jsonl

# Combine flags — single paper, custom threshold, custom output
python rubric_eval.py --paper-id zv-typ1gPxA --threshold 6.0 --output test_run.jsonl
```

**Threshold sweep** (run the same dataset at multiple thresholds to find the optimal operating point):

```bash
python rubric_eval.py --threshold 5.0 --output results_t5.0.jsonl
python rubric_eval.py --threshold 5.5 --output results_t5.5.jsonl
python rubric_eval.py --threshold 6.0 --output results_t6.0.jsonl
```

**Resume behavior:** The pipeline automatically skips papers already present in the output file. To re-evaluate a paper, delete its line from the output file. To start fresh, delete the output file entirely.

---

## 1. Motivation: Why This Pipeline Exists

Previous approaches in this project suffered from a **calibration/bias problem**:

| Method | Prompt Bias | Accept Recall | Reject Recall | Overall Accuracy |
|--------|------------|---------------|---------------|-----------------|
| Debate Council (neutral) | None | 13/15 (87%) | 3/15 (20%) | ~53% |
| Debate Council (reject-default) | Reject | 4/15 (27%) | 14/15 (93%) | ~60% |

The LLM was not discriminating paper quality — it was following whatever bias the prompt encoded. Both configurations hover near coin-flip accuracy on a balanced dataset.

**Root cause:** Asking an LLM for a direct binary Accept/Reject decision is inherently uncalibrated. The model has no grounded sense of where the acceptance threshold lies. Prompt phrasing dominates over actual paper analysis.

**Solution:** Decompose the evaluation into scored rubric dimensions using verbal anchors (descriptive text at each scale point), then make the Accept/Reject decision mechanically via a threshold on aggregated scores. The LLM never makes the binary decision — it only classifies quality on described scales where it can reason about the descriptions semantically.

---

## 2. Pipeline Architecture — End to End

The pipeline processes one paper at a time through 6 sequential phases. Phases 0, 4, and 5 are pure Python (deterministic, no LLM). Phases 1-3 involve LLM calls.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Per-Paper Pipeline                               │
│                                                                         │
│  Phase 0: Preprocess ──► Phase 1: Profile ──► Phase 2: Review           │
│  (Python, ~0s)           (1 LLM call)         (3 LLM calls, parallel)   │
│       │                       │                       │                  │
│       ▼                       ▼                       ▼                  │
│  Cleaned markdown       3 personas             3 scored reviews          │
│                                                       │                  │
│                                                       ▼                  │
│                         Phase 3: Discuss ◄────────────┘                  │
│                         (3 LLM calls, parallel)                          │
│                                │                                         │
│                                ▼                                         │
│                         3 revised reviews                                │
│                                │                                         │
│                                ▼                                         │
│                         Phase 4: Extract                                 │
│                         (Python, regex)                                  │
│                                │                                         │
│                                ▼                                         │
│                         Phase 5: Aggregate                               │
│                         (Python, threshold)                              │
│                                │                                         │
│                                ▼                                         │
│                         DECISION: Accept / Reject                        │
└─────────────────────────────────────────────────────────────────────────┘

Total LLM calls per paper: 7 (1 profiler + 3 reviewers + 3 discussants)
```

---

## 3. Phase-by-Phase Detailed Walkthrough

### Phase 0: Preprocessing (Pure Python)

**Function:** `preprocess_paper(raw_markdown: str) -> str`

**Purpose:** Reduce noise and token count by removing sections the LLM cannot meaningfully evaluate.

**What gets stripped:**

| Content | Regex Pattern | Rationale |
|---------|--------------|-----------|
| References section | `^#+\s*(References\|Bibliography)` to EOF | LLMs cannot verify citations or judge reference quality — they would hallucinate judgments about papers they haven't read |
| Appendix / Supplementary | `^#+\s*(Appendix\|Supplementary)` to EOF | Supplementary material is secondary; main claims should stand in the body |
| Image markdown tags | `!\[caption\]\(path\)` → keeps `caption` text | Base64-encoded images consume massive tokens; figure captions alone carry the key information |
| Excessive whitespace | 3+ consecutive newlines → 2 newlines | Token efficiency |

**What is preserved:**
- All LaTeX mathematical notation (marker-pdf preserves `$...$` and `$$...$$`)
- Tables and their captions
- Figure captions (the text inside `![...]`)
- All main body sections: Abstract, Introduction, Related Work, Method, Experiments, Conclusion

**Typical reduction:** Raw markdown papers are often 30K-80K characters. After preprocessing, most fall to 15K-30K characters. The review prompt then truncates to 40K characters as a safety cap.

---

### Phase 1: Dynamic Profiling

**Function:** `profile_paper(paper_text: str) -> list[dict]`

**Purpose:** Generate 3 reviewer personas tailored to the specific paper's methodology. A paper on diffusion models gets different reviewers than a paper on reinforcement learning.

**LLM call details:**
- **Model:** `gemma4:26b` (profiler model)
- **Input:** First 8,000 characters of preprocessed text (abstract + intro + beginning of method section)
- **Temperature:** 0.7
- **Output format:** JSON array of 3 objects with `persona`, `instruction`, `focus_area` fields

**Example output:**
```json
[
  {"persona": "GNN Architecture Specialist",
   "instruction": "Evaluate the graph neural network design choices...",
   "focus_area": "soundness"},
  {"persona": "Code Summarization Expert",
   "instruction": "Assess novelty relative to prior code summarization work...",
   "focus_area": "contribution"},
  {"persona": "Empirical Evaluation Auditor",
   "instruction": "Scrutinize experimental methodology, baselines, metrics...",
   "focus_area": "clarity"}
]
```

**Fallback:** If JSON parsing fails, three generic personas are used: Methodologist (soundness), Theorist (contribution), Domain Expert (clarity). This ensures the pipeline never crashes on a parse failure.

---

### Phase 2: Independent Rubric Reviews

**Function:** `rubric_review(persona_data, paper_text, model, reviewer_idx) -> dict`

**Purpose:** Each persona independently reviews the paper using the ICLR rubric with verbal anchors. This is the core innovation of the pipeline.

**Why verbal anchors instead of raw numbers:**

LLMs are known to be poorly calibrated when asked to output raw numeric scores. A model asked to "rate 1-10" will cluster around 6-7 regardless of quality. But when given descriptive anchors like:

- "6: Marginally above acceptance threshold"
- "5: Marginally below acceptance threshold"

...the model can reason *semantically* about whether a paper is "marginally above" or "marginally below" the bar. The description grounds the number in meaning.

**The ICLR rubric dimensions:**

#### Overall Recommendation (1-10 scale, 6 anchor points)

| Score | Verbal Anchor |
|-------|--------------|
| 10 | Top 5% of accepted papers, seminal paper |
| 8 | Top 15-20% of accepted papers, clear accept |
| 6 | Marginally above acceptance threshold |
| 5 | Marginally below acceptance threshold |
| 3 | Clear reject, significant technical or conceptual issues |
| 1 | Trivially wrong, irrelevant, or already known |

Note: Only 6 anchor points are provided (1, 3, 5, 6, 8, 10), not all 10. This is intentional — it forces the LLM to pick from meaningfully distinct categories rather than agonizing over the difference between 6 and 7.

#### Soundness (1-4 scale)

| Score | Verbal Anchor |
|-------|--------------|
| 4 | Excellent — claims well-supported, proofs correct, experiments rigorous |
| 3 | Good — minor issues but overall technically sound |
| 2 | Fair — some concerns about correctness of claims or methodology |
| 1 | Poor — major flaws in reasoning, proofs, or experimental design |

#### Contribution (1-4 scale)

| Score | Verbal Anchor |
|-------|--------------|
| 4 | Excellent — significant advance, likely to influence future work |
| 3 | Good — interesting contribution with clear value |
| 2 | Fair — incremental or limited contribution |
| 1 | Poor — no significant contribution or already well-known |

#### Clarity (1-4 scale)

| Score | Verbal Anchor |
|-------|--------------|
| 4 | Excellent — well-written, easy to follow, well-organized |
| 3 | Good — generally clear with minor issues |
| 2 | Fair — important parts hard to follow or poorly explained |
| 1 | Poor — very difficult to understand |

#### Confidence (1-5 scale)

| Score | Verbal Anchor |
|-------|--------------|
| 5 | Absolutely certain, familiar with all related work |
| 4 | Confident, but not absolutely certain |
| 3 | Fairly confident |
| 2 | Willing to defend, but possible I missed something |
| 1 | Not confident at all, just an educated guess |

**Model assignment:**

| Reviewer | Model | Rationale |
|----------|-------|-----------|
| R1 | `gemma4:26b` | Strong generalist, tends toward balanced assessments |
| R2 | `qwen3.5:27b` | Different architecture provides bias diversity |
| R3 | `gemma4:26b` | Second Gemma instance for statistical weight |

Using two different model architectures means their systematic biases are less likely to correlate. If both Gemma and Qwen independently score a paper highly, that is stronger evidence than three Gemma instances agreeing.

**LLM call details:**
- **Input:** Full preprocessed paper text, truncated to 40,000 characters
- **Temperature:** 0.7
- **top_p:** 0.95
- **Output:** Free-text analysis followed by structured score lines

**Expected output format (end of review):**
```
OVERALL: "6: Marginally above acceptance threshold"
SOUNDNESS: "3: Good — minor issues but overall technically sound"
CONTRIBUTION: "3: Good — interesting contribution with clear value"
CLARITY: "3: Good — generally clear with minor issues"
CONFIDENCE: "4: Confident, but not absolutely certain"
```

---

### Phase 3: Reviewer Discussion (Score Revision)

**Function:** `discussion_round(reviewer_idx, persona_data, own_review, other_reviews, model) -> dict`

**Purpose:** Simulate the ICLR reviewer forum discussion period. In real conferences, after independent reviews are submitted, reviewers read each other's assessments and can revise their scores. This phase captures that calibration step.

**How it works:**

Each reviewer receives:
1. A summary of their own initial review and scores
2. Summaries of both other reviewers' reviews and scores
3. Instructions to identify missed points, disagree where warranted, and output revised (or confirmed) scores

**What this achieves:**
- A reviewer who was overly enthusiastic may lower scores after reading well-argued weaknesses from another reviewer
- A reviewer who was too harsh may raise scores after realizing they missed a key strength
- Cross-pollination of domain-specific insights (e.g., the soundness expert flags a mathematical issue the contribution expert missed)

**LLM call details:**
- **Model:** Same model as the reviewer used in Phase 2 (preserves voice consistency)
- **Temperature:** 0.7
- **Input:** Own review summary + two other reviews' summaries
- **Output:** Discussion text + revised score lines in same format as Phase 2

**Score changes are tracked:** The output includes a `score_changes` dict showing which scores moved and by how much (e.g., `{"overall": -1, "contribution": -1}`). This is useful for analysis — papers where discussion causes large shifts indicate contentious submissions.

---

### Phase 4: Score Extraction (Pure Python)

**Function:** `extract_scores(review_text: str) -> dict | None`

**Purpose:** Parse the verbal anchor responses into numeric scores. This is entirely deterministic — no LLM is involved.

**Two-layer extraction strategy:**

**Layer 1 — Regex (primary):**
Matches patterns like `OVERALL: "6: Marginally above..."` or `OVERALL: 6: Marginally above...`

```
Pattern: DIMENSION:\s*"?(\d+)\s*:
```

This extracts the leading integer from the verbal anchor. It works regardless of whether the model includes quotes, extra spaces, or the full anchor text.

**Layer 2 — Text matching (fallback):**
If the regex fails (model didn't follow format exactly), searches for anchor text keywords:

| Text Found | Mapped Score |
|-----------|-------------|
| "top 5%" or "seminal" | 10 |
| "top 15" or "clear accept" | 8 |
| "marginally above" or "above acceptance" | 6 |
| "marginally below" or "below acceptance" | 5 |
| "clear reject" | 3 |
| "trivially wrong" or "irrelevant" | 1 |
| "excellent" | 4 (for 1-4 scales) |
| "good" | 3 |
| "fair" | 2 |
| "poor" | 1 |

**Fallback chain in Phase 5:** If post-discussion scores fail to parse, the pipeline falls back to initial review scores (Phase 2). If those also fail, the reviewer is excluded from aggregation.

---

### Phase 5: Aggregation & Decision (Pure Python)

**Function:** `aggregate_and_decide(reviewer_scores: list[dict], threshold: float) -> dict`

**Purpose:** Combine multiple reviewers' scores into a single decision using a mechanical rule. This is where the bias problem is eliminated — no LLM is asked whether to accept or reject.

**Aggregation method: Confidence-weighted average**

Each reviewer's self-reported confidence (1-5) is used as their weight. A reviewer who reports "5: Absolutely certain, familiar with all related work" counts more heavily than one who reports "2: Willing to defend, but possible I missed something."

```
avg_overall = Σ(reviewer_overall × confidence) / Σ(confidence)
```

Same weighting for soundness, contribution, and clarity.

**Decision rule (three conditions, all must be true for Accept):**

1. `avg_overall >= threshold` (default: 5.5)
2. `avg_soundness >= 2.0` (no "Poor" average on technical correctness)
3. `avg_contribution >= 2.0` (no "Poor" average on novelty)

**Why these rules:**
- Condition 1: The threshold 5.5 sits exactly between "5: Marginally below" and "6: Marginally above" — the natural decision boundary on the ICLR scale.
- Conditions 2-3: A paper with a high overall but critically poor soundness (e.g., wrong proofs) or no contribution should still be rejected. This prevents one enthusiastic reviewer from overriding a serious flaw.

---

## 4. Tunable Parameters — Complete Reference

### CLI Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--paper-id` | `None` (all papers) | Evaluate a single paper by its OpenReview ID |
| `--dataset` | `dataset.jsonl` | Path to the dataset manifest file |
| `--output` | `results_rubric.jsonl` | Path for output results file |
| `--threshold` | `5.5` | Decision threshold for weighted avg overall score |

### In-Code Constants (top of file)

| Constant | Current Value | Description |
|----------|--------------|-------------|
| `PROFILER_MODEL` | `"gemma4:26b"` | Model used for Phase 1 persona generation |
| `REVIEWER_MODELS` | `["gemma4:26b", "qwen3.5:27b", "gemma4:26b"]` | Models assigned to each reviewer slot; length determines number of reviewers |

### Preprocessing Parameters (in `preprocess_paper`)

| Parameter | Current Value | Description |
|-----------|--------------|-------------|
| Reference strip pattern | `^#+\s*(References\|Bibliography)` | Regex that identifies where to cut the references section |
| Appendix strip pattern | `^#+\s*(Appendix\|Supplementary)` | Regex that identifies where to cut supplementary material |
| Image tag handling | `!\[caption\]\(path\)` → `caption` | Strips image links but preserves caption text |

### Profiling Parameters (in `profile_paper`)

| Parameter | Current Value | Description |
|-----------|--------------|-------------|
| Input truncation | `paper_text[:8000]` chars | How much of the paper the profiler sees (abstract + intro + early method) |
| Temperature | `0.7` | See hyperparameters section below |
| Number of personas | `3` (hardcoded in prompt) | Could be changed in the prompt text |

### Review Parameters (in `rubric_review`)

| Parameter | Current Value | Description |
|-----------|--------------|-------------|
| Input truncation | `paper_text[:40000]` chars | Maximum paper text sent to each reviewer |
| Temperature | `0.7` | See hyperparameters section below |
| top_p | `0.95` | See hyperparameters section below |
| Rubric anchor points | 6 for overall (1,3,5,6,8,10), 4 for dimensions, 5 for confidence | The discrete scale points presented to the LLM |

### Discussion Parameters (in `discussion_round`)

| Parameter | Current Value | Description |
|-----------|--------------|-------------|
| Number of rounds | `1` (single round) | How many discussion iterations; could be extended to multi-round |
| Temperature | `0.7` | See hyperparameters section below |
| top_p | `0.95` | See hyperparameters section below |

### Aggregation Parameters (in `aggregate_and_decide`)

| Parameter | Current Value | Description |
|-----------|--------------|-------------|
| Decision threshold | `5.5` (CLI-tunable) | Weighted avg overall must be >= this for Accept |
| Soundness floor | `2.0` | Weighted avg soundness must be >= this for Accept |
| Contribution floor | `2.0` | Weighted avg contribution must be >= this for Accept |
| Default confidence | `3` | Confidence weight used when a reviewer's confidence is missing |
| Default overall | `5` | Overall score used when a reviewer's overall is missing |
| Default dimension | `2` | Dimension score used when soundness/contribution/clarity is missing |

---

## 5. Hyperparameters — Detailed Rationale

### Temperature: 0.7

**What it controls:** Temperature scales the logits before softmax sampling. Higher values make the distribution flatter (more random), lower values make it sharper (more deterministic).

**Why 0.7 instead of 1.0:**

The existing pipelines use temperature=1.0 for creative tasks like writing detailed reviews and generating debate rebuttals. That makes sense for those tasks — you want diverse, expressive language.

But rubric scoring is a **classification task**, not a creative generation task. The model is choosing between discrete categories ("Marginally above" vs "Marginally below"). For classification:

- **Temperature 1.0** introduces unnecessary randomness in the final score selection. The model might waver between two adjacent categories when it should commit to one.
- **Temperature 0.0** would be fully deterministic but might cause the model to get stuck in local patterns and not explore its reasoning fully.
- **Temperature 0.7** is a compromise: the reasoning portion of the review remains fluent and varied, while the final score selection is more consistent.

**Note on Ollama implementation:** Ollama's temperature parameter behaves identically to OpenAI's. Setting temperature=0.7 means logits are divided by 0.7 before softmax, making the top token ~43% more likely relative to temperature=1.0.

**If you want to experiment:**
- Lower (0.3-0.5): More deterministic scores, potentially less nuanced reasoning
- Higher (0.8-1.0): More varied scores across runs, potentially better exploration of edge cases
- For a controlled experiment, run the same papers at 0.5, 0.7, and 1.0 and compare score variance

### top_p (Nucleus Sampling): 0.95

**What it controls:** Limits sampling to the smallest set of tokens whose cumulative probability exceeds `top_p`. Tokens outside this nucleus are zeroed out.

**Why 0.95:** This is a standard value that trims only the extreme tail of the distribution (the least likely 5% of tokens). It prevents rare, incoherent token choices without significantly constraining the model's expressiveness. At temperature=0.7, the distribution is already fairly sharp, so top_p=0.95 acts as a safety net rather than a primary control.

**Relationship with temperature:** Temperature and top_p both affect randomness but in different ways:
- Temperature smoothly scales all probabilities
- top_p hard-cuts the tail

Using both together (temperature=0.7 + top_p=0.95) gives a "soft squeeze then hard cut" effect. Some practitioners argue you should only use one or the other. In practice, the combination at these values is well-tested and stable.

### Other Ollama/OpenAI Parameters Not Currently Set

| Parameter | Default | Effect | When to Modify |
|-----------|---------|--------|----------------|
| `max_tokens` | Model-dependent | Maximum output length | If reviews are being truncated before scores are output, increase this |
| `frequency_penalty` | `0.0` | Penalizes token repetition | Could help if reviews are repetitive, but risks disrupting the score format |
| `presence_penalty` | `0.0` | Penalizes topic repetition | Generally leave at 0 for structured output tasks |
| `seed` | `None` | Deterministic sampling | Set to a fixed integer for reproducible runs (Ollama support varies) |
| `stop` | `None` | Stop sequences | Could add `["\n\n\n"]` to prevent excessive output after scores |
| `num_predict` (Ollama-specific) | `-1` (unlimited) | Ollama's max_tokens equivalent | Set if using Ollama's native API instead of OpenAI-compatible endpoint |
| `num_ctx` (Ollama-specific) | `2048-8192` | Context window size | Must be large enough for paper text + prompt; set via `ollama run --num_ctx` or modelfile |

---

## 6. Score Extraction — Robustness Design

The score extraction is designed to handle LLM output that doesn't perfectly follow the requested format. Real-world LLM outputs vary significantly.

**Handled formats (examples that all correctly extract overall=6):**

```
OVERALL: "6: Marginally above acceptance threshold"         ✓ (standard)
OVERALL: 6: Marginally above acceptance threshold           ✓ (no quotes)
OVERALL:  "6:  Marginally above acceptance threshold"       ✓ (extra spaces)
Overall: "6: Marginally above acceptance threshold"         ✓ (case-insensitive)
**OVERALL:** "6: Marginally above acceptance threshold"     ✓ (markdown bold)
My overall rating is marginally above acceptance threshold  ✓ (fallback text match)
```

**Parse failure handling:**
- If post-discussion scores fail to parse → use initial review scores
- If initial scores also fail → reviewer is excluded from aggregation
- If all reviewers fail → decision is `None` (flagged for manual review)
- Parse success/failure is recorded in the output JSON (`parse_success` field)

---

## 7. Output Format

Each paper produces one JSON line in the output file with complete traceability:

```
{
  // --- Ground truth (from dataset.jsonl) ---
  "paper_id": "zv-typ1gPxA",
  "title": "Retrieval-Augmented Generation for Code Summarization via Hybrid GNN",
  "decision": "Accept",          // ground truth decision
  "label": "ACCEPT",             // ground truth label

  // --- Pipeline config ---
  "method": "rubric_council",
  "profiler_model": "gemma4:26b",
  "reviewer_models": ["gemma4:26b", "qwen3.5:27b", "gemma4:26b"],
  "threshold": 5.5,

  // --- Phase 1 output ---
  "personas": [
    {"persona": "...", "instruction": "...", "focus_area": "soundness"},
    ...
  ],

  // --- Phase 2 output ---
  "initial_reviews": [
    {
      "persona": "GNN Architecture Specialist",
      "model": "gemma4:26b",
      "raw_review": "... full review text ...",
      "scores": {"overall": 6, "soundness": 3, "contribution": 3, "clarity": 3, "confidence": 4},
      "parse_success": true
    },
    ...
  ],

  // --- Phase 3 output ---
  "discussion_reviews": [
    {
      "persona": "GNN Architecture Specialist",
      "model": "gemma4:26b",
      "raw_discussion": "... full discussion text ...",
      "revised_scores": {"overall": 5, "soundness": 3, "contribution": 2, "clarity": 3, "confidence": 4},
      "score_changes": {"overall": -1, "contribution": -1},
      "parse_success": true
    },
    ...
  ],

  // --- Phase 4-5 output ---
  "final_scores": [
    {"overall": 5, "soundness": 3, "contribution": 2, "clarity": 3, "confidence": 4},
    ...
  ],
  "aggregation": {
    "avg_overall": 5.33,
    "avg_soundness": 2.67,
    "avg_contribution": 2.33,
    "avg_clarity": 3.0,
    "decision": "Reject",
    "threshold": 5.5,
    "weights": [4, 3, 4],
    "individual_scores": [...]
  },

  // --- Final result ---
  "prediction": "Reject",
  "correct": true
}
```

---

## 8. Resume & Checkpoint Behavior

The pipeline writes results incrementally — after each paper completes, the entire results list is rewritten to the output file. On startup, it reads any existing output file and skips papers that are already present.

This means:
- **Safe to interrupt** (Ctrl+C) at any time — completed papers are preserved
- **Safe to restart** — already-evaluated papers are skipped automatically
- **To re-evaluate a paper:** delete its line from the output file, or delete the entire output file to start fresh

---

## 9. Comparison with Other Methods in This Project

| Aspect | Baseline | Dynamic Council | Debate Council | Rubric Council |
|--------|----------|----------------|----------------|----------------|
| Decision maker | LLM (direct) | LLM (Area Chair) | LLM (Area Chair) | Python threshold |
| Bias vulnerability | High | High | High | Low (mechanical) |
| Scoring method | None | None | None | ICLR verbal anchors |
| Multi-model | No | No | Yes (Gemma+Qwen) | Yes (Gemma+Qwen) |
| Discussion phase | No | No | Yes (adversarial) | Yes (collaborative) |
| LLM calls/paper | 1 | 4 | 7-13 | 7 |
| Interpretability | Low (binary) | Medium (reviews) | High (debate log) | High (numeric scores) |
| Tunable threshold | No | No | No | Yes |
