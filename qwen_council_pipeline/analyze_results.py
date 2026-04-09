"""
Analyze rubric_eval results from a .jsonl file.

Modes
-----
1. Simple report (default):
   python analyze_results.py rubric_results1.jsonl
   python analyze_results.py rubric_results1.jsonl --threshold 7.4

2. Full sweep on all papers:
   python analyze_results.py rubric_results1.jsonl --sweep

3. Val/test split — find threshold on val set, report on held-out test set:
   python analyze_results.py rubric_results1.jsonl --val-size 10
   python analyze_results.py rubric_results1.jsonl --val-size 6 --seed 99
   python analyze_results.py rubric_results1.jsonl --val-size 10 --sweep

   --val-size N  uses N/2 ACCEPT + N/2 REJECT as the validation set.
                 The best threshold found on val is then applied to the test set.
                 Remaining papers (balanced where possible) form the test set.
   --seed        random seed for the split (default: 42)
"""
import json
import random
import argparse


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def load(path: str) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def accuracy_at(results: list[dict], threshold: float) -> dict:
    tp = tn = fp = fn = 0
    for r in results:
        score = r["aggregation"]["avg_overall"]
        pred = "Accept" if score >= threshold else "Reject"
        true = r["label"]          # "ACCEPT" or "REJECT"
        if true == "ACCEPT":
            if pred == "Accept": tp += 1
            else:                fn += 1
        else:
            if pred == "Reject": tn += 1
            else:                fp += 1
    total = tp + tn + fp + fn
    acc  = (tp + tn) / total if total else 0
    prec = tp / (tp + fp)   if (tp + fp) else 0
    rec  = tp / (tp + fn)   if (tp + fn) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return dict(threshold=threshold, acc=acc, tp=tp, tn=tn, fp=fp, fn=fn,
                precision=prec, recall=rec, f1=f1, n=total)


def best_threshold(results: list[dict]) -> tuple[float, dict]:
    """Return (threshold, metrics) that maximises accuracy then F1 on results."""
    scores = sorted(set(r["aggregation"]["avg_overall"] for r in results))
    candidates: set[float] = set()
    for i in range(len(scores) - 1):
        candidates.add(round((scores[i] + scores[i + 1]) / 2, 3))
    lo, hi = min(scores) - 0.5, max(scores) + 0.5
    t = lo
    while t <= hi:
        candidates.add(round(t, 2))
        t = round(t + 0.1, 2)
    sweep = [accuracy_at(results, t) for t in sorted(candidates)]
    best = max(sweep, key=lambda m: (m["acc"], m["f1"]))
    return best["threshold"], best


# ─────────────────────────────────────────────────────────────────────────────
# Splitting
# ─────────────────────────────────────────────────────────────────────────────

def split_val_test(results: list[dict], val_size: int, seed: int) -> tuple[list, list]:
    """
    Stratified split: val_size papers into validation (val_size/2 per class),
    remainder into test set.

    val_size must be even. Both classes must have >= val_size//2 papers each.
    """
    if val_size % 2 != 0:
        raise ValueError("--val-size must be even (equal accept/reject split)")
    half = val_size // 2

    accepts = [r for r in results if r["label"] == "ACCEPT"]
    rejects = [r for r in results if r["label"] == "REJECT"]

    if len(accepts) < half or len(rejects) < half:
        raise ValueError(
            f"Not enough papers for val split: have {len(accepts)} ACCEPT, "
            f"{len(rejects)} REJECT, need {half} of each."
        )

    rng = random.Random(seed)
    val_accepts = rng.sample(accepts, half)
    val_rejects = rng.sample(rejects, half)

    val_ids = {id(r) for r in val_accepts + val_rejects}
    test = [r for r in results if id(r) not in val_ids]
    val  = val_accepts + val_rejects
    return val, test


# ─────────────────────────────────────────────────────────────────────────────
# Printing
# ─────────────────────────────────────────────────────────────────────────────

def print_report(results: list[dict], threshold: float, label: str = "Results") -> None:
    m = accuracy_at(results, threshold)
    n_accept = sum(1 for r in results if r["label"] == "ACCEPT")
    n_reject = sum(1 for r in results if r["label"] == "REJECT")

    print(f"\n{'='*62}")
    print(f"  {label}  ({n_accept} ACCEPT / {n_reject} REJECT)")
    print(f"  Threshold: avg_overall >= {threshold:.2f} → Accept")
    print(f"{'='*62}")
    print(f"  Accuracy  : {m['acc']*100:.1f}%  ({m['tp']+m['tn']}/{m['n']})")
    print(f"  Precision : {m['precision']*100:.1f}%")
    print(f"  Recall    : {m['recall']*100:.1f}%")
    print(f"  F1        : {m['f1']*100:.1f}%")
    print(f"\n  Confusion matrix (predicted →)")
    print(f"               Accept  Reject")
    print(f"  True ACCEPT    {m['tp']:3d}     {m['fn']:3d}")
    print(f"  True REJECT    {m['fp']:3d}     {m['tn']:3d}")

    # Score distribution
    a_scores = [r["aggregation"]["avg_overall"] for r in results if r["label"] == "ACCEPT"]
    r_scores = [r["aggregation"]["avg_overall"] for r in results if r["label"] == "REJECT"]
    if a_scores and r_scores:
        print(f"\n  Score distribution (avg_overall):")
        print(f"    ACCEPT  mean={sum(a_scores)/len(a_scores):.2f}  "
              f"min={min(a_scores):.2f}  max={max(a_scores):.2f}")
        print(f"    REJECT  mean={sum(r_scores)/len(r_scores):.2f}  "
              f"min={min(r_scores):.2f}  max={max(r_scores):.2f}")
        print(f"    Gap: {sum(a_scores)/len(a_scores) - sum(r_scores)/len(r_scores):.2f}")

    # Misclassified
    wrong = []
    for r in results:
        score = r["aggregation"]["avg_overall"]
        pred = "Accept" if score >= threshold else "Reject"
        if r["label"] == "ACCEPT" and pred != "Accept":
            wrong.append(("ACCEPT", pred, score, r.get("title", "")[:55]))
        elif r["label"] == "REJECT" and pred != "Reject":
            wrong.append(("REJECT", pred, score, r.get("title", "")[:55]))
    if wrong:
        print(f"\n  Misclassified ({len(wrong)}):")
        for true, pred, score, title in sorted(wrong, key=lambda x: x[2], reverse=True):
            print(f"    [{true} → {pred}]  score={score:.2f}  {title}")
    else:
        print("\n  No misclassifications!")
    print()


def print_sweep(results: list[dict], label: str = "Threshold Sweep") -> float:
    """Print full sweep table, return the best threshold found."""
    scores = sorted(set(r["aggregation"]["avg_overall"] for r in results))
    candidates: set[float] = set()
    for i in range(len(scores) - 1):
        candidates.add(round((scores[i] + scores[i + 1]) / 2, 3))
    lo, hi = min(scores) - 0.5, max(scores) + 0.5
    t = lo
    while t <= hi:
        candidates.add(round(t, 2))
        t = round(t + 0.1, 2)

    sweep = [accuracy_at(results, t) for t in sorted(candidates)]
    best = max(sweep, key=lambda m: (m["acc"], m["f1"]))

    print(f"\n{'='*62}")
    print(f"  {label}  (best threshold = {best['threshold']:.2f})")
    print(f"{'='*62}")
    print(f"  {'Threshold':>10}  {'Accuracy':>9}  {'F1':>7}  {'TP':>4}  {'TN':>4}  {'FP':>4}  {'FN':>4}")
    print(f"  {'-'*57}")
    prev_acc = None
    for m in sweep:
        marker = "  ◄ best" if m["threshold"] == best["threshold"] else ""
        if m["acc"] != prev_acc or marker:
            print(f"  {m['threshold']:>10.2f}  {m['acc']*100:>8.1f}%  {m['f1']*100:>6.1f}%  "
                  f"{m['tp']:>4}  {m['tn']:>4}  {m['fp']:>4}  {m['fn']:>4}{marker}")
        prev_acc = m["acc"]
    print(f"\n  Recommended: {best['threshold']:.2f}  "
          f"(acc={best['acc']*100:.1f}%  F1={best['f1']*100:.1f}%)")
    print()
    return best["threshold"]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze rubric_eval .jsonl results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("results_file", help="Path to .jsonl results file")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Fixed decision threshold (overrides sweep/stored value)")
    parser.add_argument("--sweep", action="store_true",
                        help="Show full threshold sweep table")
    parser.add_argument("--val-size", type=int, default=None, metavar="N",
                        help="Use N papers (N/2 per class) to find threshold; "
                             "report accuracy on the remaining held-out test set. "
                             "N must be even.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for val/test split (default: 42)")
    args = parser.parse_args()

    results = load(args.results_file)
    if not results:
        print("No results found.")
        return

    # ── Mode 1: Val/test split ──────────────────────────────────────────────
    if args.val_size is not None:
        val, test = split_val_test(results, args.val_size, args.seed)

        print(f"\n  Split: {len(val)} validation  /  {len(test)} test  (seed={args.seed})")
        print(f"  Val   = {sum(1 for r in val if r['label']=='ACCEPT')} ACCEPT "
              f"+ {sum(1 for r in val if r['label']=='REJECT')} REJECT")
        print(f"  Test  = {sum(1 for r in test if r['label']=='ACCEPT')} ACCEPT "
              f"+ {sum(1 for r in test if r['label']=='REJECT')} REJECT")

        if args.sweep:
            # Show full sweep on val set
            chosen_threshold = print_sweep(val, label="Validation Threshold Sweep")
        else:
            # Just find best silently
            chosen_threshold, val_metrics = best_threshold(val)
            print(f"\n  Best threshold on val set: {chosen_threshold:.2f}  "
                  f"(val acc={val_metrics['acc']*100:.1f}%  F1={val_metrics['f1']*100:.1f}%)")

        # Override if user also passed --threshold
        if args.threshold is not None:
            chosen_threshold = args.threshold
            print(f"  [--threshold override: using {chosen_threshold:.2f}]")

        # Report on validation set
        print_report(val, chosen_threshold, label="Validation Set")

        # Report on held-out test set
        if test:
            print_report(test, chosen_threshold, label="Test Set (held-out)")
        else:
            print("  (No papers left for test set — reduce --val-size)")

    # ── Mode 2: Sweep on all papers ─────────────────────────────────────────
    elif args.sweep:
        threshold = args.threshold  # may be None — sweep ignores it but report uses it
        if threshold is None:
            threshold = print_sweep(results, label="Full Threshold Sweep")
        else:
            print_sweep(results, label="Full Threshold Sweep")
        print_report(results, threshold, label="All Papers")

    # ── Mode 3: Simple report ────────────────────────────────────────────────
    else:
        threshold = args.threshold
        if threshold is None:
            threshold = results[0]["aggregation"].get("threshold", 5.5)
        print_report(results, threshold, label="All Papers")


if __name__ == "__main__":
    main()
