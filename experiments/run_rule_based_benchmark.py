"""
run_rule_based_benchmark.py

Benchmarks the RuleBasedClassifier against the 50-query BENCHMARK_QUERIES set.

For each query it records:
    query, expected_type, predicted_type, correct (bool),
    latency_ms (classifier only — no LLM call)

Summary metrics printed and saved to:
    experiments/results/rule_based_benchmark_<timestamp>.csv
    experiments/results/rule_based_output.txt

Usage:
    python -m experiments.run_rule_based_benchmark
    python -m experiments.run_rule_based_benchmark --output custom_path.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Force UTF-8 stdout on Windows so progress indicators render correctly
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.benchmark_queries import BENCHMARK_QUERIES, QueryType
from experiments.rule_based_classifier import RuleBasedClassifier


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------

@dataclass
class ClassificationRecord:
    query: str
    expected_type: str
    predicted_type: str
    correct: bool
    latency_ms: float


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(output_csv: Path) -> list[ClassificationRecord]:
    clf = RuleBasedClassifier()
    records: list[ClassificationRecord] = []

    total = len(BENCHMARK_QUERIES)
    print(f"\nRule-Based Classifier Benchmark — {total} queries")
    print("=" * 60)

    for i, (query, expected) in enumerate(BENCHMARK_QUERIES, start=1):
        expected_label = expected.value if hasattr(expected, "value") else str(expected)

        t0 = time.perf_counter()
        predicted = clf.classify(query)
        latency_ms = (time.perf_counter() - t0) * 1000

        predicted_label = predicted.value if hasattr(predicted, "value") else str(predicted)
        correct = (predicted_label == expected_label)

        status = "✓" if correct else "✗"
        print(
            f"[{i:02d}/{total}] {status}  "
            f"expected={expected_label:<12}  predicted={predicted_label:<12}  "
            f"({latency_ms:.3f} ms)  {query[:55]}…"
        )

        records.append(ClassificationRecord(
            query=query,
            expected_type=expected_label,
            predicted_type=predicted_label,
            correct=correct,
            latency_ms=latency_ms,
        ))

    return records


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _summary(records: list[ClassificationRecord]) -> str:
    total = len(records)
    correct = sum(1 for r in records if r.correct)
    accuracy = correct / total if total else 0.0
    avg_latency = sum(r.latency_ms for r in records) / total if total else 0.0

    # Per-class breakdown
    by_class: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in records:
        by_class[r.expected_type]["total"] += 1
        if r.correct:
            by_class[r.expected_type]["correct"] += 1

    lines = [
        "",
        "=" * 60,
        "RULE-BASED CLASSIFIER — BENCHMARK SUMMARY",
        "=" * 60,
        f"{'Total queries':<30} {total}",
        f"{'Overall accuracy':<30} {accuracy:.1%}  ({correct}/{total} correct)",
        f"{'Avg latency per query':<30} {avg_latency:.4f} ms",
        "",
        f"{'Class':<14} {'Correct':>8} {'Total':>7} {'Accuracy':>10}",
        "-" * 45,
    ]
    for cls in [QueryType.FACTUAL, QueryType.REASONING, QueryType.CREATIVE, QueryType.ANALYTICAL]:
        label = cls.value
        d = by_class.get(label, {"total": 0, "correct": 0})
        n, c = d["total"], d["correct"]
        acc = f"{c/n:.1%}" if n else "N/A"
        lines.append(f"{label:<14} {c:>8} {n:>7} {acc:>10}")

    # Confusion table
    lines += ["", "Confusion Matrix (predicted →, expected ↓):", ""]
    classes = [q.value for q in QueryType]
    header = f"{'':>12}" + "".join(f"{c:>12}" for c in classes)
    lines.append(header)
    conf: dict[tuple, int] = defaultdict(int)
    for r in records:
        conf[(r.expected_type, r.predicted_type)] += 1
    for exp in classes:
        row = f"{exp:>12}"
        for pred in classes:
            row += f"{conf[(exp, pred)]:>12}"
        lines.append(row)

    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------

def _save_csv(records: list[ClassificationRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["query", "expected_type", "predicted_type", "correct", "latency_ms"],
        )
        writer.writeheader()
        for r in records:
            writer.writerow(r.__dict__)
    print(f"\nCSV saved → {path}")


# ---------------------------------------------------------------------------
# Save text output
# ---------------------------------------------------------------------------

def _save_txt(summary_text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"Text summary saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Rule-Based Classifier benchmark")
    parser.add_argument("--output", default=None, help="Custom CSV output path")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(args.output) if args.output else results_dir / f"rule_based_benchmark_{ts}.csv"
    txt_path = results_dir / "rule_based_output.txt"

    records = run_benchmark(csv_path)
    summary_text = _summary(records)

    print(summary_text)
    _save_csv(records, csv_path)
    _save_txt(summary_text, txt_path)


if __name__ == "__main__":
    main()
