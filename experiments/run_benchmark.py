"""
run_benchmark.py

Benchmarks the new Adaptive Prompt Engine v2 against two baselines:
    1. adaptive_v2      — full system (complexity → model routing → adaptive prompt)
    2. always_medium    — always uses MEDIUM model, no routing intelligence
    3. no_cache         — adaptive_v2 but with cache disabled (measures cache savings)

For each query records:
    complexity_tier, model_used, tokens, cost_usd, cost_saved_usd,
    latency_ms, confidence, cache_hit

Results written to experiments/results/benchmark_<timestamp>.csv

Usage:
    python -m experiments.run_benchmark
    python -m experiments.run_benchmark --provider gemini --budget balanced
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
import csv
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import AdaptivePromptEngine, EngineResult
from experiments.benchmark_queries import BENCHMARK_QUERIES


@dataclass
class BenchmarkRecord:
    query: str
    expected_type: str          # from benchmark_queries label
    config: str                 # "adaptive_v2" | "always_medium" | "no_cache"
    complexity_tier: str
    model_used: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cost_saved_usd: float
    latency_ms: float
    confidence: float
    cache_hit: bool


def _result_to_record(query: str, expected_type: str, config: str, r: EngineResult) -> BenchmarkRecord:
    return BenchmarkRecord(
        query=query,
        expected_type=expected_type,
        config=config,
        complexity_tier=r.complexity_tier,
        model_used=r.model_used,
        input_tokens=r.input_tokens,
        output_tokens=r.output_tokens,
        cost_usd=r.cost_usd,
        cost_saved_usd=r.cost_saved_usd,
        latency_ms=r.latency_ms,
        confidence=r.confidence,
        cache_hit=r.cache_hit,
    )


def _load_existing_records(filepath: Path) -> list[BenchmarkRecord]:
    records = []
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(BenchmarkRecord(
                    query=row["query"],
                    expected_type=row["expected_type"],
                    config=row["config"],
                    complexity_tier=row["complexity_tier"],
                    model_used=row["model_used"],
                    input_tokens=int(row["input_tokens"]),
                    output_tokens=int(row["output_tokens"]),
                    cost_usd=float(row["cost_usd"]),
                    cost_saved_usd=float(row["cost_saved_usd"]),
                    latency_ms=float(row["latency_ms"]),
                    confidence=float(row["confidence"]),
                    cache_hit=row["cache_hit"] == "True",
                ))
    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Adaptive Prompt Engine v2 benchmark")
    parser.add_argument("--provider", default="mock", choices=["mock", "openai", "gemini", "groq", "nvidia"])
    parser.add_argument("--budget", default="balanced", choices=["low", "balanced", "quality"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume", default=None, help="Path to existing CSV file to resume from")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    # Three configurations
    adaptive = AdaptivePromptEngine(provider=args.provider, model=args.model, budget=args.budget, use_cache=True)
    always_medium = AdaptivePromptEngine(provider=args.provider, model=args.model, budget="balanced", use_cache=False)
    no_cache_eng = AdaptivePromptEngine(provider=args.provider, model=args.model, budget=args.budget, use_cache=False)

    out = Path(args.resume) if args.resume else (Path(args.output) if args.output else None)
    if out is None:
        results_dir = Path(__file__).resolve().parent / "results"
        results_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = results_dir / f"benchmark_{ts}.csv"
    else:
        out = Path(out)

    out.parent.mkdir(parents=True, exist_ok=True)

    records: list[BenchmarkRecord] = _load_existing_records(out) if args.resume else []
    completed_runs = {(r.query, r.config) for r in records}

    total = len(BENCHMARK_QUERIES)
    configs = [
        ("adaptive_v2", adaptive),
        ("always_medium", always_medium),
        ("no_cache", no_cache_eng),
    ]

    file_exists = out.exists() and args.resume
    f = open(out, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=list(BenchmarkRecord.__dataclass_fields__.keys()))
    if not file_exists:
        writer.writeheader()

    stop_benchmark = False
    try:
        for i, (query, expected_type) in enumerate(BENCHMARK_QUERIES, start=1):
            label = expected_type.value if hasattr(expected_type, "value") else str(expected_type)
            print(f"[{i}/{total}] {query[:60]}…")

            for config_name, engine in configs:
                if (query, config_name) in completed_runs:
                    continue

                try:
                    result = engine.ask(query)
                    rec = _result_to_record(query, label, config_name, result)
                    records.append(rec)
                    writer.writerow(rec.__dict__)
                    f.flush()
                except Exception as e:
                    err_str = str(e)
                    if "404" in err_str:
                        print(f"  [WARNING] Skipping {config_name} due to 404 error (Model unavailable)")
                        continue
                    elif "429" in err_str:
                        print(f"  [ERROR] Hit 429 Rate Limit on {config_name}! Pausing benchmark.")
                        stop_benchmark = True
                        break
                    else:
                        raise

            if stop_benchmark:
                break
    finally:
        f.close()

    print(f"\nTotal rows in {out}: {len(records)}")
    if records:
        _summary(records)


def _summary(records: list[BenchmarkRecord]) -> None:
    from collections import defaultdict
    by_config: dict[str, list[BenchmarkRecord]] = defaultdict(list)
    for r in records:
        by_config[r.config].append(r)

    print(f"\n{'Config':<20}{'Avg Conf':<12}{'Avg Tokens':<14}{'Avg Cost $':<14}{'Avg Saved $':<14}{'Cache Hit%'}")
    for config, rows in by_config.items():
        n = len(rows)
        print(
            f"{config:<20}"
            f"{sum(r.confidence for r in rows)/n:<12.3f}"
            f"{sum(r.input_tokens+r.output_tokens for r in rows)/n:<14.1f}"
            f"{sum(r.cost_usd for r in rows)/n:<14.6f}"
            f"{sum(r.cost_saved_usd for r in rows)/n:<14.6f}"
            f"{sum(1 for r in rows if r.cache_hit)/n:<.1%}"
        )


if __name__ == "__main__":
    main()
