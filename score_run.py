"""
快速打分脚本 — 读取已有 run 目录的 prediction.csv，不重新跑 LLM。

用法：
    uv run python score_run.py --run-dir artifacts/runs/xyma_1777448385 --config configs/xyma.local.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import load_app_config

from scoring import score_task


def main():
    parser = argparse.ArgumentParser(description="Score existing run without re-running LLM")
    parser.add_argument("--run-dir", required=True, help="Path to run directory")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = load_app_config(Path(args.config))
    gold_dir = PROJECT_ROOT / "data" / "public" / "output"
    dataset = DABenchPublicDataset(config.dataset.root_path)

    task_dirs = sorted(run_dir.iterdir())
    total_score = 0.0
    scored_count = 0
    by_difficulty: dict[str, list[float]] = {}
    by_solver: dict[str, list[float]] = {}

    print(f"{'Task':<12} {'Diff':<8} {'Solver':<16} {'Score':>6}  {'Pred':>5}  {'Gold':>5}  Status")
    print("-" * 72)

    for task_dir in task_dirs:
        if not task_dir.is_dir():
            continue
        task_id = task_dir.name
        pred_path = task_dir / "prediction.csv"
        gold_path = gold_dir / task_id / "gold.csv"

        # read solver type from trace
        solver_type = "unknown"
        trace_path = task_dir / "trace.json"
        if trace_path.exists():
            try:
                trace = json.loads(trace_path.read_text())
                solver_type = trace.get("solver_type", "unknown")
            except Exception:
                pass

        if not gold_path.exists():
            print(f"{task_id:<12} {'?':<8} {solver_type:<16} {'N/A':>6}  no gold")
            continue

        score = score_task(pred_path, gold_path)
        total_score += score
        scored_count += 1

        try:
            task = dataset.get_task(task_id)
            diff = task.difficulty
        except Exception:
            diff = "?"

        by_difficulty.setdefault(diff, []).append(score)
        by_solver.setdefault(solver_type, []).append(score)

        # count rows for context
        pred_count = len(_load_csv_values(pred_path)) if pred_path.exists() else 0
        gold_count = len(_load_csv_values(gold_path))

        flag = "OK" if score > 0 else "ZERO"
        print(f"{task_id:<12} {diff:<8} {solver_type:<16} {score:>6.3f}  {pred_count:>5}  {gold_count:>5}  {flag}")

    print("=" * 72)
    avg = total_score / scored_count if scored_count else 0
    print(f"Total: {total_score:.2f} / {scored_count}   Avg: {avg:.4f}")
    print()

    print("By difficulty:")
    for diff in sorted(by_difficulty):
        scores = by_difficulty[diff]
        print(f"  {diff:<10}: {sum(scores):.2f}/{len(scores)}  avg={sum(scores)/len(scores):.3f}")

    print()
    print("By solver:")
    for solver in sorted(by_solver):
        scores = by_solver[solver]
        print(f"  {solver:<16}: {sum(scores):.2f}/{len(scores)}  avg={sum(scores)/len(scores):.3f}")


if __name__ == "__main__":
    main()
