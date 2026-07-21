"""
增量评测脚本 — 只跑指定任务列表，比较新旧得分。

用法：
    uv run python run_incremental.py --config configs/xyma.local.yaml --tasks task_75 task_86 task_11
    uv run python run_incremental.py --config configs/xyma.local.yaml --compare artifacts/runs/20260429T071931Z
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
from data_agent_baseline.run.runner import (
    create_run_output_dir,
    run_benchmark,
    run_single_task,
)

from scoring import score_task


def main():
    parser = argparse.ArgumentParser(description="Incremental evaluation — re-run specific tasks")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tasks", nargs="+", help="Task IDs to run (e.g. task_75 task_86)")
    parser.add_argument("--compare", default=None, help="Path to baseline run dir for score comparison")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    config = load_app_config(Path(args.config))
    gold_dir = PROJECT_ROOT / "data" / "public" / "output"
    dataset = DABenchPublicDataset(config.dataset.root_path)

    # Load baseline scores
    baseline_scores: dict[str, float] = {}
    if args.compare:
        baseline_dir = Path(args.compare)
        for task_dir in baseline_dir.iterdir():
            if not task_dir.is_dir():
                continue
            task_id = task_dir.name
            pred_path = task_dir / "prediction.csv"
            gold_path = gold_dir / task_id / "gold.csv"
            if pred_path.exists() and gold_path.exists():
                baseline_scores[task_id] = score_task(pred_path, gold_path)

    # Determine which tasks to run
    if args.tasks:
        task_ids = args.tasks
    elif args.compare:
        # Re-run all zeros from baseline
        task_ids = [t for t, s in baseline_scores.items() if s == 0.0]
        print(f"Re-running {len(task_ids)} ZERO-scored tasks from baseline...")
    else:
        parser.error("Provide --tasks or --compare")

    print(f"Tasks to run: {len(task_ids)}")
    print()

    # Create output dir
    import time
    run_id = f"incr_{int(time.time())}"
    run_output_dir = Path(config.run.output_dir) / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {run_output_dir}")
    print()

    # Override config task list by monkeypatching dataset
    artifacts = []
    for task_id in sorted(task_ids):
        print(f"  Running {task_id}...", flush=True)
        artifact = run_single_task(
            task_id=task_id,
            config=config,
            run_output_dir=run_output_dir,
            use_xyma=True,
        )
        gold_path = gold_dir / task_id / "gold.csv"
        new_score = score_task(artifact.prediction_csv_path, gold_path) if artifact.prediction_csv_path and gold_path.exists() else 0.0
        old_score = baseline_scores.get(task_id, None)
        delta = f" ({'+' if new_score > (old_score or 0) else ''}{new_score - (old_score or 0):.3f} vs baseline)" if old_score is not None else ""
        flag = "OK" if new_score > 0 else "ZERO"
        print(f"    → {flag}  score={new_score:.3f}{delta}")
        artifacts.append((task_id, new_score, old_score))

    print()
    print("=" * 60)
    total_new = sum(s for _, s, _ in artifacts)
    print(f"Incremental total: {total_new:.2f} / {len(artifacts)}")
    if baseline_scores:
        total_old_for_these = sum((baseline_scores.get(t, 0) or 0) for t, _, _ in artifacts)
        delta = total_new - total_old_for_these
        print(f"Delta vs baseline:  {delta:+.2f}  (these tasks: {total_old_for_these:.2f} → {total_new:.2f})")
        # Estimate overall score if applied to full baseline
        baseline_total = sum(baseline_scores.values())
        estimated_total = baseline_total - total_old_for_these + total_new
        print(f"Estimated full score: {baseline_total:.2f} + ({delta:+.2f}) = {estimated_total:.2f} / 50")


if __name__ == "__main__":
    main()
