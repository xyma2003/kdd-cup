"""
xyma 本地评测脚本。

用法：
    uv run python eval_xyma.py --config configs/xyma.local.yaml [--limit 10] [--task task_11]

对比 prediction.csv 和 gold.csv，输出每个任务的得分和汇总。
"""
from __future__ import annotations

import argparse
import csv
import math
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


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate xyma solver on public tasks")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--limit", type=int, default=None, help="Max tasks to run")
    parser.add_argument("--task", default=None, help="Run a single task ID")
    parser.add_argument("--run-id", default=None, help="Override run_id")
    args = parser.parse_args()

    config_path = Path(args.config)
    app_config = load_app_config(config_path)

    gold_dir = PROJECT_ROOT / "data" / "public" / "output"

    # 创建输出目录
    run_id = args.run_id or app_config.run.run_id
    try:
        effective_run_id, run_output_dir = create_run_output_dir(
            app_config.run.output_dir, run_id=run_id
        )
    except FileExistsError:
        import time
        effective_run_id = f"xyma_{int(time.time())}"
        run_output_dir = app_config.run.output_dir / effective_run_id
        run_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run ID: {effective_run_id}")
    print(f"Output: {run_output_dir}")
    print()

    # 运行
    if args.task:
        artifact = run_single_task(
            task_id=args.task,
            config=app_config,
            run_output_dir=run_output_dir,
            use_xyma=True,
        )
        artifacts = [artifact]
    else:
        _, artifacts = run_benchmark(
            config=app_config,
            run_id=effective_run_id,
            limit=args.limit,
            use_xyma=True,
            progress_callback=lambda a: print(
                f"  [{a.task_id}] {'OK' if a.succeeded else 'FAIL'}"
            ),
        )

    # 评分
    print("\n" + "=" * 60)
    print(f"{'Task':<12} {'Diff':<8} {'Solver':<14} {'Score':>6}  {'Status'}")
    print("-" * 60)

    dataset = DABenchPublicDataset(app_config.dataset.root_path)
    total_score = 0.0
    scored_count = 0
    by_difficulty: dict[str, list[float]] = {}

    for artifact in artifacts:
        task = dataset.get_task(artifact.task_id)
        gold_path = gold_dir / artifact.task_id / "gold.csv"

        if artifact.prediction_csv_path and gold_path.exists():
            score = score_task(artifact.prediction_csv_path, gold_path)
        else:
            score = 0.0

        total_score += score
        scored_count += 1

        diff = task.difficulty
        by_difficulty.setdefault(diff, []).append(score)

        # 从 trace 读取 solver_type
        solver_type = "unknown"
        trace_path = artifact.task_output_dir / "trace.json"
        if trace_path.exists():
            import json
            trace = json.loads(trace_path.read_text())
            solver_type = trace.get("solver_type", "unknown")

        status = "OK" if artifact.succeeded else f"FAIL: {(artifact.failure_reason or '')[:20]}"
        print(f"{artifact.task_id:<12} {diff:<8} {solver_type:<14} {score:>6.3f}  {status}")

    print("=" * 60)
    avg = total_score / scored_count if scored_count > 0 else 0.0
    print(f"Overall: {total_score:.3f} / {scored_count} tasks  (avg={avg:.3f})")
    print()
    print("By difficulty:")
    for diff in ["easy", "medium", "hard", "extreme"]:
        scores = by_difficulty.get(diff, [])
        if scores:
            d_avg = sum(scores) / len(scores)
            print(f"  {diff:<8}: {d_avg:.3f}  ({len(scores)} tasks)")


if __name__ == "__main__":
    main()
