#!/bin/sh
set -e

CONFIG=/app/configs/eval.yaml
STAGING=/tmp/eval_staging/run
OUTPUT=/output

mkdir -p /logs

dabench run-benchmark --config "$CONFIG" 2>&1 | tee /logs/benchmark.log || true

for task_dir in "$STAGING"/task_*/; do
    task_id=$(basename "$task_dir")
    src="$task_dir/prediction.csv"
    if [ -f "$src" ]; then
        mkdir -p "$OUTPUT/$task_id"
        cp "$src" "$OUTPUT/$task_id/prediction.csv"
    fi
done

if [ -f "$STAGING/summary.json" ]; then
    cp "$STAGING/summary.json" /logs/summary.json
fi
