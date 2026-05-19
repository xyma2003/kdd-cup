"""
xyma solver 统一入口。

路由策略：
- TYPE_SQL    → sql_solver（DuckDB 跨源，单次 LLM）
- TYPE_PANDAS → pandas_solver（序列化 DataFrame，单次 LLM）
- TYPE_DOC    → structured_solver（状态机 4 步 pipeline）
- TYPE_HYBRID → structured_solver（状态机 4 步 pipeline）
"""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from data_agent_baseline.agents.model import OpenAIModelAdapter
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.knowledge.parser import parse_knowledge, enrich_knowledge_with_llm
from data_agent_baseline.solvers.classifier import AssetInventory, DifficultyBudget, TaskType, classify_task, estimate_difficulty, scan_assets
from data_agent_baseline.solvers.pandas_solver import solve_pandas
from data_agent_baseline.solvers.sql_solver import solve_sql
from data_agent_baseline.solvers.structured_solver import solve_structured


def solve_task(
    task: PublicTask,
    model: OpenAIModelAdapter,
) -> dict[str, Any]:
    """主入口：给定任务和模型，返回结果字典。"""
    context_dir = task.assets.context_dir

    # Step 1: 扫描资产
    inv = scan_assets(context_dir)

    # Step 2: 解析 knowledge.md（正则提取 + LLM 补充）
    knowledge = parse_knowledge(context_dir)
    knowledge = enrich_knowledge_with_llm(knowledge, task.question, model)

    # Step 3: 分类任务
    task_type = classify_task(inv)

    # Step 3.5: 估计难度（零 LLM，纯规则）→ 执行预算
    budget = estimate_difficulty(inv, task.question)

    # Step 4: 路由
    answer_table: AnswerTable | None = None
    df: pd.DataFrame | None = None
    error_msg: str | None = None

    try:
        if task_type == TaskType.SQL:
            df = solve_sql(task.question, inv, knowledge, model, budget=budget)

        elif task_type == TaskType.PANDAS:
            df = solve_pandas(task.question, inv, knowledge, model)

        elif task_type in (TaskType.DOC, TaskType.HYBRID):
            # 状态机驱动 pipeline：EXTRACT_ENTITIES → BUILD_QUERY → EXECUTE
            answer_table = solve_structured(task, inv, knowledge, model)
            # 若 structured_solver 返回空（超时/失败），按有无 DB 兜底
            if not answer_table or not answer_table.rows:
                if inv.has_db:
                    df_fallback = solve_sql(task.question, inv, knowledge, model, budget=budget)
                elif inv.has_csv or inv.has_json:
                    df_fallback = solve_pandas(task.question, inv, knowledge, model)
                else:
                    df_fallback = None
                if df_fallback is not None and not df_fallback.empty:
                    answer_table = AnswerTable.from_dataframe(df_fallback)

    except Exception as e:
        error_msg = f"{task_type.value} solver error: {traceback.format_exc()}"
        df = None
        answer_table = None

    # Step 5: 后处理
    if answer_table is not None:
        succeeded = bool(answer_table.columns and answer_table.rows)
        return {
            "task_id": task.task_id,
            "answer": answer_table.to_dict() if succeeded else None,
            "succeeded": succeeded,
            "failure_reason": None if succeeded else "Agent returned empty answer",
            "solver_type": task_type.value,
            "steps": [],
        }

    if df is not None and not df.empty:
        answer_table = AnswerTable.from_dataframe(df)
        succeeded = True
    elif df is not None and df.empty:
        answer_table = AnswerTable.from_dataframe(df)
        succeeded = True
        if error_msg is None:
            error_msg = "Query returned empty result"
    else:
        answer_table = AnswerTable(columns=[], rows=[])
        succeeded = False
        if error_msg is None:
            error_msg = "Solver returned None"

    return {
        "task_id": task.task_id,
        "answer": answer_table.to_dict() if succeeded and answer_table.columns else None,
        "succeeded": succeeded and bool(answer_table.columns),
        "failure_reason": error_msg if not (succeeded and answer_table.columns) else None,
        "solver_type": task_type.value,
        "steps": [],
    }
