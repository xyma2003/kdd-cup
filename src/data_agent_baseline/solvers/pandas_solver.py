"""
TYPE_PANDAS 路径：纯 CSV/JSON 数据，生成 pandas 代码执行。

流程：
1. 加载所有 CSV 和 JSON(Airtable格式) 为 DataFrame
2. 提取所有表的 schema + head(3) 样本
3. 推断需要 Join 的表（共享列名 or 值重叠）
4. 一次 LLM 调用生成完整 pandas 代码
5. subprocess 执行，捕获 result_df
6. 失败时：带错误信息重试一次
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from data_agent_baseline.agents.model import OpenAIModelAdapter
from data_agent_baseline.knowledge.parser import KnowledgeContext
from data_agent_baseline.solvers.classifier import AssetInventory


_PANDAS_SYSTEM_PROMPT = """You are an expert Python/pandas data analyst.

The DataFrames listed below are already loaded as variables. Write Python code to answer the question.

Rules:
1. DataFrames are pre-loaded — do NOT use pd.read_csv(), open(), or any file I/O.
2. Store your final answer in a variable called `result_df` (a pandas DataFrame).
3. pandas is already imported as `pd`. Import nothing else.
4. For percentage: use float division, keep precision in intermediate steps.
5. result_df must have meaningful column names (not 0, 1, 2).
6. Output ONLY the Python code, no explanation, no markdown fences.
7. Select ONLY the columns needed to answer the question. Do NOT include extra columns.
8. Column count must match the question: one thing asked → 1 column, two things → 2 columns.
9. If the question asks for "full name" and data has separate first_name/last_name columns, keep them as TWO separate columns — do NOT concatenate.
10. For "average monthly": total divided by 12 (or distinct months count), NOT mean of all rows.
11. After joining DataFrames, call drop_duplicates() if the join could produce duplicate rows.
12. Never add id, record_id, primary key columns unless the question explicitly asks for them.
13. Comparison operators — be precise:
    - "more than N" / "greater than N" → use > N  (NOT >= N)
    - "at least N" / "N or more" → use >= N
    - "less than N" / "fewer than N" → use < N  (NOT <= N)
    - "at most N" / "N or fewer" → use <= N
14. When the question asks for a single metric (count, sum, average), result_df must have exactly 1 row and 1 column — do NOT return detail rows."""


def _load_dataframe(path: Path) -> tuple[str, pd.DataFrame]:
    """加载单个 CSV 或 JSON 文件为 DataFrame，返回 (var_name, df)。"""
    suffix = path.suffix.lower()
    stem = _safe_varname(path.stem)

    if suffix == ".csv":
        df = pd.read_csv(path)
        return stem, df

    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        # Airtable 格式：{"table": "X", "records": [...]}
        if isinstance(raw, dict) and "records" in raw:
            records = raw["records"]
            df = pd.json_normalize(records)
            table_name = raw.get("table", stem)
            return _safe_varname(table_name), df
        # 普通 JSON 列表
        if isinstance(raw, list):
            df = pd.json_normalize(raw)
            return stem, df
        # 单层 dict
        df = pd.DataFrame([raw])
        return stem, df

    return stem, pd.DataFrame()


def _safe_varname(name: str) -> str:
    """将文件名转为合法 Python 变量名。"""
    import re
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if name and name[0].isdigit():
        name = "df_" + name
    return name or "df"


def _infer_join_hints(dfs: dict[str, pd.DataFrame]) -> list[str]:
    """推断可能需要 Join 的列对。"""
    hints: list[str] = []
    names = list(dfs.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            shared_cols = set(dfs[a].columns) & set(dfs[b].columns)
            for col in shared_cols:
                hints.append(f"  - `{a}` and `{b}` share column `{col}` (possible join key)")
    return hints


def _build_schema_text(
    dfs: dict[str, pd.DataFrame],
    paths: dict[str, Path] | None = None,
) -> str:
    parts: list[str] = []
    for var_name, df in dfs.items():
        path_hint = ""
        if paths and var_name in paths:
            p = paths[var_name]
            path_hint = f" — load with: pd.read_csv('{p}')" if p.suffix.lower() == ".csv" else f" — load with json.load(open('{p}'))"
        parts.append(f"\n### Table `{var_name}`{path_hint}")
        parts.append(f"  Columns: {', '.join(str(c) for c in df.columns)}")
        if not df.empty:
            sample = df.head(3).to_string(index=False, max_cols=20)
            parts.append(f"  Sample rows:\n{textwrap.indent(sample, '    ')}")
            # 枚举值展示：字符串列且 unique 值 ≤30，列出所有实际值（与 sql_solver 逻辑一致）
            enum_lines = []
            for col in df.columns:
                if df[col].dtype == object or str(df[col].dtype) in ('string', 'category'):
                    try:
                        unique_vals = df[col].dropna().unique()
                        if 2 <= len(unique_vals) <= 30:
                            vals_str = ', '.join(f"'{v}'" for v in sorted(str(v) for v in unique_vals))
                            enum_lines.append(f"  - `{col}` values: {vals_str}")
                    except Exception:
                        pass
            if enum_lines:
                parts.append("  Distinct categorical values (use EXACT spelling in filters):")
                parts.extend(enum_lines)
    return "\n".join(parts)


def _build_pandas_prompt(
    question: str,
    schema_text: str,
    join_hints: list[str],
    var_names: list[str],
    knowledge: KnowledgeContext,
    context_dir: Path | None = None,
) -> str:
    sections: list[str] = [
        f"## Available Data Files\n{schema_text}",
    ]

    if join_hints:
        sections.append("## Possible Join Keys\n" + "\n".join(join_hints))

    knowledge_section = knowledge.to_prompt_section()
    if knowledge_section.strip():
        sections.append(f"## Knowledge Context\n{knowledge_section}")

    sections.append(
        f"## Variable Names in Scope\n{', '.join(var_names)}"
        "\n(These DataFrames are already loaded. Do NOT use pd.read_csv or open().)"
    )
    sections.append(f"## Question\n{question}")
    sections.append(
        "## Python Code\n"
        "# Write pandas code using the loaded DataFrames. Final answer in `result_df`."
    )

    return "\n\n".join(sections)


def _run_pandas_code(
    code: str,
    dfs: dict[str, pd.DataFrame],
    context_dir: Path,
    inject_dfs: bool = False,
) -> tuple[pd.DataFrame | None, str | None]:
    """在 subprocess 中执行 pandas 代码，以 context_dir 为工作目录。

    inject_dfs=False（默认）：代码通过文件路径读取数据（真实路径，保留类型）
    inject_dfs=True：序列化 dfs 传入（doc 提取的内存 DataFrame 用此模式）
    """
    if inject_dfs:
        # 内存 DataFrame 模式：序列化传入
        dfs_json = {k: v.to_json(orient="records") for k, v in dfs.items()}
        prelude = textwrap.dedent(f"""
import json, sys, io
import pandas as pd

_dfs_json = {json.dumps(dfs_json)}
{chr(10).join(f'{k} = pd.read_json(io.StringIO(_dfs_json["{k}"]))' for k in dfs)}
""")
    else:
        # 文件路径模式：代码自己读文件
        prelude = textwrap.dedent(f"""
import json, sys, os
import pandas as pd

os.chdir({repr(str(context_dir.resolve()))})
""")

    runner_code = prelude + "\n# User code\n" + code + textwrap.dedent("""

# Output result
if 'result_df' in dir():
    print("__RESULT__" + result_df.to_json(orient="records", force_ascii=False))
else:
    print("__ERROR__result_df not defined", file=sys.stderr)
    sys.exit(1)
""")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(runner_code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(context_dir),
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if "__RESULT__" in stdout:
            json_str = stdout.split("__RESULT__", 1)[1]
            records = json.loads(json_str)
            df = pd.DataFrame(records)
            return df, None

        error_msg = stderr or stdout or "Unknown error"
        return None, error_msg

    except subprocess.TimeoutExpired:
        return None, "Execution timed out (30s)"
    except Exception as e:
        return None, str(e)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _extract_code(response: str) -> str:
    """从 LLM 响应中提取 Python 代码。"""
    response = re.sub(r"```(?:python)?\s*", "", response, flags=re.IGNORECASE)
    response = re.sub(r"```", "", response)
    return response.strip()


def _make_preload_code(dfs: dict[str, pd.DataFrame]) -> str:
    """生成把 dfs 序列化并注入的 prelude 代码（兼容旧模式）。"""
    dfs_json = {k: v.to_json(orient="records") for k, v in dfs.items()}
    lines = ["import json, sys, io", "import pandas as pd", f"_dfs_json = {json.dumps(dfs_json)}"]
    for k in dfs:
        lines.append(f'{k} = pd.read_json(io.StringIO(_dfs_json["{k}"]))')
    return "\n".join(lines)


# 单文件超过这个行数时，整个任务切换到 DuckDB（流式读取，不全量装内存）
_LARGE_FILE_ROW_THRESHOLD = 50_000


def _estimate_csv_rows(path: Path) -> int:
    """快速估算 CSV 行数：只读文件字节数 + 头部采样，不全量加载。"""
    try:
        file_size = path.stat().st_size
        if file_size == 0:
            return 0
        # 读头部 4KB 估算平均行长
        with path.open("rb") as f:
            head = f.read(4096)
        lines_in_head = head.count(b"\n")
        if lines_in_head <= 1:
            return 0
        avg_line_bytes = len(head) / lines_in_head
        return int(file_size / avg_line_bytes)
    except Exception:
        return 0


def solve_pandas(
    question: str,
    inv: AssetInventory,
    knowledge: KnowledgeContext,
    model: OpenAIModelAdapter,
) -> pd.DataFrame | None:
    """TYPE_PANDAS 路径主函数。

    大文件保护：任意 CSV 超过 _LARGE_FILE_ROW_THRESHOLD 行时，
    自动切换到 sql_solver（DuckDB 流式读取），避免全量加载 OOM。
    """
    all_files = inv.csv_files + inv.json_files
    if not all_files:
        return pd.DataFrame()

    # 大文件检测：估算行数，超阈值则降级到 DuckDB
    for csv_path in inv.csv_files:
        estimated_rows = _estimate_csv_rows(csv_path)
        if estimated_rows > _LARGE_FILE_ROW_THRESHOLD:
            import sys
            print(
                f"[pandas_solver] {csv_path.name} estimated {estimated_rows:,} rows "
                f"> threshold {_LARGE_FILE_ROW_THRESHOLD:,}, falling back to sql_solver",
                file=sys.stderr,
            )
            from data_agent_baseline.solvers.sql_solver import solve_sql
            return solve_sql(question, inv, knowledge, model)

    # 加载所有 DataFrame（用于 schema 描述和 join 推断）
    dfs: dict[str, pd.DataFrame] = {}
    paths: dict[str, Path] = {}

    for path in all_files:
        try:
            var_name, df = _load_dataframe(path)
            if var_name in dfs:
                var_name = var_name + "_" + path.stem[-4:]
            dfs[var_name] = df
            paths[var_name] = path
        except Exception:
            pass

    if not dfs:
        return pd.DataFrame()

    # context_dir：取所有文件的最近公共祖先
    context_dir = _common_context_dir(all_files)

    schema_text = _build_schema_text(dfs, paths)
    join_hints = _infer_join_hints(dfs)
    prompt = _build_pandas_prompt(
        question, schema_text, join_hints, list(dfs.keys()), knowledge, context_dir
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _PANDAS_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    response = model.chat(messages)
    code = _extract_code(response)

    # 使用 inject_dfs=True：序列化传入，保证路径问题不影响结果
    df, error = _run_pandas_code(code, dfs, context_dir, inject_dfs=True)

    # ── 内部辅助：对非空结果做语义复查，不合理则带诊断重新生成代码 ──────────
    def _verify_and_fix_pandas(current_df: pd.DataFrame, current_code: str, rounds: int = 2) -> pd.DataFrame:
        """对执行成功的非空结果做最多 rounds 轮语义复查+修复。"""
        from data_agent_baseline.solvers.sql_solver import _verify_result, _VERIFY_SYSTEM_PROMPT
        best_df = current_df
        best_code = current_code
        for _vround in range(rounds):
            try:
                verdict = _verify_result(question, best_code, best_df, {}, knowledge, model)
            except Exception:
                break
            if verdict.get("is_correct", True):
                break
            diagnosis = verdict.get("diagnosis", "")
            exp_rows = verdict.get("expected_rows")
            exp_cols = verdict.get("expected_cols")
            import sys
            print(f"[pandas_solver] verify round {_vround+1}: NOT correct — {diagnosis}", file=sys.stderr)
            rows_hint = f"\nExpected rows: {exp_rows}" if exp_rows is not None else ""
            cols_hint = f"\nExpected columns: {exp_cols}" if exp_cols is not None else ""
            fix_messages: list[dict[str, Any]] = [
                {"role": "system", "content": _PANDAS_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": best_code},
                {
                    "role": "user",
                    "content": (
                        f"The result is semantically incorrect.\n"
                        f"Diagnosis: {diagnosis}{rows_hint}{cols_hint}\n\n"
                        "The DataFrames are already loaded as variables — do NOT use pd.read_csv().\n"
                        "Fix the code based on the diagnosis. Output ONLY the corrected Python code."
                    ),
                },
            ]
            try:
                fixed_resp = model.chat(fix_messages)
                new_code = _extract_code(fixed_resp)
                new_df, _ = _run_pandas_code(new_code, dfs, context_dir, inject_dfs=True)
                if new_df is not None and not new_df.empty:
                    best_df = new_df
                    best_code = new_code
                else:
                    break
            except Exception:
                break
        return best_df

    if df is not None and not df.empty:
        return _verify_and_fix_pandas(df, code)

    # 第二次尝试：执行错误 OR 空结果，都触发 LLM 修复
    if error or (df is not None and df.empty):
        empty_hint = (
            "\n\nThe code returned an empty DataFrame. Common causes and fixes:\n"
            "1. Column name wrong — use EXACT names from schema (case-sensitive)\n"
            "2. Case mismatch in filter — use .str.lower() on both sides:\n"
            "   df[df['col'].str.lower() == 'value'.lower()]\n"
            "3. Type mismatch in merge — cast to same type:\n"
            "   df1['id'].astype(str).merge(df2['id'].astype(str))\n"
            "4. Filter value wrong — check actual values in sample rows above\n"
            "5. Date format mismatch — use .str.startswith() or .str.contains()\n"
            "6. AND condition too strict — verify all conditions hold simultaneously"
            if df is not None and df.empty else ""
        )
        fix_messages: list[dict[str, Any]] = [
            {"role": "system", "content": _PANDAS_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": code},
            {
                "role": "user",
                "content": (
                    f"The code failed with error:\n{error}\n{empty_hint}\n\n"
                    if error else
                    f"The code ran but returned empty results.{empty_hint}\n\n"
                ) + "The DataFrames are already loaded as variables — do NOT use pd.read_csv().\n"
                  "Please fix the code. Output ONLY the corrected Python code."
            },
        ]
        try:
            fixed_response = model.chat(fix_messages)
            fixed_code = _extract_code(fixed_response)
            df2, _ = _run_pandas_code(fixed_code, dfs, context_dir, inject_dfs=True)
            if df2 is not None and not df2.empty:
                return _verify_and_fix_pandas(df2, fixed_code)
            if df2 is not None:
                return df2
        except Exception:
            pass

    if df is not None:
        return df

    return pd.DataFrame()


def _common_context_dir(files: list[Path]) -> Path:
    """找所有文件的最近公共目录（context/ 级别）。"""
    if not files:
        return Path(".")
    resolved = [f.resolve() for f in files]
    # 找公共前缀
    common = resolved[0].parent
    for f in resolved[1:]:
        # 向上找直到两者都是子路径
        while not str(f).startswith(str(common)):
            common = common.parent
    return common
