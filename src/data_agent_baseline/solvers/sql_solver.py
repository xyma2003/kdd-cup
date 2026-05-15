"""
TYPE_SQL 路径：有 SQLite/DB 文件时，生成并执行 SQL。

流程：
1. 提取所有 DB 的 schema（表名、列名、类型、前5行样本）
2. 提取 knowledge SQL 示例作为 few-shot
3. 一次 LLM 调用生成 SQL
4. 执行 SQL → DataFrame
5. 失败时：规则修复 → 有限 LLM 重试
"""
from __future__ import annotations

import json
import re
import sqlite3
import traceback
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from data_agent_baseline.agents.model import OpenAIModelAdapter
from data_agent_baseline.knowledge.parser import KnowledgeContext
from data_agent_baseline.solvers.classifier import AssetInventory, DifficultyBudget


def _strip_markdown_fences(text: str) -> str:
    """去掉 LLM 响应里的 markdown 代码围栏（```json、```sql、``` 等）。"""
    text = re.sub(r"```(?:\w+)?\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r"```", "", text).strip()


def _find_matching_brace(text: str, start: int) -> int:
    """从 start 位置（必须是 '{' 字符）开始，用括号计数法找对应的 '}'。
    比 rfind('}') 更准确，不会被字符串内容里的 } 干扰。
    返回匹配的 } 的位置，找不到返回 -1。
    """
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        c = text[i]
        if in_string:
            if c == '\\':
                i += 2  # 跳过转义字符（含 \" ），不影响 in_string 状态
                continue
            if c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
            elif c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return -1


def _parse_llm_json(text: str) -> dict | None:
    """5层容错 JSON 解析器，处理 LLM 输出格式不规范的情况。

    Level 1: 标准 ```json ... ``` 代码块
    Level 2: 任意 ``` ... ``` 代码块
    Level 3: 直接 json.loads 整个响应（裸 JSON）
    Level 4: 找最外层 { + 括号匹配法找对应 }（比 rfind 更准确）
    Level 5: 正则逐字段提取关键字段（最后兜底）
    """
    if not text:
        return None
    cleaned = text.strip()

    # Level 1: ```json ... ```
    m = re.search(r"```json\s*\n?(.*?)\n?\s*```", cleaned, re.IGNORECASE | re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Level 2: 任意 ``` ... ```
    m = re.search(r"```\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Level 3: 直接解析整个响应
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Level 4: 找最外层 { } 用括号匹配法
    start = cleaned.find("{")
    if start != -1:
        end = _find_matching_brace(cleaned, start)
        if end != -1:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass
        # 括号匹配失败时降级到 rfind（原有逻辑）
        end_rfind = cleaned.rfind("}")
        if end_rfind != -1:
            try:
                return json.loads(cleaned[start:end_rfind + 1])
            except json.JSONDecodeError:
                pass

    # Level 5: 正则逐字段提取（专为 intent JSON 设计的兜底）
    result: dict = {}
    for key in ("aggregation", "group_by", "output_cols", "requires_distinct", "reasoning"):
        m = re.search(rf'"{key}"\s*:\s*("(?:[^"\\]|\\.)*"|\d+|true|false|null)', cleaned)
        if m:
            try:
                result[key] = json.loads(m.group(1))
            except json.JSONDecodeError:
                result[key] = m.group(1).strip('"')
    if result:
        return result

    return None


# ── Step 1: 语义理解 prompt ──────────────────────────────────────────────────

_INTENT_SYSTEM_PROMPT = """You are a data analysis intent extractor.

Your job: read the question carefully and output a structured JSON describing WHAT to compute.
Do NOT write SQL. Do NOT look at the schema yet. Focus only on understanding the question.

Output ONLY valid JSON in this exact format:
{
  "aggregation": "AVG" | "SUM" | "COUNT" | "MAX" | "MIN" | "RATIO" | "PERCENTAGE" | null,
  "aggregation_note": "<how exactly to compute it, e.g. 'AVG(col)/12 for monthly avg', 'COUNT DISTINCT'>",
  "filters": [{"description": "<what to filter>", "op": ">" | ">=" | "<" | "<=" | "==" | "IN", "note": "<exact condition from question>"}],
  "output_cols": <number of columns in final result>,
  "group_by": "<what to group by, or null>",
  "requires_distinct": true | false,
  "reasoning": "<one sentence: what the question is really asking>"
}

Rules:
- "more than N" / "greater than N" → op: ">"  (NEVER ">=")
- "at least N" / "N or more" → op: ">="
- "average monthly" / "average per month" → aggregation: "AVG", aggregation_note: "AVG(col) / N months (NOT row-level AVG() over groups)"
- "how many" / "count" → aggregation: "COUNT"
- "what is the X with highest/lowest Y" → output_cols: 1 (only X, not Y)
- If question asks for a single number → output_cols: 1
- output_cols should reflect ONLY what the question explicitly asks to return
- requires_distinct: set true when question asks for unique/distinct entities (e.g. "names of", "list of", "which X", "how many distinct/unique X"). Set false for aggregations over rows (AVG, SUM) or when duplicates are expected."""


_INTENT_REVIEW_SYSTEM_PROMPT = """You are a careful reviewer of data analysis intent.

You will be given:
1. The original question
2. A structured intent extracted from the question

Your job: ONLY fix clear, unambiguous errors. If you are not certain something is wrong, leave it unchanged.
Default behavior: return the intent unchanged. Only modify what you are 100% sure is wrong.

The ONLY things you are allowed to fix:
- "more than N" / "greater than N" → op must be ">" not ">="
- "at least N" / "N or more" → op must be ">="
- "less than N" → op must be "<" not "<="
- "how many X" → aggregation must be COUNT (not null)

Do NOT change: aggregation_note, output_cols, group_by, reasoning, or filters beyond the op field.
When in doubt, do not change anything.

Output ONLY valid JSON (same format as input), no explanation."""


# ── Step 2: SQL 生成 prompt ───────────────────────────────────────────────────

_SQL_SYSTEM_PROMPT = """You are an expert SQL analyst using DuckDB. All tables (SQLite DB, CSV, JSON) are registered as views and can be queried together in a single SQL statement.

Rules:
1. Output ONLY the SQL query, no explanation, no markdown fences.
2. Use CAST(... AS REAL) for division to get decimal results.
3. Use strftime('%Y', date_column) or YEAR(date_column) for year extraction.
4. For percentage calculations: CAST(numerator AS REAL) * 100.0 / denominator
5. Use DISTINCT when the question asks for unique or distinct entities. Add DISTINCT if JOINs could produce duplicates.
6. Follow the exact style of the reference SQL examples if provided.
7. For "abnormal" values, use the thresholds defined in the field definitions.
8. Do NOT use SELECT * — select ONLY the exact columns asked for in the question.
9. Column count must match the question: one value asked → 1 column, two things → exactly 2 columns.
10. If "full name" and table has first_name/last_name columns, SELECT both as separate columns.
11. IMPORTANT: CSV and JSON tables are also available — you can JOIN them with SQLite tables in one query.
12. For LOWER(element) comparisons use lowercase values (e.g. LOWER(element) IN ('p', 'br')).
13. Never add id/record_id/primary key columns unless explicitly asked for.
14. For "average monthly" or "average per month": compute as total / number_of_months, NOT AVG() over rows.
    Example: "average monthly sales" → SELECT SUM(sales) / COUNT(DISTINCT month) FROM ...
14.5. When combining "average monthly" with a HAVING/WHERE filter on that derived value (e.g. "average monthly unit price > N"):
    Step 1: compute the per-entity average monthly value in a CTE/subquery.
    Step 2: filter on that computed value in the outer query.
    Do NOT try to filter on AVG() directly in HAVING without first dividing by month count.
    Example pattern:
      WITH monthly_avg AS (
        SELECT entity_id, SUM(price*qty) / COUNT(DISTINCT month) AS avg_monthly_price
        FROM orders GROUP BY entity_id
      )
      SELECT t.name FROM monthly_avg m JOIN entity t ON t.id = m.entity_id WHERE m.avg_monthly_price > N
15. Never add columns like id, record_id, primary key, or row number unless the question explicitly asks for them.
16. Comparison operators — be precise:
    - "more than N" / "greater than N" → use > N  (NOT >= N)
    - "at least N" / "N or more" → use >= N
    - "less than N" / "fewer than N" → use < N  (NOT <= N)
    - "at most N" / "N or fewer" → use <= N
17. When the question asks for a single metric (count, sum, average, etc.), return ONLY that aggregate value — do NOT return the underlying detail rows.
18. When the question asks "what is the [entity] with the highest/lowest [metric]?" — return ONLY the entity column, not the metric column. Example: "what is the comment with the highest score?" → SELECT Text (NOT Score).
19. When computing "per unit" / "unit price" / "price per X" via division (e.g. Price / Amount), always filter out rows where the denominator is 0 or NULL to avoid inf/NaN results.
20. When filtering by a derived condition (e.g. unit price > N) and then JOINing to another table: ALWAYS use a subquery/CTE to get DISTINCT entity IDs first, then JOIN. Pattern:
    WITH qualified AS (SELECT DISTINCT entity_id FROM ... WHERE <derived condition>)
    SELECT t2.col FROM other_table t2 JOIN qualified q ON t2.entity_id = q.entity_id WHERE ...
    NEVER do: SELECT ... FROM table1 JOIN table2 ON ... WHERE <derived condition> — this produces duplicate rows when one entity has multiple qualifying records.
21. Return ONLY the columns the question explicitly asks for. If the question asks for an attribute of some entities (e.g. "their consumption", "the score", "the name"), do NOT add entity ID columns or other unrequested columns alongside it."""


def _build_schema_text(inv: AssetInventory) -> str:
    """提取所有 DB 文件和 CSV 文件的 schema 和样本数据。"""
    parts: list[str] = []

    for db_path in inv.db_files:
        parts.append(f"### SQLite Database: {db_path.name}")
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            try:
                cur = conn.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
                tables = [r[0] for r in cur.fetchall()]
                for table in tables:
                    cur.execute(f"PRAGMA table_info({table})")
                    cols = cur.fetchall()
                    col_defs = ", ".join(f"{c['name']} {c['type']}" for c in cols)
                    parts.append(f"\nTable `{table}` ({col_defs})")
                    try:
                        cur.execute(f"SELECT * FROM {table} LIMIT 3")
                        rows = cur.fetchall()
                        if rows:
                            col_names = [c["name"] for c in cols]
                            parts.append(f"  Sample rows ({', '.join(col_names)}):")
                            for row in rows:
                                parts.append(f"    {tuple(row)}")
                        # 枚举列检测：TEXT 列且不重复值少，展示实际值
                        for col_info in cols:
                            col_name = col_info["name"]
                            col_type = (col_info["type"] or "").upper()
                            if "TEXT" in col_type or col_type == "":
                                try:
                                    cur.execute(f"SELECT COUNT(DISTINCT \"{col_name}\") FROM \"{table}\"")
                                    n_unique = cur.fetchone()[0]
                                    if 2 <= n_unique <= 30:
                                        cur.execute(f"SELECT DISTINCT \"{col_name}\" FROM \"{table}\" WHERE \"{col_name}\" IS NOT NULL ORDER BY \"{col_name}\"")
                                        vals = [str(r[0]) for r in cur.fetchall()]
                                        vals_str = ', '.join(f"'{v}'" for v in vals)
                                        parts.append(f"  - `{col_name}` values: {vals_str}")
                                except Exception:
                                    pass
                    except Exception:
                        pass
            finally:
                conn.close()
        except Exception as e:
            parts.append(f"  (Error reading schema: {e})")

    for csv_path in inv.csv_files:
        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", csv_path.stem)
        parts.append(f"\n### CSV Table: `{safe_name}` (from {csv_path.name})")
        try:
            # 只读少量行用于 schema 展示和 YYYYMM 检测，避免大文件 OOM
            df_sample = pd.read_csv(csv_path, nrows=1000)
            col_notes = []
            for c in df_sample.columns:
                note = ""
                if df_sample[c].dtype in ['int64', 'float64']:
                    vals = df_sample[c].dropna().astype(int)
                    if len(vals) > 0 and vals.between(190001, 209912).all():
                        note = " [YYYYMM format: year=value//100, month=value%100]"
                col_notes.append(f"{c}{note}")
            parts.append(f"  Columns: {', '.join(col_notes)}")
            parts.append(f"  Sample rows:\n{df_sample.head(3).to_string(index=False)}")

            # 枚举值用 DuckDB 全量扫描，不受采样行数限制（流式读取，不 OOM）
            enum_notes = []
            try:
                conn_enum = duckdb.connect()
                abs_path = str(csv_path.resolve())
                for c in df_sample.columns:
                    if str(df_sample[c].dtype) in ('object', 'string', 'str', 'category'):
                        try:
                            n_unique = conn_enum.execute(
                                f"SELECT COUNT(DISTINCT \"{c}\") FROM read_csv_auto('{abs_path}')"
                            ).fetchone()[0]
                            if 2 <= n_unique <= 30:
                                vals = conn_enum.execute(
                                    f"SELECT DISTINCT \"{c}\" FROM read_csv_auto('{abs_path}') "
                                    f"WHERE \"{c}\" IS NOT NULL ORDER BY \"{c}\""
                                ).fetchall()
                                vals_str = ', '.join(f"'{r[0]}'" for r in vals)
                                enum_notes.append(f"  - `{c}` values: {vals_str}")
                        except Exception:
                            pass
                conn_enum.close()
            except Exception:
                pass

            if enum_notes:
                parts.append("  Distinct categorical values (use EXACT spelling in WHERE clauses):")
                parts.extend(enum_notes)
        except Exception as e:
            parts.append(f"  (Error: {e})")

    for json_path in inv.json_files:
        try:
            raw = json.loads(json_path.read_text())
            table_name = raw.get("table", json_path.stem) if isinstance(raw, dict) else json_path.stem
            safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", table_name)
            records = raw.get("records", raw) if isinstance(raw, dict) else raw
            parts.append(f"\n### JSON Table: `{safe_name}` (from {json_path.name})")
            if isinstance(records, list) and records:
                df = pd.json_normalize(records[:3])
                parts.append(f"  Columns: {', '.join(df.columns[:10])}")
                parts.append(f"  Sample rows:\n{df.to_string(index=False)}")
        except Exception as e:
            parts.append(f"\n### JSON file: {json_path.name}  (Error: {e})")

    return "\n".join(parts)


def _build_sql_prompt(
    question: str,
    schema_text: str,
    knowledge: KnowledgeContext,
) -> str:
    sections: list[str] = [
        f"## Database Schema\n{schema_text}",
    ]

    knowledge_section = knowledge.to_prompt_section()
    if knowledge_section.strip():
        sections.append(f"## Knowledge Context\n{knowledge_section}")

    sections.append(f"## Question\n{question}")
    sections.append("## SQL Query")

    return "\n\n".join(sections)


def _execute_with_duckdb(inv: AssetInventory, sql: str) -> pd.DataFrame:
    """用 DuckDB 执行 SQL，支持跨 SQLite + CSV + JSON 联合查询。

    所有表都注册为裸名视图，LLM 可以直接用表名查询。
    """
    conn = duckdb.connect()
    try:
        # 把 SQLite DB 里的每张表都注册为 DuckDB 视图（裸表名）
        for db_path in inv.db_files:
            db_alias = re.sub(r"[^a-zA-Z0-9_]", "_", db_path.stem)
            conn.execute(f"ATTACH '{db_path}' AS {db_alias} (TYPE SQLITE)")
            # 枚举表名，逐一创建同名视图
            tables = conn.execute(
                f"SELECT table_name FROM information_schema.tables WHERE table_catalog='{db_alias}' AND table_schema='main'"
            ).fetchall()
            for (tbl,) in tables:
                safe_tbl = re.sub(r"[^a-zA-Z0-9_]", "_", tbl)
                try:
                    conn.execute(f'CREATE VIEW "{safe_tbl}" AS SELECT * FROM {db_alias}."{tbl}"')
                except Exception:
                    pass  # 视图名冲突时跳过

        # 注册所有 CSV 为视图（用文件 stem 作为表名，绝对路径）
        for csv_path in inv.csv_files:
            safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", csv_path.stem)
            abs_path = str(csv_path.resolve())
            try:
                conn.execute(f"CREATE VIEW \"{safe_name}\" AS SELECT * FROM read_csv_auto('{abs_path}')")
            except Exception:
                pass

        # 注册所有 JSON 为视图
        for json_path in inv.json_files:
            try:
                raw = json.loads(json_path.read_text())
                records = raw.get("records", raw) if isinstance(raw, dict) else raw
                table_name = raw.get("table", json_path.stem) if isinstance(raw, dict) else json_path.stem
                safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", table_name)
                if isinstance(records, list):
                    df_tmp = pd.json_normalize(records)
                    conn.register(safe_name, df_tmp)
            except Exception:
                pass

        return conn.execute(sql).df()
    finally:
        conn.close()


def _try_execute_sql(inv: AssetInventory, sql: str) -> tuple[pd.DataFrame | None, str | None]:
    """先用 DuckDB 跨源执行，失败则逐个 SQLite 尝试。"""
    # 尝试 DuckDB（支持跨 CSV+DB）
    try:
        df = _execute_with_duckdb(inv, sql)
        return df, None
    except Exception as duck_err:
        pass

    # 降级：逐个 SQLite
    last_error = None
    for db_path in inv.db_files:
        try:
            conn = sqlite3.connect(str(db_path))
            df = pd.read_sql_query(sql, conn)
            conn.close()
            return df, None
        except Exception as e:
            last_error = str(e)
    return None, last_error


def _extract_sql_from_response(text: str) -> str:
    """从 LLM 响应中提取 SQL（去掉 markdown 代码块等）。"""
    # 去掉 ```sql ... ``` 或 ``` ... ```
    text = re.sub(r"```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    return text.strip()


def _try_fix_sql_rules(sql: str, schema_text: str, error: str) -> str | None:
    """规则修复 SQL（不调 LLM）。"""
    # 如果是列名/表名错误，尝试从 schema 中模糊匹配
    if "no such column" in error.lower() or "no such table" in error.lower():
        # 提取出错的名称
        match = re.search(r"no such (?:column|table):\s*(\S+)", error, re.IGNORECASE)
        if match:
            bad_name = match.group(1).strip("\"'`")
            # 从 schema 中找最相似的名称
            candidates = re.findall(r"`(\w+)`", schema_text)
            best = _fuzzy_match(bad_name, candidates)
            if best and best.lower() != bad_name.lower():
                fixed = re.sub(
                    r"\b" + re.escape(bad_name) + r"\b",
                    best,
                    sql,
                    flags=re.IGNORECASE,
                )
                return fixed

    return None


def _fuzzy_match(target: str, candidates: list[str]) -> str | None:
    """简单模糊匹配：找编辑距离最近的候选。"""
    if not candidates:
        return None
    target_lower = target.lower()
    # 先找包含关系
    for c in candidates:
        if target_lower in c.lower() or c.lower() in target_lower:
            return c
    # 再找首字母匹配
    for c in candidates:
        if c.lower().startswith(target_lower[:3]):
            return c
    return None


def _build_fix_prompt(
    question: str,
    schema_text: str,
    knowledge: KnowledgeContext,
    failed_sql: str,
    error: str,
) -> str:
    extra_hint = ""
    if "empty" in error.lower():
        extra_hint = (
            "\n\n## Diagnosis: Empty Result\n"
            "The query returned no rows. Common causes and fixes:\n"
            "1. Case mismatch — use LOWER()/UPPER() or LIKE:\n"
            "   WHERE LOWER(col) = LOWER('value')  or  WHERE col LIKE '%value%'\n"
            "2. String filter too strict — use wildcards:\n"
            "   WHERE name LIKE '%Riverside%' instead of WHERE name = 'Riverside'\n"
            "3. Type mismatch in JOIN — cast to same type:\n"
            "   JOIN ON CAST(a.id AS TEXT) = CAST(b.id AS TEXT)\n"
            "4. Date format mismatch — use LIKE or strftime:\n"
            "   WHERE date LIKE '2013-06%'  or  WHERE strftime('%Y-%m', date) = '2013-06'\n"
            "5. AND conditions too strict — check if OR should be used instead\n"
            "6. Wrong column — check schema for where the value actually lives\n"
            "Fix the most likely cause based on the question and schema."
        )
    elif "duplicate" in error.lower() or "too many rows" in error.lower():
        extra_hint = (
            "\n\n## Diagnosis: Duplicate / Too Many Rows\n"
            "The query returned more rows than expected. Common causes:\n"
            "1. JOIN without DISTINCT — one entity matches multiple rows in the joined table.\n"
            "   Fix: use a CTE to get DISTINCT entity_ids first, then JOIN:\n"
            "   WITH q AS (SELECT DISTINCT entity_id FROM t WHERE ...) SELECT t2.col FROM t2 JOIN q ON ...\n"
            "2. Missing DISTINCT on the final SELECT — add SELECT DISTINCT.\n"
            "3. Wrong aggregation level — compute aggregate per entity in a subquery, not row-level.\n"
            "Fix the most likely cause based on the question and schema."
        )
    return (
        f"## Database Schema\n{schema_text}\n\n"
        f"## Knowledge Context\n{knowledge.to_prompt_section()}\n\n"
        f"## Question\n{question}\n\n"
        f"## Previous SQL (failed)\n{failed_sql}\n\n"
        f"## Error\n{error}"
        f"{extra_hint}\n\n"
        "## Fixed SQL Query (output ONLY the SQL, no explanation)"
    )


_VERIFY_SYSTEM_PROMPT = """You are a strict data analysis output verifier.

You will be given:
1. The original question
2. The SQL or code that was generated
3. The execution result (row count, column count, sample rows)
4. The query intent (aggregation type, group_by, output_cols)
5. Relevant knowledge context (SQL examples, field definitions)

Your job: decide if the result correctly answers the question.

Output ONLY valid JSON:
{
  "is_correct": true | false,
  "diagnosis": "<one sentence: what is wrong, or 'looks correct'>",
  "expected_rows": <integer or null if unknown>,
  "expected_cols": <integer or null if unknown>
}

Rules for judging:
- "how many" / "count" / "calculate" / "what is the" → expect 1 row, 1 col
- "list" / "which" / "names of" / "identify" → expect multiple rows, be CONSERVATIVE (do NOT flag as wrong just because there are many rows)
- "percentage" / "ratio" / "average" / "total" → expect 1 row, 1 col
- If group_by is NOT null → multiple rows are expected (one per group)
- If group_by IS null AND aggregation is not null → expect exactly 1 row
- NEVER flag a result as wrong just because the row count seems high — only flag if you are CERTAIN it's wrong based on the question semantics
- If unsure → set is_correct: true (conservative — do not break working results)"""


def _verify_result(
    question: str,
    query: str,
    df: pd.DataFrame,
    intent: dict,
    knowledge: KnowledgeContext,
    model: OpenAIModelAdapter,
) -> dict:
    """复查执行结果是否语义正确，返回 {is_correct, diagnosis, expected_rows, expected_cols}。"""
    sample_rows = df.head(3).to_dict(orient="records")
    intent_summary = {
        "aggregation": intent.get("aggregation"),
        "group_by": intent.get("group_by"),
        "output_cols": intent.get("output_cols"),
        "reasoning": intent.get("reasoning", ""),
    }
    knowledge_section = knowledge.to_prompt_section()
    # 只取 knowledge 的 SQL 示例部分，避免 prompt 过长
    knowledge_snippet = "\n".join(
        line for line in knowledge_section.splitlines()
        if any(kw in line for kw in ["SQL", "Example", "SELECT", "COUNT", "AVG", "SUM", "->", "→"])
    )[:1500]

    user_content = (
        f"## Question\n{question}\n\n"
        f"## Query / Code\n{query}\n\n"
        f"## Execution Result\n"
        f"- Rows returned: {len(df)}\n"
        f"- Columns: {list(df.columns)}\n"
        f"- Sample rows (first 3): {sample_rows}\n\n"
        f"## Query Intent\n{json.dumps(intent_summary, ensure_ascii=False)}\n\n"
        f"## Knowledge Context (relevant excerpts)\n{knowledge_snippet or '(none)'}\n\n"
        "Is this result correct? Output ONLY the JSON."
    )
    try:
        response = model.chat([
            {"role": "system", "content": _VERIFY_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ])
        parsed = _parse_llm_json(response)
        if parsed:
            return parsed
    except Exception:
        pass
    # 解析失败时保守返回 correct，不破坏已有结果
    return {"is_correct": True, "diagnosis": "verification parse failed", "expected_rows": None, "expected_cols": None}


def _build_verify_fix_prompt(
    question: str,
    schema_text: str,
    knowledge: KnowledgeContext,
    failed_sql: str,
    diagnosis: str,
    expected_rows: int | None,
    expected_cols: int | None,
) -> str:
    """基于复查诊断结果构建修复 prompt。"""
    rows_hint = f"Expected rows: {expected_rows}" if expected_rows is not None else ""
    cols_hint = f"Expected columns: {expected_cols}" if expected_cols is not None else ""
    return (
        f"## Database Schema\n{schema_text}\n\n"
        f"## Knowledge Context\n{knowledge.to_prompt_section()}\n\n"
        f"## Question\n{question}\n\n"
        f"## Previous SQL (semantically incorrect)\n{failed_sql}\n\n"
        f"## Diagnosis\n{diagnosis}\n"
        f"{rows_hint}\n{cols_hint}\n\n"
        "Fix the SQL based on the diagnosis above. "
        "Output ONLY the corrected SQL, no explanation."
    )


def _extract_intent(question: str, model: OpenAIModelAdapter) -> dict:
    """Step 1: 从问题中提取结构化查询意图（不看 schema）。"""
    response = model.chat([
        {"role": "system", "content": _INTENT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}"},
    ])
    parsed = _parse_llm_json(response)
    return parsed if parsed else {}


def _review_intent(question: str, intent: dict, model: OpenAIModelAdapter) -> dict:
    """Step 1.5: 验证并修正意图（reviewer 视角）。"""
    intent_str = json.dumps(intent, ensure_ascii=False, indent=2)
    response = model.chat([
        {"role": "system", "content": _INTENT_REVIEW_SYSTEM_PROMPT},
        {"role": "user", "content": f"Original question: {question}\n\nExtracted intent:\n{intent_str}"},
    ])
    parsed = _parse_llm_json(response)
    return parsed if parsed else intent  # 解析失败时保留原意图


def _build_sql_prompt_with_intent(
    question: str,
    schema_text: str,
    knowledge: KnowledgeContext,
    intent: dict,
) -> str:
    """Step 2: 基于意图 + schema 构建 SQL 生成 prompt。"""
    intent_str = json.dumps(intent, ensure_ascii=False, indent=2)
    sections = [
        f"## Query Intent (verified — follow this exactly)\n```json\n{intent_str}\n```",
        f"## Database Schema\n{schema_text}",
    ]
    knowledge_section = knowledge.to_prompt_section()
    if knowledge_section.strip():
        sections.append(f"## Knowledge Context\n{knowledge_section}")
    sections.append(f"## Question\n{question}")
    group_by_val = intent.get("group_by")
    group_by_rule = (
        "CRITICAL: intent.group_by is null — the query asks for a GLOBAL aggregate (single row result). "
        "Do NOT use GROUP BY anywhere in the SQL. Use subqueries or CTEs with HAVING to apply filters, "
        "then compute the final aggregate over the filtered set without GROUP BY."
        if not group_by_val
        else f"The query must GROUP BY: {group_by_val}"
    )
    distinct_val = intent.get("requires_distinct", False)
    distinct_rule = (
        "CRITICAL: intent.requires_distinct is true — use SELECT DISTINCT to avoid duplicate rows in the result."
        if distinct_val
        else "intent.requires_distinct is false — do NOT add DISTINCT unless a JOIN would clearly produce duplicates."
    )
    sections.append(
        "## SQL Query\n"
        "# Write SQL that implements the intent above. "
        "If intent says aggregation=AVG, your SELECT must use AVG(). "
        "If intent says output_cols=1, SELECT exactly 1 column. "
        "If intent says op='>', use > (not >=). "
        f"{group_by_rule} "
        f"{distinct_rule}"
    )
    return "\n\n".join(sections)


def _drop_id_columns(df: pd.DataFrame, expected_cols: int) -> pd.DataFrame:
    """当实际列数多于期望时，丢掉明显是 ID 类的列。

    仅在 intent 明确指定了列数（expected_cols > 0）且实际列数超出时触发，
    且只有在去掉 ID 列后恰好等于期望列数时才执行，避免误删。
    """
    if df is None or df.empty or expected_cols <= 0 or len(df.columns) <= expected_cols:
        return df
    id_patterns = re.compile(r"(^id$|_id$|^customerid$|^userid$|^recordid$)", re.IGNORECASE)
    non_id_cols = [c for c in df.columns if not id_patterns.search(c)]
    if len(non_id_cols) == expected_cols:
        return df[non_id_cols]
    return df


def solve_sql(
    question: str,
    inv: AssetInventory,
    knowledge: KnowledgeContext,
    model: OpenAIModelAdapter,
    budget: DifficultyBudget | None = None,
) -> pd.DataFrame | None:
    """TYPE_SQL 路径主函数（3步：意图提取 → 意图校验 → SQL生成执行）。"""
    schema_text = _build_schema_text(inv)
    max_repair_rounds = budget.max_repair_rounds if budget else 1

    # Step 1: 理解查询意图（不看 schema）
    intent = _extract_intent(question, model)

    # Step 1.5: 校验并修正意图（easy 任务跳过，节省 1 次 LLM）
    if intent and not (budget and budget.skip_intent_review):
        intent = _review_intent(question, intent, model)

    # Step 2: 基于意图生成 SQL
    if intent:
        prompt = _build_sql_prompt_with_intent(question, schema_text, knowledge, intent)
    else:
        # 意图提取失败，降级到原来的单步生成
        prompt = _build_sql_prompt(question, schema_text, knowledge)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SQL_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    response = model.chat(messages)
    sql = _extract_sql_from_response(response)

    # 尝试执行（DuckDB 跨源 → SQLite 降级）
    df, error = _try_execute_sql(inv, sql)

    expected_cols = intent.get("output_cols", 0) if intent else 0

    # 检测 COUNT 返回 0 的情况（SQL 执行成功但语义错误）
    def _is_zero_count(df: pd.DataFrame | None) -> bool:
        if df is None or df.empty:
            return False
        if len(df) == 1 and len(df.columns) == 1:
            val = df.iloc[0, 0]
            try:
                return float(val) == 0.0
            except (ValueError, TypeError):
                return False
        return False

    # ── 内部辅助：对非空结果做语义复查，不合理则带诊断重新生成 SQL ──────────
    def _verify_and_fix(current_df: pd.DataFrame, current_sql: str, rounds: int = 2) -> pd.DataFrame:
        """对执行成功的非空结果做最多 rounds 轮语义复查+修复。
        复查认为正确 → 直接返回；复查认为不对 → 带诊断重新生成 SQL 并重新执行。
        任何一轮验证失败时保守返回上一个结果，不强行破坏。
        相同 SQL 两次生成 → 停止（卡死保护）。
        """
        best_df = current_df
        best_sql = current_sql
        seen_sqls: set[str] = {current_sql.strip()}
        for _vround in range(rounds):
            try:
                verdict = _verify_result(question, best_sql, best_df, intent or {}, knowledge, model)
            except Exception:
                break
            if verdict.get("is_correct", True):
                break
            diagnosis = verdict.get("diagnosis", "")
            exp_rows = verdict.get("expected_rows")
            exp_cols = verdict.get("expected_cols")
            print(f"[sql_solver] verify round {_vround+1}: NOT correct — {diagnosis}", file=sys.stderr)
            fix_p = _build_verify_fix_prompt(
                question, schema_text, knowledge, best_sql, diagnosis, exp_rows, exp_cols
            )
            try:
                fixed_resp = model.chat([
                    {"role": "system", "content": _SQL_SYSTEM_PROMPT},
                    {"role": "user", "content": fix_p},
                ])
                new_sql = _extract_sql_from_response(fixed_resp)
                # 卡死保护：如果生成了相同的 SQL，停止重试
                if new_sql.strip() in seen_sqls:
                    print(f"[sql_solver] verify round {_vround+1}: same SQL generated, stopping", file=sys.stderr)
                    break
                seen_sqls.add(new_sql.strip())
                new_df, new_err = _try_execute_sql(inv, new_sql)
                if new_df is not None and not new_df.empty:
                    best_df = _drop_id_columns(new_df, expected_cols)
                    best_sql = new_sql
                else:
                    # 修复后空结果或报错，保留原结果，停止复查
                    break
            except Exception:
                break
        return best_df

    if df is not None and not df.empty:
        result = _drop_id_columns(df, expected_cols)
        return _verify_and_fix(result, sql)

    # 规则修复
    if error or (df is not None and df.empty):
        fixed_sql = _try_fix_sql_rules(sql, schema_text, error or "empty result")
        if fixed_sql and fixed_sql != sql:
            df2, error2 = _try_execute_sql(inv, fixed_sql)
            if df2 is not None and not df2.empty:
                result2 = _drop_id_columns(df2, expected_cols)
                return _verify_and_fix(result2, fixed_sql)
            if df2 is not None and error2 is None:
                return df2

    zero_count_result = _is_zero_count(df)

    # LLM 修复循环（轮数由难度预算决定：easy=1, medium=1, hard/extreme=2）
    empty_result = df is not None and df.empty
    current_sql = sql
    for _repair_round in range(max_repair_rounds):
        if not (error or zero_count_result or empty_result):
            break
        if zero_count_result:
            error_msg = (
                "The query returned COUNT = 0, which is likely wrong.\n"
                "Please verify:\n"
                "1. Are you using ALL available tables? "
                f"Available: {', '.join([p.name for p in inv.db_files] + [p.stem for p in inv.csv_files] + [p.stem for p in inv.json_files])}\n"
                "2. Check JOIN conditions — the result should be non-zero.\n"
                "3. If filtering by element/label, check exact values (case, spelling)."
            )
        elif empty_result:
            error_msg = "empty result"
        else:
            error_msg = error or "unknown error"
        fix_prompt = _build_fix_prompt(question, schema_text, knowledge, current_sql, error_msg)
        fix_messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SQL_SYSTEM_PROMPT},
            {"role": "user", "content": fix_prompt},
        ]
        try:
            fixed_response = model.chat(fix_messages)
            fixed_sql_n = _extract_sql_from_response(fixed_response)
            df_n, error_n = _try_execute_sql(inv, fixed_sql_n)
            if df_n is not None and not df_n.empty:
                result_n = _drop_id_columns(df_n, expected_cols)
                return _verify_and_fix(result_n, fixed_sql_n)
            # 更新状态，进入下一轮修复
            current_sql = fixed_sql_n
            df = df_n
            error = error_n
            zero_count_result = _is_zero_count(df_n)
            empty_result = df_n is not None and df_n.empty
        except Exception:
            break

    if df is not None:
        return _drop_id_columns(df, expected_cols)

    return pd.DataFrame()
