"""
structured_solver：TYPE_HYBRID / TYPE_DOC 任务的状态机驱动 pipeline。

Pipeline 状态机：
  EXTRACT_ENTITIES
    → 从 doc 文件中提取 ID→实体 映射（JSON 格式）
    → 输出：{entities: [{id, name, ...}], id_field}
  BUILD_QUERY
    → 基于实体 ID + CSV/DB schema，生成 Python 查询代码
    → 输出：Python 代码字符串
  EXECUTE (+ RETRY 最多 2 次)
    → subprocess 执行代码，捕获 result_df JSON
    → 失败则带错误信息重新 BUILD_QUERY
  DONE
    → 返回 AnswerTable
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from enum import Enum, auto
from pathlib import Path
from typing import Any

import pandas as pd

from data_agent_baseline.agents.model import OpenAIModelAdapter
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.knowledge.parser import KnowledgeContext
from data_agent_baseline.solvers.classifier import AssetInventory


# ── 常量 ────────────────────────────────────────────────────────────────────

# 小文档阈值：低于此值直接全文发给 LLM，高于此值走 BM25 筛选
_SMALL_DOC_THRESHOLD = 12_000

# 小文档全文提取时的分块大小（旧逻辑保留，供 _chunk_text 使用）
_CHUNK_SIZE_FOR_LLM = 3_000

# BM25 打分时的分块大小（每块字符数，带重叠）
_BM25_CHUNK_SIZE = 1_200
_BM25_CHUNK_OVERLAP = 150

# BM25 取 top-K 块
_BM25_TOP_K = 8

# 两个高分块之间的间距小于此值时，把中间内容一起带上（覆盖"先说后纠正"的情况）
_BM25_MERGE_GAP = 2_000

# 最终发给 LLM 的文本上限（字符数）
_BM25_MAX_SEND = 14_000

# JSON 预加载时单个文件的最大 records 数（防止超大 JSON 撑爆 prelude）
_MAX_JSON_RECORDS_PRELOAD = 500

# ── 状态枚举 ────────────────────────────────────────────────────────────────

class SolverState(Enum):
    EXTRACT_ENTITIES = auto()
    BUILD_QUERY      = auto()
    EXECUTE          = auto()
    DONE             = auto()
    FAILED           = auto()


MAX_RETRIES = 2  # BUILD_QUERY + EXECUTE 的最大重试次数


# ── Prompt 模板 ─────────────────────────────────────────────────────────────

_SYSTEM_EXTRACT = """You are a precise data extraction assistant.

Your job: extract a clean ID→name/value mapping from a noisy document.

The document contains entities described in verbose prose. Each entity has an ID and a name or key attribute.
IDs can be in ANY format: Airtable-style (recXXXXXXXX), plain integers (Race ID: 14, ID 1655, identifier 26),
or other patterns. Extract ALL of them.

Output ONLY valid JSON in this exact format:
{
  "id_field": "<the column name in the structured data that links to these IDs, e.g. raceId, superhero_id, cards_id>",
  "id_to_name": {
    "<id>": "<clean_name_or_value>",
    ...
  }
}

Rules:
1. Extract the CLEAN name only — no "The program for", "The", "track for", prefixes.
   BAD: "The general Business program" → GOOD: "Business"
   BAD: "The IT Support and Web Development track" → GOOD: "IT Support and Web Development"
2. If a name was corrected ("initially X, corrected to Y"), use ONLY the final corrected name Y.
3. Extract ALL entities you can find (comprehensive) — both Airtable rec IDs and plain numeric IDs.
4. For numeric IDs, use the integer as the key (e.g. "14", "1655", "26").
5. The id_field should match a column name in the structured data (DB/CSV/JSON) that contains these ID values.
6. Output ONLY the JSON, no explanation, no markdown."""


_SYSTEM_QUERY = """You are an expert Python data analyst.

Write complete Python code to answer the question using pre-loaded variables.

Pre-loaded variables (already available — do NOT redefine them):
- `csv_paths`: dict {filename → path} — use pd.read_csv(csv_paths['file.csv'])
- `db_paths`:  dict {filename → path} — use sqlite3.connect(db_paths['file.db'])
- `{table}_df`: DataFrame pre-loaded from each JSON file (e.g. constructors_df, publisher_df)
  Use these directly — do NOT call json.load() or pd.read_json() on JSON files.
- `id_to_name`: dict {id → name} — pre-extracted mapping from documents
  Keys may be Airtable rec-IDs (e.g. "recABC123") OR plain integers as strings (e.g. "14", "18").
  When keys are numeric strings, join using: WHERE {id_field} = int(key)
- `id_field`: the column name in the DB/CSV that corresponds to the keys in id_to_name

The code must:
1. Load ONLY the files listed in the schema below — do NOT invent tables or files that are not listed
2. For JSON data: use the pre-loaded {table}_df variables directly (already DataFrames)
3. Use id_to_name to resolve IDs to names:
   - If keys are rec-IDs: df['name'] = df['link_col'].map(id_to_name)
   - If keys are numeric strings: filter by int(key) or use WHERE {id_field} IN (...)
4. Store the final answer in `result_df` (a pandas DataFrame)
5. Print: print("__RESULT__" + result_df.to_json(orient="records", force_ascii=False))

Rules:
- CRITICAL: Only use tables/files explicitly listed in the schema. Never query a table that is not in the schema.
- For JSON files, use the pre-loaded DataFrame variables (e.g. constructors_df) — never json.load()
- result_df must have meaningful column names (not 0, 1, 2)
- result_df must contain EXACTLY the columns the question asks for — no more, no less.
  If the question asks for one thing (e.g. "what is the major?"), result_df has 1 column.
  If the question asks for two things (e.g. "name and cost"), result_df has 2 columns.
- If question asks for "full name" but CSV has first_name/last_name, keep them as TWO separate columns
- Never include id/record_id/link_to_*/expense_description columns unless explicitly asked
- Never concatenate first_name + last_name — keep them separate
- Output ONLY the Python code, no markdown fences, no explanation."""


def _build_extract_prompt(
    question: str,
    doc_contents: dict[str, str],
    csv_headers: dict[str, list[str]],
    knowledge_text: str,
) -> str:
    parts = [f"## Question\n{question}\n"]

    if csv_headers:
        parts.append("## Available Data Columns (use these to determine id_field)")
        for fname, cols in csv_headers.items():
            parts.append(f"  {fname}: {', '.join(cols)}")

    if knowledge_text.strip():
        parts.append(f"## Knowledge Context\n{knowledge_text}")

    # 小文档直接全文；大文档由调用方负责分块，这里只接收已经合适大小的内容
    parts.append("## Documents to Extract From")
    for fname, content in doc_contents.items():
        parts.append(f"\n### {fname}\n{content}")

    parts.append("\n## Extract JSON")
    return "\n\n".join(parts)


def _chunk_text(text: str, chunk_size: int = _CHUNK_SIZE_FOR_LLM) -> list[str]:
    """按段落切块，每块不超过 chunk_size 字符。"""
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            if len(para) > chunk_size:
                for i in range(0, len(para), chunk_size):
                    chunks.append(para[i: i + chunk_size])
                current = ""
            else:
                current = para
    if current:
        chunks.append(current)
    return chunks


def _tokenize(text: str) -> list[str]:
    """小写词级 tokenize（与 lyp BM25 评分逻辑一致）。"""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def _bm25_score(chunk_tokens: list[str], query_tokens: set[str]) -> float:
    """关键词重叠率：命中 query token 数 / query token 总数。"""
    if not query_tokens:
        return 0.0
    return sum(1 for t in chunk_tokens if t in query_tokens) / len(query_tokens)


def _bm25_select_spans(content: str, question: str) -> str:
    """用 BM25 关键词打分从大文档中筛选最相关的段落，返回拼合后的文本。

    流程：
    1. 按固定大小切块（带重叠，保留跨块上下文）
    2. 每块对问题关键词打分
    3. 取 top-K 块，按原文位置排序
    4. 合并相邻/间距小的块（覆盖"先陈述后纠正"的情况）
    5. 截断到最大发送上限
    """
    # Step 1: 切块（带重叠）
    chunks: list[dict] = []
    start = 0
    length = len(content)
    while start < length:
        end = min(start + _BM25_CHUNK_SIZE, length)
        chunks.append({"text": content[start:end], "start": start, "end": end})
        if end == length:
            break
        start = end - _BM25_CHUNK_OVERLAP

    if not chunks:
        return content[:_BM25_MAX_SEND]

    # Step 2: BM25 打分
    q_tokens = set(_tokenize(question))
    for chunk in chunks:
        chunk["score"] = _bm25_score(_tokenize(chunk["text"]), q_tokens)

    # Step 3: 取 top-K，按原文位置排序
    top_chunks = sorted(
        sorted(chunks, key=lambda c: c["score"], reverse=True)[:_BM25_TOP_K],
        key=lambda c: c["start"],
    )

    if not top_chunks:
        return content[:_BM25_MAX_SEND]

    # Step 4: 合并相邻块（间距 < _BM25_MERGE_GAP 的块把中间内容也带上）
    merged_spans: list[tuple[int, int]] = []
    cur_start = top_chunks[0]["start"]
    cur_end = top_chunks[0]["end"]

    for chunk in top_chunks[1:]:
        if chunk["start"] - cur_end <= _BM25_MERGE_GAP:
            # 间距足够小，延伸当前 span（包含中间内容）
            cur_end = chunk["end"]
        else:
            merged_spans.append((cur_start, cur_end))
            cur_start = chunk["start"]
            cur_end = chunk["end"]
    merged_spans.append((cur_start, cur_end))

    # Step 5: 拼合并截断
    parts = [content[s:e] for s, e in merged_spans]
    result = "\n\n[...]\n\n".join(parts)
    return result[:_BM25_MAX_SEND]


def _llm_extract_large_doc(
    fname: str,
    content: str,
    question: str,
    csv_headers: dict[str, list[str]],
    knowledge_text: str,
    model: OpenAIModelAdapter,
) -> dict[str, str]:
    """大文档实体提取：BM25 筛选相关段落 → 单次 LLM 调用。

    替换原来的"盲目分块 + 并行多次 LLM"方案，减少 API 调用次数并降低超时风险。
    """
    selected = _bm25_select_spans(content, question)
    print(
        f"[structured] BM25 selected {len(selected)}/{len(content)} chars from {fname}",
        file=sys.stderr,
    )
    prompt = _build_extract_prompt(question, {fname: selected}, csv_headers, knowledge_text)
    try:
        response = _llm(model, _SYSTEM_EXTRACT, prompt)
        result = _parse_json_response(response)
        if result and isinstance(result.get("id_to_name"), dict):
            return result["id_to_name"]
    except Exception:
        pass
    return {}


def _build_query_prompt(
    question: str,
    entities_json: dict,
    file_schemas: str,
    knowledge_text: str,
    doc_contents: dict[str, str] = {},  # kept for signature compat, no longer used in prompt
    prev_error: str | None = None,
) -> str:
    parts = [
        f"## Question\n{question}",
        f"## Available Data Files (schemas and access instructions)\n{file_schemas}",
    ]

    if knowledge_text.strip():
        parts.append(f"## Knowledge Context\n{knowledge_text}")

    if prev_error:
        parts.append(
            f"## Previous Attempt Failed\nError:\n{prev_error}\n\n"
            "Fix the code. Common issues:\n"
            "- ID values in CSV may have different format — use str(id).strip() when matching\n"
            "- If filter returns empty, check id_to_name has the expected keys\n"
            "- For doc→CSV joins: df['col'] = df['link_col'].map(id_to_name)"
        )

    # id_to_name 字典是唯一的 doc 信息来源，不再传 doc 原文
    id_to_name_preview = ""
    if entities_json.get("id_to_name"):
        id_map = entities_json["id_to_name"]
        preview_items = list(id_map.items())[:5]
        total = len(id_map)
        # 判断 key 类型：是否为数字字符串
        sample_keys = list(id_map.keys())[:3]
        keys_are_numeric = bool(sample_keys) and all(k.strip().lstrip('-').isdigit() for k in sample_keys if k)
        id_field = entities_json.get("id_field") or "(unknown)"
        if keys_are_numeric:
            key_type_hint = (
                f"NUMERIC string keys (e.g. {sample_keys[:2]}) — "
                f"join with: WHERE {id_field} = int(key)  OR  df[df['{id_field}'].astype(str).isin(id_to_name)]"
            )
        else:
            key_type_hint = (
                f"rec-ID string keys (e.g. {sample_keys[:1]}) — "
                f"join with: df['name'] = df['{id_field}'].map(id_to_name)"
            )
        id_to_name_preview = (
            f"\n\nid_to_name: {total} entries, id_field='{id_field}'\n"
            f"Key type: {key_type_hint}\n"
            "First 5 entries:\n"
            + "\n".join(f"  {k!r}: {v!r}" for k, v in preview_items)
        )

    parts.append(
        "## Python Code\n"
        "# IMPORTANT STRATEGY:\n"
        "#\n"
        "# Case A — multiple CSVs, one has name columns (first_name, last_name):\n"
        "#   → Join CSVs; get name from the CSV that has first_name/last_name columns\n"
        "#\n"
        "# Case B — rec-ID keys in id_to_name, name only available via doc:\n"
        "#   → df['name'] = df['link_col'].map(id_to_name)\n"
        "#\n"
        "# Case C — NUMERIC keys in id_to_name (raceId, superhero_id, etc.):\n"
        "#   → Use the numeric ID to filter/join the DB directly:\n"
        "#       valid_ids = [int(k) for k in id_to_name]\n"
        "#       df = df[df['id_field'].isin(valid_ids)]\n"
        "#   → Then look up the name: df['name'] = df['id_field'].astype(str).map(id_to_name)\n"
        + id_to_name_preview
    )
    return "\n\n".join(parts)


# ── 文件扫描辅助 ─────────────────────────────────────────────────────────────

def _read_doc_files(inv: AssetInventory) -> dict[str, str]:
    """读取所有 doc 文件内容，不截断（全文供 Python 正则使用）。"""
    result = {}
    for path in inv.doc_files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            result[path.name] = text
        except Exception:
            pass
    return result


def _python_extract_id_mapping(doc_contents: dict[str, str]) -> dict[str, str]:
    """
    Python 侧精准提取所有 rec_id → name 映射，不依赖 LLM。

    覆盖多种文档格式：
    - "Name (Registry ID: recXXX)"
    - "identifier recXXX ... is Name"
    - "recXXX is Name" / "recXXX, identified as Name"
    - "(recXXX)" with preceding name
    """
    id_to_name: dict[str, str] = {}

    def _is_real_rec_id(s: str) -> bool:
        """真实 Airtable rec_id 必须含数字，排除普通英语单词如 reconstructive。"""
        return bool(re.search(r'\d', s))

    for fname, text in doc_contents.items():
        # 按句子分割，逐句扫描
        sentences = re.split(r'(?<=[.!?])\s+|\n\n', text)

        for sentence in sentences:
            # 找句子里所有真实 rec ID（必须含数字，排除纯字母的英语单词）
            ids_in_sentence = [
                m for m in re.findall(r'\b(rec[A-Za-z0-9]{8,})\b', sentence)
                if re.search(r'\d', m)  # 真实 rec_id 一定含数字
            ]
            if not ids_in_sentence:
                continue

            for rec_id in ids_in_sentence:
                if rec_id in id_to_name:
                    continue

                # Pattern 1: "for/of NAME (Registry ID: recXXX)" or "NAME (Registry ID: recXXX)"
                # Strict: capture only the word(s) immediately before the parenthesis
                m = re.search(
                    r'(?:for|of|as)\s+([A-Z][A-Za-z ,&]+?)\s*\((?:Registry ID[:\s]+)?' + re.escape(rec_id),
                    sentence
                )
                if m:
                    name = _clean_name(m.group(1))
                    if name and len(name.split()) <= 6:  # sanity: names shouldn't be too long
                        id_to_name[rec_id] = name
                        continue

                # Pattern 2: "is now NAME (recXXX)" or "corrected to NAME (recXXX)"
                m = re.search(
                    r'(?:is now|amended to|updated to|corrected to|titled)\s+([A-Z][A-Za-z ,&]+?)\s*\(' + re.escape(rec_id),
                    sentence
                )
                if m:
                    name = _clean_name(m.group(1))
                    if name and len(name.split()) <= 6:
                        id_to_name[rec_id] = name
                        continue

                # Pattern 3: "recXXX ... NAME" (ID before name)
                m = re.search(
                    re.escape(rec_id) + r'[^.]*?(?:identified as|is)\s+([A-Z][A-Za-z ,&]+?)[\.\(,]',
                    sentence
                )
                if m:
                    name = _clean_name(m.group(1))
                    if name:
                        id_to_name[rec_id] = name
                        continue

                # Pattern 4: "(recXXX)" with full NAME before it in same sentence
                # e.g. "Business (Registry ID: recxK3MHQFbR9J5uO)"
                m = re.search(
                    r'\b([A-Z][A-Za-z ,&]+?)\s*\(?(?:Registry ID[:\s]+)?' + re.escape(rec_id) + r'\)?',
                    sentence
                )
                if m:
                    name = _clean_name(m.group(1))
                    if name and len(name) < 80:
                        id_to_name[rec_id] = name
                        continue

                # Pattern 5: ID in sentence, name appears as "verified identity is NAME" or
                # "corresponds to NAME" or "is NAME" or "as NAME" in same or next sentence
                # For person docs: "associated with registration ID recXXX ... is FirstName LastName"
                m = re.search(
                    re.escape(rec_id) + r'[^.]{0,200}?(?:is|as|corresponds to|verified identity[^.]*?is)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
                    sentence,
                    re.DOTALL
                )
                if m:
                    name = m.group(1).strip()
                    if name:
                        id_to_name[rec_id] = name

        # Second pass: "recXXX ... designated as the NAME event/program/meeting"
        # Overrides dirty (long) first-pass entries
        for m in re.finditer(
            r'\b(rec[A-Za-z0-9]{8,})\b[^.]{0,150}?designated as[^.]{0,50}?([A-Z][A-Za-z\'\s]+?)(?:\s+event|\s+program|\s+meeting|\s+initiative|\s+session|[,.])',
            text,
            re.DOTALL
        ):
            rec_id, name = m.group(1), m.group(2).strip()
            if not _is_real_rec_id(rec_id):
                continue
            name = re.sub(r'^the\s+', '', name, flags=re.IGNORECASE).strip()
            existing = id_to_name.get(rec_id, "")
            if name and len(name.split()) <= 6 and len(name) < len(existing):
                id_to_name[rec_id] = name

        # Also: "NAME event/meeting (recXXX)" — name before ID
        for m in re.finditer(
            r"([A-Z][A-Za-z'`\s]+?)\s+(?:event|meeting|session)\s*\((rec[A-Za-z0-9]{8,})\)",
            text
        ):
            name, rec_id = m.group(1).strip(), m.group(2)
            if not _is_real_rec_id(rec_id):
                continue
            existing = id_to_name.get(rec_id, "x" * 999)
            if name and len(name.split()) <= 6 and len(name) < len(existing):
                id_to_name[rec_id] = name

        # Second pass: for person docs where "ID recXXX ... NAME" spans multiple sentences
        for m in re.finditer(
            r'(?:registration ID|tracking number|registry code|identifier|Registry Ref:)\s+(rec[A-Za-z0-9]{8,})[^.]*?\.\s+[^.]*?(?:corresponds to|is|as|identity[^.]*?is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            text,
            re.DOTALL
        ):
            rec_id, name = m.group(1), m.group(2).strip()
            if not _is_real_rec_id(rec_id):
                continue
            if rec_id not in id_to_name and name:
                id_to_name[rec_id] = name

        # Also handle corrected names: "corrected ... full identity as NAME"
        for m in re.finditer(
            r'(rec[A-Za-z0-9]{8,})[^.]*?full identity as\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
            text, re.DOTALL
        ):
            rec_id, name = m.group(1), m.group(2).strip()
            if not _is_real_rec_id(rec_id):
                continue
            if name:
                id_to_name[rec_id] = name  # override with corrected name

        # ── 第三 pass：非 rec_id 格式（纯数字ID、字母数字编号）──────────────────
        # 覆盖 TR391、Case ID 43003、registry code 1、Competition ID: 1729 等格式

        # Pattern A: 提取分子/化合物的毒性分类状态
        # 格式1: "compound/specimen/structure TRNNN ... classified as carcinogenic"
        # 格式2: "structure designated TRNNN. This compound was classified as..."
        # 只在同一段落内搜索（不跨两个空行）
        paragraphs = re.split(r'\n\n+', text)
        for para in paragraphs:
            # 找段落里的分子编号
            mol_ids_in_para = re.findall(r'\b([A-Z]{2,4}\d+)\b', para)
            if not mol_ids_in_para:
                continue
            # 找段落里的毒性状态（取最后一个，因为最后的描述最可能是修正值）
            status_matches = re.findall(
                r'(non[-\s]carcinogenic|carcinogenic|positive carcinogenic|'
                r'positive(?:\s+carcinogenic)?|negative)',
                para, re.IGNORECASE
            )
            if not status_matches:
                continue
            # 用最后一个状态（处理 initially X, corrected to Y 的情况）
            final_status = status_matches[-1].lower().strip()
            is_carcinogenic = 'non' not in final_status and 'negative' not in final_status
            status_label = 'carcinogenic' if is_carcinogenic else 'non-carcinogenic'
            for mol_id in set(mol_ids_in_para):
                if mol_id not in id_to_name or id_to_name[mol_id] != status_label:
                    id_to_name[mol_id] = status_label

        # Pattern B: "registry code N", "Competition ID: N", "ID N", "race N", "Case ID N"
        # → {str(N) → label} 用于 task_330/408/344 类
        for m in re.finditer(
            r'(?:registry code|Competition ID[:\s]+|operating under[^,]*?code\s+|'
            r'race\s+|raceId\s+|Case ID\s+|Patient\s+(?:ID\s+)?|tracked under[^,]*?designation\s+)'
            r'(\d+)',
            text, re.IGNORECASE
        ):
            num_id = m.group(1)
            # 提取这个 ID 对应的名称（前面的名词短语）
            idx = m.start()
            # 往前找最近的大写词组
            preceding = text[max(0, idx-200):idx]
            name_m = re.search(
                r'([A-Z][A-Za-z\s]+?)\s*(?:was|is|has been|,)\s*(?:officially|formally|finally|now|corrected|confirmed)?\s*$',
                preceding
            )
            if name_m:
                name = _clean_name(name_m.group(1))
                if name and len(name.split()) <= 6:
                    id_to_name[num_id] = name

    return id_to_name


def _clean_name(raw: str) -> str:
    """清理提取的名字：去掉冗余前缀和后缀。"""
    name = raw.strip().rstrip('., ')
    # 去掉常见前缀
    for prefix in ['The ', 'A ', 'An ', 'This ']:
        if name.startswith(prefix):
            name = name[len(prefix):]
    # 去掉 "program", "track", "discipline" 等后缀（如果不是名字的一部分）
    name = re.sub(r'\s+(program|track|discipline|unit|course|vocational track)$', '', name, flags=re.IGNORECASE)
    return name.strip()


def _infer_id_field(inv: AssetInventory, id_to_name: dict[str, str]) -> str:
    """从 CSV 列名启发式推断 id_field（不调 LLM）。"""
    if not id_to_name:
        return ""
    # 优先：link_to_* 前缀列
    for path in inv.csv_files:
        try:
            df = pd.read_csv(path, nrows=5)
            for col in df.columns:
                if col.lower().startswith("link_to_"):
                    return col
        except Exception:
            pass
    # 次选：列名包含已提取的某个 rec_id 的值（抽样检查）
    sample_ids = set(list(id_to_name.keys())[:20])
    for path in inv.csv_files:
        try:
            df = pd.read_csv(path, nrows=100)
            for col in df.columns:
                vals = set(str(v) for v in df[col].dropna().unique())
                if vals & sample_ids:
                    return col
        except Exception:
            pass
    return ""


def _get_csv_headers(inv: AssetInventory) -> dict[str, list[str]]:
    """读取所有结构化文件的列名（CSV + DB + JSON），用于 LLM 推断 id_field。"""
    import sqlite3 as _sqlite3
    result = {}
    for path in inv.csv_files:
        try:
            df = pd.read_csv(path, nrows=0)
            result[path.name] = list(df.columns)
        except Exception:
            pass
    for path in inv.db_files:
        try:
            con = _sqlite3.connect(str(path))
            try:
                tables = con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
                for (tbl,) in tables:
                    cols = [r[1] for r in con.execute(f"PRAGMA table_info({tbl})").fetchall()]
                    result[f"{path.name}::{tbl}"] = cols
            finally:
                con.close()
        except Exception:
            pass
    for path in inv.json_files:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            records = raw.get("records", raw) if isinstance(raw, dict) else raw
            if isinstance(records, list) and records:
                result[path.name] = list(pd.json_normalize(records[:1]).columns[:15])
        except Exception:
            pass
    return result


def _build_file_schemas(inv: AssetInventory) -> str:
    """构建文件 schema 描述，使用预注入变量名访问。"""
    import sqlite3
    parts = []

    for path in inv.csv_files:
        parts.append(f"### CSV `{path.name}` — access: pd.read_csv(csv_paths['{path.name}'])")
        try:
            df = pd.read_csv(path, nrows=3)
            parts.append(f"  Columns: {', '.join(df.columns)}")
            parts.append(f"  Sample:\n{textwrap.indent(df.to_string(index=False), '    ')}")
        except Exception as e:
            parts.append(f"  (read error: {e})")

    for path in inv.db_files:
        parts.append(f"### SQLite DB `{path.name}` — access: sqlite3.connect(db_paths['{path.name}'])")
        try:
            conn = sqlite3.connect(str(path.resolve()))
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cur.fetchall()]
            for tbl in tables:
                cur.execute(f"PRAGMA table_info({tbl})")
                cols = [r[1] for r in cur.fetchall()]
                cur.execute(f"SELECT * FROM {tbl} LIMIT 2")
                rows = cur.fetchall()
                parts.append(f"  Table `{tbl}`: {', '.join(cols)}")
                if rows:
                    parts.append(f"    Sample: {rows}")
            conn.close()
        except Exception as e:
            parts.append(f"  (read error: {e})")

    for path in inv.json_files:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            records = raw.get("records", raw) if isinstance(raw, dict) else raw
            table_name = raw.get("table", path.stem) if isinstance(raw, dict) else path.stem
            safe_var = re.sub(r"[^a-zA-Z0-9_]", "_", table_name) + "_df"
            parts.append(f"### JSON `{path.name}` — pre-loaded as `{safe_var}` (DataFrame, use directly)")
            if isinstance(records, list) and records:
                sample = pd.json_normalize(records[:2])
                parts.append(f"  Columns: {', '.join(sample.columns[:10])}")
                parts.append(f"  Sample: {records[:1]}")
        except Exception as e:
            parts.append(f"### JSON `{path.name}` — (read error: {e})")

    return "\n".join(parts)


# ── 代码执行 ─────────────────────────────────────────────────────────────────

def _execute_code(
    code: str,
    context_dir: Path,
    inv: AssetInventory,
    entities_json: dict,
    timeout: int = 60,
) -> dict[str, Any]:
    """在 subprocess 中执行代码，预注入 csv_paths / db_paths / id_to_name 变量。"""
    csv_paths_repr = json.dumps(
        {p.name: str(p.resolve()) for p in inv.csv_files}, ensure_ascii=False
    )
    db_paths_repr = json.dumps(
        {p.name: str(p.resolve()) for p in inv.db_files}, ensure_ascii=False
    )
    json_paths_repr = json.dumps(
        {p.name: str(p.resolve()) for p in inv.json_files}, ensure_ascii=False
    )

    # Build id_to_name from entities_json
    id_to_name: dict[str, str] = {}
    if entities_json.get("id_to_name"):
        id_to_name = entities_json["id_to_name"]
    elif entities_json.get("entities"):
        for e in entities_json.get("entities", []):
            if e.get("id") and e.get("name"):
                id_to_name[e["id"]] = e["name"]

    id_field = entities_json.get("id_field") or ""
    id_to_name_repr = json.dumps(id_to_name, ensure_ascii=False)

    # JSON 文件预加载为 DataFrame，避免 LLM 手动解析 Airtable {"table":..., "records":[...]} 格式
    # 超大 JSON 用 json_paths 读取，小 JSON（≤ _MAX_JSON_RECORDS_PRELOAD 条）直接内联
    json_df_preloads = []
    for p in inv.json_files:
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            records = raw.get("records", raw) if isinstance(raw, dict) else raw
            table_name = raw.get("table", p.stem) if isinstance(raw, dict) else p.stem
            safe_var = re.sub(r"[^a-zA-Z0-9_]", "_", table_name) + "_df"
            if isinstance(records, list) and len(records) <= _MAX_JSON_RECORDS_PRELOAD:
                # 小文件：直接内联到 prelude
                records_repr = json.dumps(records, ensure_ascii=False)
                json_df_preloads.append(
                    f"# JSON '{p.name}' pre-loaded as DataFrame\n"
                    f"{safe_var} = pd.DataFrame({records_repr})"
                )
            else:
                # 大文件：运行时读取，保持一致的变量名
                abs_path = str(p.resolve())
                json_df_preloads.append(
                    f"# JSON '{p.name}' loaded at runtime (large file)\n"
                    f"with open({abs_path!r}) as _f: _raw_{safe_var} = json.load(_f)\n"
                    f"_recs_{safe_var} = _raw_{safe_var}.get('records', _raw_{safe_var}) "
                    f"if isinstance(_raw_{safe_var}, dict) else _raw_{safe_var}\n"
                    f"{safe_var} = pd.DataFrame(_recs_{safe_var})"
                )
        except Exception:
            pass
    json_df_block = "\n".join(json_df_preloads)

    prelude = textwrap.dedent(f"""\
import json, re, sys, os, sqlite3
import pandas as pd

# Pre-loaded context — use these variables directly
csv_paths  = {csv_paths_repr}
db_paths   = {db_paths_repr}
json_paths = {json_paths_repr}

# JSON files pre-loaded as DataFrames (use these directly — no need to json.load)
{json_df_block}

# Pre-extracted ID→name mapping from documents (ready to use)
# id_field hint: '{id_field}' (the CSV column that links to these IDs)
id_to_name = {id_to_name_repr}

os.chdir({repr(str(context_dir.resolve()))})

""")

    full_code = prelude + "\n# User code\n" + code

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(full_code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(context_dir),
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if "__RESULT__" in stdout:
            json_str = stdout.split("__RESULT__", 1)[1]
            records = json.loads(json_str)
            if records:
                df = pd.DataFrame(records)
                return {"success": True, "df": df, "rows": records, "error": None}
            else:
                return {"success": False, "df": pd.DataFrame(), "rows": [], "error": "empty result set"}

        error_msg = stderr or stdout or "No __RESULT__ marker in output"
        return {"success": False, "df": None, "rows": [], "error": error_msg}

    except subprocess.TimeoutExpired:
        return {"success": False, "df": None, "rows": [], "error": f"Execution timed out ({timeout}s)"}
    except Exception as e:
        return {"success": False, "df": None, "rows": [], "error": str(e)}
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ── LLM 调用辅助 ─────────────────────────────────────────────────────────────

def _llm(model: OpenAIModelAdapter, system: str, user: str) -> str:
    return model.chat([
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ])


def _parse_json_response(text: str) -> dict | None:
    """从 LLM 响应中提取 JSON（5层容错，复用 sql_solver._parse_llm_json）。"""
    from data_agent_baseline.solvers.sql_solver import _parse_llm_json
    return _parse_llm_json(text)


def _extract_code(text: str) -> str:
    """从 LLM 响应中提取 Python 代码。"""
    text = re.sub(r"```(?:python)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    return text.strip()


# ── 主入口 ────────────────────────────────────────────────────────────────────

def solve_structured(
    task: PublicTask,
    inv: AssetInventory,
    knowledge: KnowledgeContext,
    model: OpenAIModelAdapter,
) -> AnswerTable:
    """
    状态机驱动的 structured solver。

    返回 AnswerTable（可能为空，但不会抛出异常）。
    """
    context_dir = task.assets.context_dir
    question = task.question
    knowledge_text = knowledge.to_prompt_section()

    # ── 读取所有输入 ──────────────────────────────────────────────────────────
    doc_contents = _read_doc_files(inv)
    csv_headers = _get_csv_headers(inv)
    file_schemas = _build_file_schemas(inv)

    # ── 状态变量 ──────────────────────────────────────────────────────────────
    state = SolverState.EXTRACT_ENTITIES
    entities_json: dict = {}
    code: str = ""
    last_error: str | None = None
    retry_count = 0
    result_df: pd.DataFrame | None = None

    # ── 状态机循环 ────────────────────────────────────────────────────────────
    while state not in (SolverState.DONE, SolverState.FAILED):

        # ── 阶段 1：从 doc 提取实体（纯 Python，不调 LLM）────────────────────
        if state == SolverState.EXTRACT_ENTITIES:
            if not doc_contents:
                entities_json = {"id_field": None, "id_to_name": {}}
                state = SolverState.BUILD_QUERY
                continue

            # Python regex 全文提取（快速、覆盖完整文档）
            python_id_map = _python_extract_id_mapping(doc_contents)

            # 如果 Python 正则提取到空映射，降级到 LLM 提取（支持数字 ID 等非 rec 格式）
            # 小文档（< _SMALL_DOC_THRESHOLD）：全文一次提取，最多重试 2 次
            # 大文档（≥ _SMALL_DOC_THRESHOLD）：分块并行提取，合并所有块结果
            if not python_id_map:
                total_doc_size = sum(len(v) for v in doc_contents.values())
                print(
                    f"[structured] No rec_id via regex, trying LLM extraction "
                    f"(doc_size={total_doc_size})...",
                    file=sys.stderr,
                )

                merged_id_to_name: dict[str, str] = {}
                merged_id_field: str | None = None

                if total_doc_size < _SMALL_DOC_THRESHOLD:
                    # 小文档：全文一次提取，最多重试 2 次
                    for _extract_attempt in range(2):
                        extract_prompt = _build_extract_prompt(
                            question, doc_contents, csv_headers, knowledge_text
                        )
                        llm_response = _llm(model, _SYSTEM_EXTRACT, extract_prompt)
                        result = _parse_json_response(llm_response)
                        if result and result.get("id_to_name"):
                            merged_id_to_name = result["id_to_name"]
                            merged_id_field = result.get("id_field")
                            break
                        print(
                            f"[structured] LLM extraction attempt {_extract_attempt + 1} empty, retrying...",
                            file=sys.stderr,
                        )
                else:
                    # 大文档：分块并行提取，合并所有块结果
                    for fname, content in doc_contents.items():
                        chunk_map = _llm_extract_large_doc(
                            fname, content, question, csv_headers, knowledge_text, model
                        )
                        merged_id_to_name.update(chunk_map)
                    # id_field 从小文档路径补一次（拿整体 schema 推断）
                    if merged_id_to_name:
                        extract_prompt = _build_extract_prompt(
                            question, {k: v[:500] for k, v in doc_contents.items()},
                            csv_headers, knowledge_text
                        )
                        llm_response = _llm(model, _SYSTEM_EXTRACT, extract_prompt)
                        result = _parse_json_response(llm_response)
                        if result:
                            merged_id_field = result.get("id_field")

                llm_entities = (
                    {"id_to_name": merged_id_to_name, "id_field": merged_id_field}
                    if merged_id_to_name else None
                )

                if llm_entities and llm_entities.get("id_to_name"):
                    # LLM 提取成功，继续走 BUILD_QUERY
                    print(
                        f"[structured] LLM extraction OK: {len(llm_entities['id_to_name'])} entries, "
                        f"id_field={llm_entities.get('id_field')!r}",
                        file=sys.stderr,
                    )
                    entities_json = llm_entities
                    state = SolverState.BUILD_QUERY
                    continue
                else:
                    # LLM 也提取不到映射，doc 里确实没有 ID 信息 → 降级到 sql/pandas solver
                    print("[structured] LLM extraction empty after retries, falling back to sql/pandas solver", file=sys.stderr)
                    if inv.has_db:
                        from data_agent_baseline.solvers.sql_solver import solve_sql
                        from data_agent_baseline.solvers.classifier import estimate_difficulty
                        fallback_budget = estimate_difficulty(inv, question)
                        df = solve_sql(question, inv, knowledge, model, budget=fallback_budget)
                    else:
                        from data_agent_baseline.solvers.pandas_solver import solve_pandas
                        df = solve_pandas(question, inv, knowledge, model)
                    if df is not None and not df.empty:
                        return AnswerTable.from_dataframe(df)
                    return AnswerTable(columns=[], rows=[])

            # 启发式推断 id_field：找 CSV 中 "link_to_" 前缀或包含 rec 值的列
            id_field_hint = _infer_id_field(inv, python_id_map)

            entities_json = {
                "id_field": id_field_hint,
                "id_to_name": python_id_map,
            }

            print(
                f"[structured] EXTRACT done (no LLM). "
                f"extracted={len(python_id_map)}, id_field={id_field_hint!r}",
                file=sys.stderr,
            )
            state = SolverState.BUILD_QUERY

        # ── 阶段 2：生成查询代码 ──────────────────────────────────────────────
        elif state == SolverState.BUILD_QUERY:
            query_prompt = _build_query_prompt(
                question, entities_json, file_schemas, knowledge_text,
                prev_error=last_error,
            )
            response = _llm(model, _SYSTEM_QUERY, query_prompt)
            code = _extract_code(response)
            print(f"[structured] attempt={retry_count+1}\n--- CODE ---\n{code}\n--- END ---", file=sys.stderr)
            state = SolverState.EXECUTE

        # ── 阶段 3：执行代码 ──────────────────────────────────────────────────
        elif state == SolverState.EXECUTE:
            exec_result = _execute_code(code, context_dir, inv, entities_json, timeout=60)

            if exec_result["success"] and exec_result["df"] is not None:
                result_df = exec_result["df"]
                state = SolverState.DONE
            else:
                last_error = exec_result["error"] or "unknown error"
                print(f"[structured] execute failed (attempt {retry_count+1}): {last_error[:200]}", file=sys.stderr)
                retry_count += 1
                if retry_count >= MAX_RETRIES:
                    state = SolverState.FAILED
                else:
                    # 带错误信息重新生成代码
                    state = SolverState.BUILD_QUERY

    # ── 后处理并返回 ──────────────────────────────────────────────────────────
    if result_df is not None and not result_df.empty:
        return AnswerTable.from_dataframe(result_df)

    # 空结果也返回 AnswerTable（外层 base.py 会标记为 failed）
    return AnswerTable(columns=[], rows=[])
