"""
knowledge.md 结构化解析器。

knowledge.md 是每个任务都有的数据字典，包含：
- 字段定义和正常值范围（如 LDH > 500 为异常）
- SQL 示例（作为 few-shot 提供给 LLM）
- 实体别名
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class KnowledgeContext:
    field_definitions: dict[str, str] = field(default_factory=dict)
    normal_ranges: dict[str, str] = field(default_factory=dict)
    sql_examples: list[str] = field(default_factory=list)
    aliases: dict[str, list[str]] = field(default_factory=dict)
    raw_text: str = ""

    def to_prompt_section(self) -> str:
        parts: list[str] = []

        if self.field_definitions:
            parts.append("## Field Definitions")
            for field_name, desc in self.field_definitions.items():
                parts.append(f"- {field_name}: {desc}")

        if self.normal_ranges:
            parts.append("\n## Normal/Abnormal Value Thresholds")
            for field_name, condition in self.normal_ranges.items():
                parts.append(f"- {field_name}: {condition}")

        if self.sql_examples:
            parts.append("\n## Reference SQL Examples (follow this style)")
            for i, sql in enumerate(self.sql_examples, 1):
                parts.append(f"-- Example {i}\n{sql}")

        if self.aliases:
            parts.append("\n## Entity Aliases")
            for canonical, alias_list in self.aliases.items():
                parts.append(f"- '{canonical}' may also appear as: {', '.join(alias_list)}")

        return "\n".join(parts)


def enrich_knowledge_with_llm(
    ctx: KnowledgeContext,
    question: str,
    model: object,
) -> KnowledgeContext:
    """LLM 补充理解 knowledge.md，与正则解析结果取并集。

    每道有 knowledge.md 的题都调用一次：LLM 理解自然语言描述的语义
    （如阈值、枚举编码），补充正则无法捕获的字段定义和阈值。
    正则已提取的内容优先，LLM 只新增正则没有的字段。
    """
    if not ctx.raw_text:
        return ctx

    # 记录正则已提取到的 keys，LLM 结果只补充正则没有的部分
    regex_field_keys = set(ctx.field_definitions.keys())
    regex_range_keys = set(ctx.normal_ranges.keys())

    _LLM_KNOWLEDGE_SYSTEM = (
        "You are a data semantics expert. Extract field definitions and "
        "abnormal/normal value thresholds from the knowledge document. "
        "Output ONLY valid JSON, no explanation."
    )
    _LLM_KNOWLEDGE_USER = """\
Question: {question}

knowledge.md content:
{knowledge}

Extract in JSON:
{{
  "field_definitions": {{"field_name": "meaning and value encoding, e.g. SEX: M=male F=female"}},
  "normal_ranges": {{"field_name": "threshold condition, e.g. WBC: normal 3.5-9.0; abnormal <3.5 or >9.0"}}
}}

Rules:
- Only include fields relevant to answering the question
- For gender/status/category fields, include the value encoding
- For numeric fields, include the normal/abnormal range if mentioned
- If not mentioned in the document, omit the field entirely
"""

    try:
        user_content = _LLM_KNOWLEDGE_USER.format(
            question=question,
            knowledge=ctx.raw_text[:4000],
        )
        response = model.chat([
            {"role": "system", "content": _LLM_KNOWLEDGE_SYSTEM},
            {"role": "user", "content": user_content},
        ])
        # 5层容错 JSON 解析
        from data_agent_baseline.solvers.sql_solver import _parse_llm_json
        parsed = _parse_llm_json(response)
        if parsed:
            field_defs = parsed.get("field_definitions") or {}
            normal_ranges = parsed.get("normal_ranges") or {}
            if isinstance(field_defs, dict):
                # 只补充正则没有的字段（正则结果优先，不覆盖）
                for k, v in field_defs.items():
                    if k not in regex_field_keys:
                        ctx.field_definitions[k] = v
            if isinstance(normal_ranges, dict):
                # 只补充正则没有的阈值（正则结果优先，不覆盖）
                for k, v in normal_ranges.items():
                    if k not in regex_range_keys:
                        ctx.normal_ranges[k] = v
    except Exception:
        pass  # LLM 补充失败时静默，不影响原有流程

    return ctx


def parse_knowledge(context_dir: Path) -> KnowledgeContext:
    """解析 context/ 目录下的 knowledge.md 文件。"""
    knowledge_path = context_dir / "knowledge.md"
    if not knowledge_path.exists():
        return KnowledgeContext()

    raw = knowledge_path.read_text(encoding="utf-8")
    ctx = KnowledgeContext(raw_text=raw)

    # 提取 SQL 代码块
    sql_blocks = re.findall(r"```sql\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    ctx.sql_examples = [s.strip() for s in sql_blocks if s.strip()]

    # 提取字段定义：匹配 "- **FieldName (type):** description" 或 "- **FieldName:** description"
    # 过滤掉 knowledge.md 模板里的节标题关键词（不是真实字段名）
    _SECTION_KEYWORDS = {
        "metric", "metrics", "explanation", "sql", "sql logic", "formula",
        "description", "natural language", "note", "notes", "example",
        "use case", "kpi", "key performance indicator",
    }
    field_pattern = re.findall(
        r"\*\*([A-Za-z_\s\(\)]+?)\*\*[:\s]+(.+?)(?=\n|$)",
        raw,
    )
    for name_raw, desc in field_pattern:
        name = name_raw.strip().rstrip(":")
        desc_clean = desc.strip()
        if len(name) > 0 and len(desc_clean) > 0 and name.lower() not in _SECTION_KEYWORDS:
            ctx.field_definitions[name] = desc_clean

    # 提取正常值范围：匹配常见模式
    # "values above 500 considered beyond the normal range"
    # "normal range: X to Y"
    # "Metric: FieldName > N"
    range_patterns = [
        # "FIELDNAME level, with values above X considered beyond the normal range"
        # Only match short all-caps field names (2-6 chars) to avoid matching long words
        (r"\b([A-Z]{2,6})\s+(?:level|value)[,\s]+with values (?:above|over) ([\d.]+) considered", ">{1}"),
        # "**FIELDNAME (integer):** ... with values above X considered"
        (r"\*\*([A-Z]{2,6})[^*]*\*\*[^.]*values (?:above|over) ([\d.]+)", ">{1}"),
        # "BETWEEN X AND Y" in SQL context
        (r"\b([A-Z]{2,6})\s+BETWEEN\s+([\d.]+)\s+AND\s+([\d.]+)", "between"),
        # "**Metric:** SomeName > N" or "**Metric:** SomeName >= N" (bullet style)
        (r"\*\*Metric[:\*]+\s*([A-Za-z][A-Za-z0-9 _()/(%-]+?)\s*(>=?)\s*([\d.]+%?)", "metric_op"),
        # plain "**SomeName** > N" or "**SomeName:** > N" in bullet lines (non-SQL context)
        (r"^\s*[-*]\s+\*\*([A-Za-z][A-Za-z0-9 _()-]+?)\*\*[:\s]+(>=?)\s*([\d.]+)", "metric_op"),
        # "normal range: X to Y" or "normal range: X–Y"
        (r"(?:normal range|normalrange)[:\s]+([\d.]+)\s*(?:to|–|-)\s*([\d.]+)", "range_text"),
    ]
    for pattern, kind in range_patterns:
        for match in re.finditer(pattern, raw, re.IGNORECASE | re.MULTILINE):
            if kind == "between":
                field_name = match.group(1)
                ctx.normal_ranges[field_name] = f"normal: BETWEEN {match.group(2)} AND {match.group(3)}"
            elif kind == "metric_op":
                field_name = match.group(1).strip()
                op = match.group(2)
                val = match.group(3)
                ctx.normal_ranges[field_name] = f"threshold: {op} {val}"
            elif kind == "range_text":
                # range_text: no named field from match; attach to the nearest preceding field name
                # store as generic "range" keyed by match position placeholder
                lo, hi = match.group(1), match.group(2)
                # try to find nearest bold field name before this match
                before = raw[:match.start()]
                field_match = re.findall(r"\*\*([A-Za-z][A-Za-z0-9 _()-]+?)\*\*", before)
                key = field_match[-1].strip() if field_match else "_range"
                ctx.normal_ranges[key] = f"normal: {lo} to {hi}"
            else:
                field_name = match.group(1)
                ctx.normal_ranges[field_name] = f"abnormal: > {match.group(2)}"

    # 提取别名：匹配 "X may also be referred to as Y" 或 "X is also known as Y"
    alias_patterns = [
        r"'([^']+)'\s+(?:may also be|is also)\s+(?:referred to as|known as)\s+'([^']+)'",
        r'"([^"]+)"\s+(?:may also be|is also)\s+(?:referred to as|known as)\s+"([^"]+)"',
    ]
    for pattern in alias_patterns:
        for match in re.finditer(pattern, raw, re.IGNORECASE):
            canonical = match.group(1).strip()
            alias = match.group(2).strip()
            ctx.aliases.setdefault(canonical, []).append(alias)

    return ctx
