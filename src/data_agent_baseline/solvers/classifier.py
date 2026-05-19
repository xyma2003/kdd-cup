"""
任务分类器：纯规则，零 LLM 调用。

根据 context/ 目录下的文件类型，将任务分为 4 种：
- TYPE_SQL    : 有 SQLite/DB 文件，走 SQL 生成路径
- TYPE_PANDAS : 纯 CSV/JSON，走 pandas 代码生成路径
- TYPE_DOC    : 纯文档（.md/.txt in doc/），走 NLP 信息提取路径
- TYPE_HYBRID : 文档 + 结构化数据，先提取文档再 join
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# 问题复杂度关键词（来自 lyp difficulty.py）
_COMPLEXITY_KEYWORDS = [
    "for each", "ratio", "percentage", "rank", "compare",
    "difference between", "average of", "per year", "over time",
    "trend", "correlation", "distribution", "breakdown",
]


class TaskType(str, Enum):
    SQL = "TYPE_SQL"
    PANDAS = "TYPE_PANDAS"
    DOC = "TYPE_DOC"
    HYBRID = "TYPE_HYBRID"


@dataclass
class AssetInventory:
    db_files: list[Path] = field(default_factory=list)
    csv_files: list[Path] = field(default_factory=list)
    json_files: list[Path] = field(default_factory=list)
    doc_files: list[Path] = field(default_factory=list)
    knowledge_path: Path | None = None

    @property
    def has_db(self) -> bool:
        return len(self.db_files) > 0

    @property
    def has_csv(self) -> bool:
        return len(self.csv_files) > 0

    @property
    def has_json(self) -> bool:
        return len(self.json_files) > 0

    @property
    def has_doc(self) -> bool:
        return len(self.doc_files) > 0

    @property
    def has_structured(self) -> bool:
        return self.has_db or self.has_csv or self.has_json


def scan_assets(context_dir: Path) -> AssetInventory:
    """扫描 context/ 目录，返回文件资产清单。"""
    inv = AssetInventory()

    if not context_dir.exists():
        return inv

    for item in context_dir.rglob("*"):
        if not item.is_file():
            continue

        rel = item.relative_to(context_dir)
        parts = rel.parts

        # knowledge.md 单独处理
        if item.name == "knowledge.md" and len(parts) == 1:
            inv.knowledge_path = item
            continue

        suffix = item.suffix.lower()
        # 判断是否在 doc/ 子目录下
        in_doc_dir = len(parts) > 1 and parts[0] == "doc"

        if in_doc_dir and suffix in (".md", ".txt", ".rst"):
            inv.doc_files.append(item)
        elif suffix in (".db", ".sqlite", ".sqlite3"):
            inv.db_files.append(item)
        elif suffix == ".csv":
            inv.csv_files.append(item)
        elif suffix == ".json":
            inv.json_files.append(item)
        elif suffix in (".md", ".txt") and not in_doc_dir:
            # 根目录下的 .md 文件（非 knowledge.md）也视为 doc
            inv.doc_files.append(item)

    return inv


@dataclass
class DifficultyBudget:
    """任务难度驱动的执行预算（零 LLM 开销，纯规则估计）。"""
    level: str               # "easy_like" | "medium_like" | "hard_like" | "extreme_like"
    skip_intent_review: bool # easy 任务跳过意图校验，节省 1 次 LLM
    max_repair_rounds: int   # LLM 修复最多尝试几次（0=不修复，1=修复一次，2=修复两次）


def estimate_difficulty(inv: AssetInventory, question: str) -> DifficultyBudget:
    """基于可观测信号（文件大小、schema复杂度、问题关键词）估计任务难度。

    纯规则，零 LLM 调用。信号来源参考 lyp 分支 orchestrator/difficulty.py。
    """
    score = 0

    # 1. 文档大小（文档越大，实体映射越复杂）
    total_doc_size = sum(p.stat().st_size for p in inv.doc_files if p.exists())
    if total_doc_size > 60_000:
        score += 4
    elif total_doc_size > 20_000:
        score += 3
    elif total_doc_size > 5_000:
        score += 2
    elif total_doc_size > 0:
        score += 1  # 有文档就至少 +1

    # 2. 多资产类型（需要跨源 JOIN 时更难）
    structured_count = len(inv.db_files) + len(inv.csv_files) + len(inv.json_files)
    if structured_count >= 3:
        score += 2
    elif structured_count >= 2:
        score += 1
    if len(inv.doc_files) >= 2:
        score += 1

    # 3. Schema 复杂度（DB + CSV 的总列数）
    total_cols = 0
    for db_path in inv.db_files:
        try:
            con = sqlite3.connect(str(db_path))
            try:
                tables = con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
                for (tbl,) in tables:
                    cols = con.execute(f"PRAGMA table_info({tbl})").fetchall()
                    total_cols += len(cols)
            finally:
                con.close()
        except Exception:
            pass
    for csv_path in inv.csv_files:
        try:
            with open(csv_path, encoding="utf-8", errors="ignore") as f:
                header = f.readline()
            total_cols += len(header.split(","))
        except Exception:
            pass
    if total_cols > 40:
        score += 2
    elif total_cols > 15:
        score += 1

    # 4. 问题关键词复杂度
    q_lower = question.lower()
    score += sum(1 for kw in _COMPLEXITY_KEYWORDS if kw in q_lower)

    if score <= 2:
        return DifficultyBudget(level="easy_like", skip_intent_review=True, max_repair_rounds=1)
    if score <= 5:
        return DifficultyBudget(level="medium_like", skip_intent_review=False, max_repair_rounds=1)
    if score <= 9:
        return DifficultyBudget(level="hard_like", skip_intent_review=False, max_repair_rounds=2)
    return DifficultyBudget(level="extreme_like", skip_intent_review=False, max_repair_rounds=2)


def classify_task(inv: AssetInventory) -> TaskType:
    """根据资产清单判断任务类型。"""
    if inv.has_doc and not inv.has_structured:
        return TaskType.DOC

    if inv.has_doc and inv.has_structured:
        return TaskType.HYBRID

    if inv.has_db:
        return TaskType.SQL

    # CSV 或 JSON
    return TaskType.PANDAS
