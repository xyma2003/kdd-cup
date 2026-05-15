from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True, slots=True)
class TaskRecord:
    task_id: str
    difficulty: str
    question: str


@dataclass(frozen=True, slots=True)
class TaskAssets:
    task_dir: Path
    context_dir: Path


@dataclass(frozen=True, slots=True)
class PublicTask:
    record: TaskRecord
    assets: TaskAssets

    @property
    def task_id(self) -> str:
        return self.record.task_id

    @property
    def difficulty(self) -> str:
        return self.record.difficulty

    @property
    def question(self) -> str:
        return self.record.question

    @property
    def task_dir(self) -> Path:
        return self.assets.task_dir

    @property
    def context_dir(self) -> Path:
        return self.assets.context_dir


@dataclass(frozen=True, slots=True)
class AnswerTable:
    columns: list[str]
    rows: list[list[Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "rows": [list(row) for row in self.rows],
        }

    @classmethod
    def from_dataframe(cls, df: "pd.DataFrame") -> "AnswerTable":
        """将 DataFrame 直接转为 AnswerTable，不做任何裁剪或后处理。"""
        import pandas as pd  # local import to keep module lightweight
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return cls(columns=[], rows=[])
        columns = [str(c) for c in df.columns]
        rows = [[v if v is not None else "" for v in row] for row in df.values.tolist()]
        return cls(columns=columns, rows=rows)
