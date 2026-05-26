from dataclasses import dataclass, field
from typing import Any


@dataclass
class DataEntry:
    texts: list[str]
    labels: dict[str, Any] = field(default_factory=dict)
    extra_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuestionEntry:
    id: str
    question: str
    answer: str | int | float | bool | dict[str, str | int | float | bool]
    categories: list[str] = field(default_factory=list)
