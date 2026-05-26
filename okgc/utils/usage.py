from dataclasses import dataclass
from typing import Any


@dataclass
class UsageInfo:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def is_zero(self) -> bool:
        return self.prompt_tokens == self.completion_tokens == 0

    def __add__(self, other: "UsageInfo") -> "UsageInfo":
        return UsageInfo(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
        )


def parse_usage_info(info: dict[str, Any]) -> UsageInfo:
    (metrics,) = list(info.values())
    return UsageInfo(metrics["prompt_tokens"], metrics["completion_tokens"])
