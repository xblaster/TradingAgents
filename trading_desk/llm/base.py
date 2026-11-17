"""Backend-agnostic LLM request/response types."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


class LLMError(RuntimeError):
    """A model call failed. `retryable` tells the caller whether trying again makes sense."""

    def __init__(self, message: str, *, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


class LLMTimeout(LLMError):
    pass


@dataclass(frozen=True)
class LLMRequest:
    system: str
    prompt: str
    model: str
    schema: dict[str, Any] | None = None
    tools: tuple[str, ...] = ()
    label: str = ""  # agent name, for logs; not part of the cache key

    def cache_key(self) -> str:
        material = json.dumps(
            {
                "system": self.system,
                "prompt": self.prompt,
                "model": self.model,
                "schema": self.schema,
                "tools": list(self.tools),
            },
            sort_keys=True,
        )
        return hashlib.sha256(material.encode()).hexdigest()


@dataclass
class LLMResponse:
    text: str
    data: dict[str, Any] | None
    model: str
    cost_usd: float = 0.0
    duration_ms: int = 0
    cached: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LLMBackend(Protocol):
    async def complete(self, request: LLMRequest) -> LLMResponse: ...
