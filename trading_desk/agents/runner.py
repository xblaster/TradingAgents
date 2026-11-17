"""One structured agent call: build the request, call the backend, validate with Pydantic."""

from __future__ import annotations

import logging
import time
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from ..llm import LLMBackend, LLMError, LLMRequest
from .schemas import json_schema

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class StepRecord(BaseModel):
    """Audit trail of one agent call, saved with every run."""

    agent: str
    model: str
    ok: bool
    output: dict[str, Any] | None = None
    error: str | None = None
    cost_usd: float = 0.0
    duration_ms: int = 0
    cached: bool = False


class AgentResult(Generic[T]):
    def __init__(self, record: StepRecord, value: T | None):
        self.record = record
        self.value = value

    @property
    def ok(self) -> bool:
        return self.value is not None


async def run_agent(
    backend: LLMBackend,
    *,
    name: str,
    model: str,
    system: str,
    prompt: str,
    output: type[T],
    tools: tuple[str, ...] = (),
) -> AgentResult[T]:
    """Never raises for model or validation errors: failures come back as `ok=False`."""
    schema = json_schema(output)
    request = LLMRequest(system=system, prompt=prompt, model=model, schema=schema, tools=tools, label=name)
    started = time.monotonic()
    cost = 0.0
    error = ""
    for attempt in range(2):  # one repair attempt if the payload fails validation
        try:
            response = await backend.complete(request)
        except LLMError as exc:
            error = str(exc)
            break
        cost += response.cost_usd
        try:
            value = output.model_validate(response.data)
        except ValidationError as exc:
            error = f"invalid output: {exc.errors(include_url=False)[:3]}"
            log.warning("%s returned invalid output (attempt %d): %s", name, attempt + 1, error)
            request = LLMRequest(
                system=system,
                prompt=f"{prompt}\n\nYour previous answer was rejected by validation: {error}\nAnswer again, strictly following the schema.",
                model=model, schema=schema, tools=tools, label=name,
            )
            continue
        record = StepRecord(
            agent=name, model=response.model, ok=True, output=value.model_dump(mode="json"),
            cost_usd=cost, duration_ms=response.duration_ms, cached=response.cached,
        )
        return AgentResult(record, value)

    log.error("%s failed: %s", name, error)
    record = StepRecord(
        agent=name, model=model, ok=False, error=error, cost_usd=cost,
        duration_ms=int((time.monotonic() - started) * 1000),
    )
    return AgentResult(record, None)
