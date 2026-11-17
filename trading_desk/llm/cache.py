"""On-disk response cache: reruns and backtests become free and reproducible."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .base import LLMBackend, LLMRequest, LLMResponse


class CachedBackend:
    def __init__(self, inner: LLMBackend, directory: str | Path):
        self.inner = inner
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    async def complete(self, request: LLMRequest) -> LLMResponse:
        path = self._path(request.cache_key())
        if path.exists():
            try:
                raw = json.loads(path.read_text())
                return LLMResponse(**{**raw, "cached": True})
            except (json.JSONDecodeError, TypeError):
                path.unlink(missing_ok=True)

        response = await self.inner.complete(request)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(response.to_dict()))
        os.replace(tmp, path)  # atomic: concurrent readers never see a half-written file
        return response
