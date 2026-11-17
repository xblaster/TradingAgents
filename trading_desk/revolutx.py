"""Minimal Revolut X REST client (https://revx.revolut.com/api/1.0).

Authentication, per Revolut X's spec: each private request carries
  X-Revx-API-Key   the 64-char key Revolut X issues for your registered public key,
  X-Revx-Timestamp Unix time in milliseconds,
  X-Revx-Signature base64(Ed25519_sign(timestamp + METHOD + path + query + body)),
where path starts at /api, query has no leading '?', body is the exact minified JSON sent,
and the parts are concatenated without separators.

The private key never leaves your machine: generate it locally (openssl genpkey -algorithm
ed25519), register only the public key in Revolut X (Profile > API keys).

This module is read-only for now (market data, balances). Order placement comes with the
broker layer, behind explicit live-trading switches.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

BASE_URL = "https://revx.revolut.com"
API_PREFIX = "/api/1.0"
DEFAULT_KEY_PATH = Path("~/.config/revolut-x/private.pem").expanduser()


class RevolutXError(RuntimeError):
    def __init__(self, status: int, message: str, error_id: str | None = None):
        super().__init__(f"Revolut X {status}: {message}")
        self.status, self.message, self.error_id = status, message, error_id


def signing_message(timestamp_ms: int, method: str, path: str, query: str = "", body: str = "") -> bytes:
    return f"{timestamp_ms}{method.upper()}{path}{query}{body}".encode()


class RevolutXClient:
    def __init__(
        self,
        api_key: str | None = None,
        private_key_pem: bytes | None = None,
        *,
        base_url: str = BASE_URL,
        timeout_s: float = 15.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._key = None
        if private_key_pem is not None:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
            from cryptography.hazmat.primitives.serialization import load_pem_private_key

            key = load_pem_private_key(private_key_pem, password=None)
            if not isinstance(key, Ed25519PrivateKey):
                raise ValueError("Revolut X needs an Ed25519 private key")
            self._key = key

    @classmethod
    def from_env(cls) -> "RevolutXClient":
        """REVOLUTX_API_KEY + REVOLUTX_PRIVATE_KEY_PATH (default ~/.config/revolut-x/private.pem).

        Without credentials the client still serves the public endpoints."""
        api_key = os.environ.get("REVOLUTX_API_KEY")
        key_path = Path(os.environ.get("REVOLUTX_PRIVATE_KEY_PATH", DEFAULT_KEY_PATH)).expanduser()
        pem = key_path.read_bytes() if api_key and key_path.exists() else None
        return cls(api_key if pem else None, pem)

    @property
    def authenticated(self) -> bool:
        return self.api_key is not None and self._key is not None

    def sign(self, timestamp_ms: int, method: str, path: str, query: str = "", body: str = "") -> str:
        if self._key is None:
            raise RevolutXError(401, "no private key configured")
        return base64.b64encode(self._key.sign(signing_message(timestamp_ms, method, path, query, body))).decode()

    # ------------------------------------------------------------------ transport

    def build_request(self, method: str, endpoint: str, params: dict[str, Any] | None = None, body: dict | None = None, *, auth: bool = True) -> urllib.request.Request:
        path = API_PREFIX + endpoint
        query = urllib.parse.urlencode({k: v for k, v in (params or {}).items() if v is not None})
        payload = json.dumps(body, separators=(",", ":")) if body is not None else ""
        headers = {"Accept": "application/json"}
        if payload:
            headers["Content-Type"] = "application/json"
        if auth:
            if not self.authenticated:
                raise RevolutXError(401, "API key and private key required (set REVOLUTX_API_KEY)")
            ts = int(time.time() * 1000)
            headers.update({
                "X-Revx-API-Key": self.api_key,
                "X-Revx-Timestamp": str(ts),
                "X-Revx-Signature": self.sign(ts, method, path, query, payload),
            })
        url = f"{self.base_url}{path}" + (f"?{query}" if query else "")
        return urllib.request.Request(url, data=payload.encode() or None, headers=headers, method=method.upper())

    def _send(self, request: urllib.request.Request) -> Any:
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            try:
                err = json.loads(exc.read() or b"{}")
            except json.JSONDecodeError:
                err = {}
            raise RevolutXError(exc.code, err.get("message") or exc.reason, err.get("error_id")) from None
        except urllib.error.URLError as exc:
            raise RevolutXError(0, f"network error: {exc.reason}") from None
        return json.loads(raw) if raw else None

    async def request(self, method: str, endpoint: str, params: dict[str, Any] | None = None, body: dict | None = None, *, auth: bool = True) -> Any:
        req = self.build_request(method, endpoint, params, body, auth=auth)
        return await asyncio.to_thread(self._send, req)

    # ------------------------------------------------------------------ endpoints

    async def public_order_book(self, symbol: str) -> dict[str, Any]:
        """Top 5 levels per side, no authentication."""
        return (await self.request("GET", f"/public/order-book/{symbol}", auth=False))["data"]

    async def order_book(self, symbol: str, limit: int = 20) -> dict[str, Any]:
        return (await self.request("GET", f"/order-book/{symbol}", {"limit": limit}))["data"]

    async def candles(self, symbol: str, interval: int = 1440, since_ms: int | None = None, until_ms: int | None = None) -> list[dict[str, Any]]:
        return (await self.request("GET", f"/candles/{symbol}", {"interval": interval, "since": since_ms, "until": until_ms}))["data"]

    async def pairs(self) -> dict[str, Any]:
        return await self.request("GET", "/configuration/pairs")

    async def tickers(self, symbols: list[str] | None = None) -> list[dict[str, Any]]:
        return (await self.request("GET", "/tickers", {"symbols": ",".join(symbols) if symbols else None}))["data"]

    async def balances(self) -> list[dict[str, Any]]:
        return await self.request("GET", "/balances")


def summarize_order_book(book: dict[str, Any]) -> dict[str, float | None]:
    """Spread and near-touch liquidity from an order book payload (prices/quantities are strings)."""
    asks = sorted(((float(l["p"]), float(l["q"])) for l in book.get("asks", [])), key=lambda x: x[0])
    bids = sorted(((float(l["p"]), float(l["q"])) for l in book.get("bids", [])), key=lambda x: -x[0])
    if not asks or not bids:
        return {"best_bid": bids[0][0] if bids else None, "best_ask": asks[0][0] if asks else None, "spread_bps": None}
    bid, ask = bids[0][0], asks[0][0]
    mid = (bid + ask) / 2
    bid_depth = sum(p * q for p, q in bids)
    ask_depth = sum(p * q for p, q in asks)
    return {
        "best_bid": bid,
        "best_ask": ask,
        "spread_bps": round((ask - bid) / mid * 10_000, 2),
        "bid_depth_quote": round(bid_depth, 2),
        "ask_depth_quote": round(ask_depth, 2),
        "depth_imbalance": round((bid_depth - ask_depth) / (bid_depth + ask_depth), 4) if bid_depth + ask_depth else None,
        "levels": float(min(len(bids), len(asks))),
    }
