import base64

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from trading_desk.revolutx import RevolutXClient, RevolutXError, signing_message, summarize_order_book


@pytest.fixture
def keypair():
    key = Ed25519PrivateKey.generate()
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    return key.public_key(), pem


def test_signing_message_layout():
    msg = signing_message(1700000000000, "get", "/api/1.0/orders/active", "order_states=new&limit=10")
    assert msg == b"1700000000000GET/api/1.0/orders/activeorder_states=new&limit=10"


def test_signed_request_verifies(keypair):
    public, pem = keypair
    client = RevolutXClient("k" * 64, pem)
    req = client.build_request("POST", "/orders", body={"symbol": "BTC-EUR", "side": "buy"})
    ts = req.get_header("X-revx-timestamp")
    assert req.get_header("X-revx-api-key") == "k" * 64
    assert req.data == b'{"symbol":"BTC-EUR","side":"buy"}'  # minified, exactly what was signed
    expected = f'{ts}POST/api/1.0/orders{{"symbol":"BTC-EUR","side":"buy"}}'.encode()
    public.verify(base64.b64decode(req.get_header("X-revx-signature")), expected)  # raises if wrong


def test_query_signed_without_question_mark(keypair):
    public, pem = keypair
    req = RevolutXClient("k", pem).build_request("GET", "/candles/BTC-EUR", {"interval": 1440, "since": None})
    assert req.full_url == "https://revx.revolut.com/api/1.0/candles/BTC-EUR?interval=1440"
    ts = req.get_header("X-revx-timestamp")
    public.verify(base64.b64decode(req.get_header("X-revx-signature")), f"{ts}GET/api/1.0/candles/BTC-EURinterval=1440".encode())


def test_public_request_has_no_auth_headers():
    req = RevolutXClient().build_request("GET", "/public/order-book/BTC-EUR", auth=False)
    assert req.get_header("X-revx-signature") is None


def test_private_request_without_credentials_fails_early():
    with pytest.raises(RevolutXError, match="required"):
        RevolutXClient().build_request("GET", "/balances")


def test_rejects_non_ed25519_key():
    from cryptography.hazmat.primitives.asymmetric import ec

    pem = ec.generate_private_key(ec.SECP256R1()).private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    with pytest.raises(ValueError, match="Ed25519"):
        RevolutXClient("k", pem)


def test_from_env_without_credentials(monkeypatch, tmp_path):
    monkeypatch.delenv("REVOLUTX_API_KEY", raising=False)
    monkeypatch.setenv("REVOLUTX_PRIVATE_KEY_PATH", str(tmp_path / "missing.pem"))
    assert not RevolutXClient.from_env().authenticated


def test_summarize_order_book():
    level = lambda p, q: {"p": str(p), "q": str(q)}
    book = {"asks": [level(101, 1), level(102, 2)], "bids": [level(99, 3), level(100, 1)]}
    s = summarize_order_book(book)
    assert (s["best_bid"], s["best_ask"]) == (100, 101)
    assert s["spread_bps"] == pytest.approx(1 / 100.5 * 10_000, rel=1e-3)
    assert s["bid_depth_quote"] == 397 and s["ask_depth_quote"] == 305
