from __future__ import annotations

from fastapi.testclient import TestClient
from starlette.requests import Request

from app.main import app
from app.services.user_context import get_user_id_from_request


def _request_with_query(user_id: str) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [],
        "query_string": f"user_id={user_id}".encode("utf-8"),
    }
    return Request(scope)


def test_research_indicators_endpoint_returns_snapshot(monkeypatch):
    class Point:
        def __init__(self, close: float, high: float, low: float, volume: float):
            self.close = close
            self.high = high
            self.low = low
            self.volume = volume

    data = [Point(100 + idx * 0.5, 101 + idx * 0.5, 99 + idx * 0.5, 1000 + idx * 10) for idx in range(260)]

    monkeypatch.setattr("app.routers.research.fetch_candles", lambda *_args, **_kwargs: data)
    monkeypatch.setattr("app.routers.research.resolve_symbol_input", lambda raw: raw.upper())

    with TestClient(app) as client:
        response = client.get("/research/indicators/AAPL")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ticker"] == "AAPL"
    assert "latest" in payload
    assert isinstance(payload["available"], list)
    assert "sma20" in payload["latest"]


def test_research_quote_handles_provider_failure(monkeypatch):
    def _boom(_symbol: str):
        raise RuntimeError("provider down")

    monkeypatch.setattr("app.routers.research.fetch_metric", _boom)
    monkeypatch.setattr("app.routers.research.resolve_symbol_input", lambda raw: raw.upper())

    with TestClient(app) as client:
        response = client.get("/research/quote/AAPL")
    assert response.status_code == 502
    assert "provider unavailable" in response.json()["detail"].lower()


def test_query_user_fallback_disabled_in_production(monkeypatch):
    user_id = "00000000-0000-0000-0000-000000000123"
    monkeypatch.setenv("ENVIRONMENT", "production")
    request = _request_with_query(user_id)
    assert get_user_id_from_request(request) is None


def test_query_user_fallback_allowed_in_development(monkeypatch):
    user_id = "00000000-0000-0000-0000-000000000123"
    monkeypatch.setenv("ENVIRONMENT", "development")
    request = _request_with_query(user_id)
    assert get_user_id_from_request(request) == user_id
