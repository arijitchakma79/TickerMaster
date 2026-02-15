from fastapi.testclient import TestClient

from app.main import app


def test_market_movers_uses_cached_snapshot(monkeypatch):
    cached_snapshot = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "tickers": [
            {"ticker": "AAA", "price": 10, "change_percent": 4.2},
            {"ticker": "BBB", "price": 20, "change_percent": -3.1},
            {"ticker": "CCC", "price": 30, "change_percent": 1.1},
        ],
    }

    def _get_cached(symbol: str, data_type: str):
        if symbol == "GLOBAL" and data_type == "movers:snapshot:1h:v1":
            return cached_snapshot
        return None

    monkeypatch.setattr("app.routers.research.get_cached_research", _get_cached)

    with TestClient(app) as client:
        response = client.get("/research/movers", params={"limit": 3})

    assert response.status_code == 200
    payload = response.json()
    assert payload["universe_size"] == 3
    assert len(payload["winners"]) == 2
    assert len(payload["losers"]) == 1


def test_market_movers_falls_back_to_quote_last(monkeypatch):
    class _Metric:
        def __init__(self, ticker: str, price: float, change_percent: float):
            self._payload = {
                "ticker": ticker,
                "price": price,
                "change_percent": change_percent,
            }

        def model_dump(self):
            return dict(self._payload)

    def _fetch_metric(symbol: str):
        if symbol == "AAA":
            return _Metric("AAA", 101.0, 3.5)
        raise RuntimeError("provider unavailable")

    def _get_cached(symbol: str, data_type: str):
        if symbol == "GLOBAL" and data_type in {"movers:snapshot:1h:v1", "movers:snapshot:last:v1"}:
            return None
        if symbol == "BBB" and data_type == "quote:last":
            return {"ticker": "BBB", "price": 77.0, "change_percent": -2.2}
        return None

    monkeypatch.setattr("app.routers.research._GENERAL_MOVER_UNIVERSE", ["AAA", "BBB"])
    monkeypatch.setattr("app.routers.research.fetch_metric", _fetch_metric)
    monkeypatch.setattr("app.routers.research.get_cached_research", _get_cached)
    monkeypatch.setattr("app.routers.research.set_cached_research", lambda *_args, **_kwargs: None)

    with TestClient(app) as client:
        response = client.get("/research/movers", params={"limit": 3})

    assert response.status_code == 200
    payload = response.json()
    assert payload["universe_size"] == 2
    tickers = payload["tickers"]
    assert any(row.get("ticker") == "AAA" for row in tickers)
    assert any(row.get("ticker") == "BBB" and row.get("stale") is True for row in tickers)
