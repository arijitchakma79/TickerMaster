from __future__ import annotations

from fastapi import Request
from fastapi.testclient import TestClient

from app.main import app
from app.services.user_context import get_user_id_from_request


def _build_request(path: str) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "query_string": b"",
        "headers": [],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "scheme": "http",
    }
    return Request(scope)


def test_query_user_id_fallback_removed() -> None:
    req = _build_request("/api/tracker/agents?user_id=00000000-0000-0000-0000-000000000001")
    assert get_user_id_from_request(req) is None


def test_unauthenticated_tracker_agent_list_is_rejected() -> None:
    with TestClient(app) as client:
        response = client.get("/api/tracker/agents")
    assert response.status_code == 401


def test_unauthenticated_tracker_agent_create_is_rejected() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/tracker/agents",
            json={"symbol": "AAPL", "name": "Agent Alpha", "triggers": {}, "auto_simulate": False},
        )
    assert response.status_code == 401
