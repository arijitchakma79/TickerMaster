from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import get_settings
from app.routers import api as api_router


class _DummyWSManager:
    async def broadcast(self, payload, channel=None):  # pragma: no cover - helper
        return None


def _build_test_client() -> TestClient:
    app = FastAPI()
    app.include_router(api_router.router)
    app.state.settings = get_settings()
    app.state.ws_manager = _DummyWSManager()
    app.state.orchestrator = SimpleNamespace()
    return TestClient(app)


class ApiHardeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _build_test_client()

    def test_tracker_agents_requires_auth(self) -> None:
        response = self.client.get("/api/tracker/agents")
        self.assertEqual(response.status_code, 401)

    def test_create_tracker_agent_requires_auth(self) -> None:
        response = self.client.post(
            "/api/tracker/agents",
            json={"symbol": "AAPL", "name": "Agent"},
        )
        self.assertEqual(response.status_code, 401)

    def test_patch_preferences_requires_auth(self) -> None:
        response = self.client.patch(
            "/api/user/preferences",
            json={"display_name": "tester"},
        )
        self.assertEqual(response.status_code, 401)

    def test_patch_missing_tracker_agent_returns_404(self) -> None:
        response = self.client.patch(
            "/api/tracker/agents/missing-agent",
            headers={"x-user-id": "11111111-1111-1111-1111-111111111111"},
            json={"status": "active"},
        )
        self.assertEqual(response.status_code, 404)

    def test_ticker_full_returns_502_when_quote_fails(self) -> None:
        class _DummyResearch:
            def model_dump(self):
                return {"summary": "ok"}

        with (
            patch.object(api_router, "fetch_metric", side_effect=RuntimeError("upstream quote failure")),
            patch.object(api_router, "run_research", new=AsyncMock(return_value=_DummyResearch())),
            patch.object(api_router, "get_macro_indicators", new=AsyncMock(return_value={"inflation": 2.0})),
            patch.object(api_router, "run_deep_research", new=AsyncMock(return_value={"ok": True})),
        ):
            response = self.client.get("/api/ticker/NVDA")
        self.assertEqual(response.status_code, 502)
        self.assertIn("Market data provider unavailable", response.text)


if __name__ == "__main__":
    unittest.main()
