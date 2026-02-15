from app.services import simulation_agents


def test_simulation_agents_fallback_roundtrip(monkeypatch):
    store: dict[tuple[str, str], dict] = {}

    monkeypatch.setattr("app.services.simulation_agents.get_supabase", lambda: None)
    monkeypatch.setattr(
        "app.services.simulation_agents.get_cached_research",
        lambda symbol, data_type: store.get((symbol, data_type)),
    )

    def _set_cached(symbol: str, data_type: str, data: dict, ttl_minutes: int = 15):
        store[(symbol, data_type)] = data

    monkeypatch.setattr("app.services.simulation_agents.set_cached_research", _set_cached)

    user_id = "00000000-0000-0000-0000-000000000123"
    entry = {
        "config": {
            "name": "My Trading Agent 2",
            "personality": "quant_momentum",
            "model": "meta-llama/llama-3.1-8b-instruct",
            "strategy_prompt": "Buy breakouts.",
            "aggressiveness": 0.7,
            "risk_limit": 0.6,
            "trade_size": 25,
            "active": True,
        },
        "iconEmoji": "🧠",
        "editor": {"risk": 60, "tempo": 65, "style": 70, "news": 45},
    }

    saved = simulation_agents.set_simulation_agents(user_id, [entry])
    assert len(saved) == 1

    loaded = simulation_agents.list_simulation_agents(user_id)
    assert len(loaded) == 1
    assert loaded[0]["config"]["name"] == "My Trading Agent 2"

    after_delete = simulation_agents.delete_simulation_agent(user_id, "My Trading Agent 2")
    assert after_delete == []


def test_simulation_agents_prefers_fallback_when_db_returns_empty(monkeypatch):
    store: dict[tuple[str, str], dict] = {}

    class _FakeResult:
        def __init__(self, data):
            self.data = data

    class _FakeQuery:
        def select(self, *_args, **_kwargs):
            return self

        def eq(self, *_args, **_kwargs):
            return self

        def order(self, *_args, **_kwargs):
            return self

        def execute(self):
            return _FakeResult([])

    class _FakeClient:
        def table(self, _name: str):
            return _FakeQuery()

    monkeypatch.setattr("app.services.simulation_agents.get_supabase", lambda: _FakeClient())
    monkeypatch.setattr(
        "app.services.simulation_agents.get_cached_research",
        lambda symbol, data_type: store.get((symbol, data_type)),
    )
    monkeypatch.setattr(
        "app.services.simulation_agents.set_cached_research",
        lambda symbol, data_type, data, ttl_minutes=15: store.__setitem__((symbol, data_type), data),
    )

    user_id = "00000000-0000-0000-0000-000000000123"
    fallback_entry = {
        "config": {
            "name": "My Trading Agent 9",
            "personality": "quant_momentum",
            "model": "meta-llama/llama-3.1-8b-instruct",
            "strategy_prompt": "Fallback strategy.",
            "aggressiveness": 0.6,
            "risk_limit": 0.5,
            "trade_size": 20,
            "active": True,
        },
        "iconEmoji": "🧠",
        "editor": {"risk": 55, "tempo": 60, "style": 65, "news": 40},
    }
    store[(user_id, "simulation_agents:v1")] = {"user_id": user_id, "entries": [fallback_entry]}

    loaded = simulation_agents.list_simulation_agents(user_id)
    assert len(loaded) == 1
    assert loaded[0]["config"]["name"] == "My Trading Agent 9"
