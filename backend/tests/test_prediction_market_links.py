from app.services.prediction_markets import _build_polymarket_link


def test_build_polymarket_link_prefers_valid_polymarket_url():
    item = {"url": "https://polymarket.com/event/will-fed-cut-rates-in-june"}
    assert _build_polymarket_link(item) == "https://polymarket.com/event/will-fed-cut-rates-in-june"


def test_build_polymarket_link_uses_market_slug():
    item = {"slug": "will-nvda-be-above-200-by-june"}
    assert _build_polymarket_link(item) == "https://polymarket.com/market/will-nvda-be-above-200-by-june"


def test_build_polymarket_link_rejects_invalid_slug_and_falls_back_to_search():
    item = {"slug": "NVIDIA earnings beat?", "question": "Will NVIDIA beat earnings?"}
    assert _build_polymarket_link(item) == "https://polymarket.com/search?q=Will%20NVIDIA%20beat%20earnings%3F"


def test_build_polymarket_link_rejects_non_polymarket_url():
    item = {"url": "https://gamma-api.polymarket.com/markets/123", "question": "Will SPY close green?"}
    assert _build_polymarket_link(item) == "https://polymarket.com/search?q=Will%20SPY%20close%20green%3F"
