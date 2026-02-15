from app.services.prediction_markets import _semantic_relevance


def test_semantic_relevance_rejects_generic_macro_market_for_single_stock():
    item = {
        "question": "Will the U.S. collect between $500b and $1t in revenue in 2025?",
        "description": "Federal revenue forecast market.",
    }
    assert _semantic_relevance(item, "CELH", company_name="Celsius Holdings") == 0.0


def test_semantic_relevance_accepts_direct_ticker_market():
    item = {
        "question": "Will CELH close above 80 by June 30?",
        "description": "Celsius Holdings stock price market.",
    }
    assert _semantic_relevance(item, "CELH", company_name="Celsius Holdings") > 0.0
