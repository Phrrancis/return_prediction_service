"""
End-to-end integration test: synthetic merchant -> train -> shadow-score -> Money Report.
Mirrors run_demo.py at a much smaller scale, fully isolated to a tmp DB/models dir.
"""

from __future__ import annotations

import backend.train as train_module
from backend.report import generate as generate_report
from backend.score import mature_outcomes, score_pending
from backend.train import train
from demo.generate_synthetic import generate as generate_synthetic

MERCHANT = "testmerchant"
N_CARTS = 700  # comfortably clears train()'s >=500 matured-labeled-cart minimum


def test_full_pipeline_produces_a_money_report(tmp_path, monkeypatch):
    monkeypatch.setattr(train_module, "MODELS_DIR", tmp_path / "models")
    db_path = str(tmp_path / "pipeline.db")

    generate_synthetic(MERCHANT, n_carts=N_CARTS, db_path=db_path)

    metrics, version = train(MERCHANT, db_path=db_path)
    assert metrics["n_train"] > 0
    assert metrics["n_valid"] > 0
    assert 0.0 <= metrics["auc"] <= 1.0

    scored = score_pending(MERCHANT, db_path=db_path)
    assert scored > 0

    n_mature = mature_outcomes(MERCHANT, db_path=db_path)
    assert n_mature > 0

    report = generate_report(MERCHANT, db_path=db_path)
    assert report["merchant_id"] == MERCHANT
    assert report["corpus"]["carts_scored_and_mature"] > 0
    assert 0.0 <= report["corpus"]["verified_auc"] <= 1.0
    assert len(report["operating_points"]) > 0
    for op in report["operating_points"]:
        assert 0.0 <= op["verified_precision"] <= 1.0
        assert 0.0 <= op["recall_of_all_returns"] <= 1.0
    assert "bracketing_return_rate" in report["merchandising"]
