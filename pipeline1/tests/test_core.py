"""Run with:  PYTHONPATH=src pytest -q   (from the pipeline1 folder)"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge


def test_ridge_closed_form_matches_sklearn():
    from daily_models import ridge_fit
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(12, 5)), rng.normal(size=12)
    m = ridge_fit(X, y, lam=2.0)
    Z = (X - X.mean(0)) / X.std(0)
    assert np.allclose(m["beta"], Ridge(alpha=2.0).fit(Z, y).coef_)


def test_forward_algorithm_matches_hmmlearn():
    from hmm_states import fit_hmm, forward_loglik
    rng = np.random.default_rng(1)
    x = np.r_[rng.normal(-1, .5, 60), rng.normal(1, .5, 60)][:, None]
    m = fit_hmm(x, [120], 2)
    assert forward_loglik(m, x) == pytest.approx(m.score(x), abs=1e-6)


def test_sigma_is_the_mle_with_one_over_n():
    from stage_a import uhr
    a = pd.DataFrame({"hr": [70., 72., 68., 90.]}, index=pd.date_range("2026-01-01", periods=4, freq="5min"))
    pred = np.array([70., 70., 70., 70.])
    _, z, sigma = uhr(a, pred)
    assert sigma == pytest.approx(np.sqrt(np.mean((a.hr.values - pred) ** 2)))     # divides by N, not N-1


def test_parser_keeps_watch_records_only(tmp_path):
    from parse_health import parse
    xml = tmp_path / "export.xml"
    xml.write_text('<HealthData>'
                   '<Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Ann’s Apple Watch" startDate="2026-01-01 10:00:00 +0100" endDate="2026-01-01 10:00:00 +0100" value="70" unit="count/min"/>'
                   '<Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Ann’s iPhone" startDate="2026-01-01 10:05:00 +0100" endDate="2026-01-01 10:05:00 +0100" value="99" unit="count/min"/>'
                   '</HealthData>')
    counts = parse(str(xml), str(tmp_path / "out"))
    assert counts == {"heart_rate": 1}
    assert pd.read_csv(tmp_path / "out" / "heart_rate.csv").value.tolist() == [70]


def test_demo_pipeline_runs_and_finds_planted_effect(tmp_path):
    from synthetic import make
    from pipeline import run_all
    make(str(tmp_path / "watch"), days=14)
    res = run_all(str(tmp_path / "watch"), str(tmp_path / "calendar_tags.csv"), str(tmp_path / "out"))
    assert len(res["mornings"]) >= 10 and res["plan"]
    # planted effect: stress lowers HRV, so readiness should track HRV positively
    m = res["mornings"].dropna(subset=["composite", "hrv_night"])
    assert np.corrcoef(m.composite, m.hrv_night)[0, 1] > 0.3
