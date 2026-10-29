"""Headless smoke test of every dashboard page (skipped until a model has been trained: run `python -m selfheal train` first)."""
import pytest

from selfheal.config import ROOT, SUBSETS, load_config, resolve

PAGES = ["Fleet overview", "Engine detail", "Work orders", "Data quality & drift", "Model & MLOps", "Live simulation"]
TRAINED = [s for s in SUBSETS if (resolve(load_config(s), "artifacts_dir") / s / "champion.json").exists()]


def _app():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    return at


def _set(at, label, value):
    next(r for r in list(at.sidebar.radio) if r.label == label).set_value(value).run()


@pytest.mark.skipif(not TRAINED, reason="no trained champion yet")
@pytest.mark.parametrize("page", PAGES)
def test_dashboard_page_renders_without_exception(page):
    at = _app()
    _set(at, "View", page)
    assert not at.exception, f"{page}: {at.exception}"
    if page == "Live simulation":
        next(b for b in at.button if b.label == "▶ Step").click().run()
        assert not at.exception, at.exception
        assert any(m.label == "Engines reporting" for m in at.metric)


@pytest.mark.skipif(not TRAINED, reason="no trained champion yet")
@pytest.mark.parametrize("page", ["Fleet overview", "Engine detail", "Work orders", "Data quality & drift"])
def test_fault_scenario_pages_render(page):
    at = _app()
    _set(at, "Replay scenario", "faults")
    _set(at, "View", page)
    assert not at.exception, f"faults/{page}: {at.exception}"
