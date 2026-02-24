from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd

from src.ui.components import sidebar as sidebar_module


class _DummyContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _streamlit_mock(button_values: list[bool]) -> MagicMock:
    st = MagicMock()
    st.sidebar = MagicMock()
    st.sidebar.__enter__.return_value = st.sidebar
    st.sidebar.__exit__.return_value = False
    st.container.return_value = _DummyContext()
    st.spinner.return_value = _DummyContext()
    st.checkbox.return_value = False
    st.number_input.return_value = 1.0
    st.button.side_effect = button_values
    return st


def test_render_sidebar_auto_repair_success(monkeypatch):
    st = _streamlit_mock([False, False, True])
    monkeypatch.setattr(sidebar_module, "st", st)

    load_df = pd.DataFrame({"回号": [1, 2], "抽せん日": ["2025-01-01", "2025-01-02"]})
    mock_cached_load = MagicMock(return_value=load_df)
    mock_cached_load.clear = MagicMock()
    monkeypatch.setattr(sidebar_module, "cached_load_draws", mock_cached_load)
    monkeypatch.setattr(sidebar_module, "cached_update_draws", MagicMock())
    monkeypatch.setattr(sidebar_module, "bootstrap_database", MagicMock())

    db = MagicMock()
    db.auto_repair_all_schemas.return_value = {
        "repaired_tables": ["numbers3_draws", "predictions"],
        "failed_tables": [],
        "results": {},
    }
    monkeypatch.setattr(sidebar_module, "Numbers3Database", MagicMock(return_value=db))

    df, latest_round, latest_date = sidebar_module.render_sidebar()

    assert len(df) == 2
    assert latest_round == 2
    assert latest_date is not None
    st.sidebar.success.assert_called_with("スキーマ自動修復が完了しました。")
    st.sidebar.info.assert_called()
    mock_cached_load.clear.assert_called()


def test_render_sidebar_migrate_csv_missing(monkeypatch, tmp_path):
    st = _streamlit_mock([False, True, False])
    monkeypatch.setattr(sidebar_module, "st", st)

    mock_cached_load = MagicMock(return_value=pd.DataFrame())
    mock_cached_load.clear = MagicMock()
    monkeypatch.setattr(sidebar_module, "cached_load_draws", mock_cached_load)
    monkeypatch.setattr(sidebar_module, "cached_update_draws", MagicMock())
    monkeypatch.setattr(sidebar_module, "bootstrap_database", MagicMock())
    monkeypatch.setattr(sidebar_module, "CSV_FALLBACK_PATH", tmp_path / "missing.csv")
    monkeypatch.setattr(sidebar_module, "Numbers3Database", MagicMock(return_value=MagicMock()))

    sidebar_module.render_sidebar()

    st.sidebar.warning.assert_called_with("numbers3_clean.csv が見つかりません。")


def test_render_sidebar_update_branch(monkeypatch):
    st = _streamlit_mock([True, False, False])
    monkeypatch.setattr(sidebar_module, "st", st)

    updated_df = pd.DataFrame({"回号": [100], "抽せん日": ["2026-01-01"]})
    mock_cached_update = MagicMock(return_value=updated_df)
    mock_cached_load = MagicMock(return_value=pd.DataFrame())
    mock_cached_load.clear = MagicMock()
    bootstrap_mock = MagicMock()

    monkeypatch.setattr(sidebar_module, "cached_update_draws", mock_cached_update)
    monkeypatch.setattr(sidebar_module, "cached_load_draws", mock_cached_load)
    monkeypatch.setattr(sidebar_module, "bootstrap_database", bootstrap_mock)
    monkeypatch.setattr(sidebar_module, "Numbers3Database", MagicMock(return_value=MagicMock()))

    df, latest_round, _ = sidebar_module.render_sidebar()

    assert latest_round == 100
    assert len(df) == 1
    assert mock_cached_update.called
    bootstrap_mock.assert_not_called()
    st.sidebar.success.assert_called_with("更新完了！")


def test_render_health_alert_critical_with_resource_alerts(monkeypatch):
    st = _streamlit_mock([False, False, False])
    monkeypatch.setattr(sidebar_module, "st", st)

    health = SimpleNamespace(overall_status="critical", drift_score_last=0.1)
    monkeypatch.setattr("src.utils.health_check.run_health_check", lambda: health)

    fake_psutil = SimpleNamespace(
        cpu_percent=lambda interval=0.1: 95.0,
        virtual_memory=lambda: SimpleNamespace(percent=92.0),
    )
    monkeypatch.setitem(__import__("sys").modules, "psutil", fake_psutil)
    monkeypatch.setattr("src.ui.analysis_helpers.cached_regime_detection", lambda _df: None)

    result = sidebar_module.render_health_alert()

    assert result is health
    st.error.assert_called()


def test_render_health_alert_circuit_breaker_exception(monkeypatch):
    st = _streamlit_mock([False, False, False])
    monkeypatch.setattr(sidebar_module, "st", st)

    health = SimpleNamespace(overall_status="warning", drift_score_last=0.5)
    monkeypatch.setattr("src.utils.health_check.run_health_check", lambda: health)
    monkeypatch.setattr("src.ui.analysis_helpers.cached_regime_detection", lambda _df: SimpleNamespace(confidence=0.8))

    class _BrokenTrader:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("broken")

    monkeypatch.setattr("src.utils.paper_trader.PaperTrader", _BrokenTrader)

    sidebar_module.render_health_alert()

    st.info.assert_called()
