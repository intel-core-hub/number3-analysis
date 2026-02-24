"""
src.ui.components.sidebar — Streamlit サイドバーコンポーネント

責務:
    - データ更新パネル
    - システム設定パネル
"""
from pathlib import Path

import streamlit as st

from src.config import CSV_FALLBACK_PATH, DB_PATH
from src.database import Numbers3Database
from src.helpers import get_logger
from src.scraper import bootstrap_database
from src.ui.data_loaders import cached_load_draws, cached_update_draws, latest_round_info

logger = get_logger(__name__)


def render_sidebar():
    """サイドバー全体をレンダリング

    Returns:
        tuple: (df, latest_round, latest_date)
    """
    db = Numbers3Database(DB_PATH)

    with st.sidebar:
        st.header("データ更新")
        force_full = st.checkbox("初回フル取得", value=False)
        sleep_seconds = st.number_input(
            "取得間隔 (秒)", min_value=0.0, max_value=5.0, value=1.0, step=0.5
        )
        do_update = st.button("最新データに更新")
        migrate_csv = st.button("CSVからSQLiteへ移行")
        auto_repair_schema = st.button("スキーマ自動修復")

    if auto_repair_schema:
        with st.spinner("スキーマを自動修復中..."):
            repair_report = db.auto_repair_all_schemas()
            cached_load_draws.clear()

        repaired = repair_report.get("repaired_tables", [])
        failed = repair_report.get("failed_tables", [])
        if failed:
            st.sidebar.error(f"修復失敗テーブル: {', '.join(failed)}")
        else:
            st.sidebar.success("スキーマ自動修復が完了しました。")
        if repaired:
            st.sidebar.info(f"修復/再生成: {', '.join(repaired)}")

    if migrate_csv:
        if CSV_FALLBACK_PATH.exists():
            with st.spinner("CSVデータをSQLiteへ移行中..."):
                migrated = db.migrate_from_csv(CSV_FALLBACK_PATH)
                cached_load_draws.clear()
            st.sidebar.success(f"移行完了: {migrated}件")
        else:
            st.sidebar.warning("numbers3_clean.csv が見つかりません。")

    if do_update:
        with st.spinner("データを取得中..."):
            df = cached_update_draws(str(DB_PATH), force_full, sleep_seconds)
            cached_load_draws.clear()
        st.sidebar.success("更新完了！")
    else:
        bootstrap_database(db)
        df = cached_load_draws(str(DB_PATH))

    # 最新情報表示
    if not df.empty:
        latest_round, latest_date = latest_round_info(df)
        with st.sidebar:
            st.header("更新情報")
            if latest_date is None:
                st.write("最終更新日時: 未取得")
            else:
                st.write(f"最終更新日時: {latest_date:%Y-%m-%d}")
            if latest_round is None:
                st.write("最終回号: 不明")
            else:
                st.write(f"最終回号: {latest_round}")
    else:
        latest_round, latest_date = None, None

    return df, latest_round, latest_date


def render_health_alert():
    """システムヘルスアラート表示"""
    from src.utils.health_check import run_health_check

    try:
        import psutil
    except Exception:
        psutil = None

    with st.container():
        health = run_health_check()
        if health.overall_status == "critical":
            st.error("🛑 システムヘルス: CRITICAL")
        elif health.overall_status == "warning":
            st.warning("⚠️ システムヘルス: WARNING")

        if psutil is not None:
            cpu = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory().percent
            if cpu > 90:
                st.error(f"🛑 CPU使用率が高すぎます: {cpu:.1f}%")
            elif cpu > 75:
                st.warning(f"⚠️ CPU使用率が高めです: {cpu:.1f}%")

            if mem > 90:
                st.error(f"🛑 メモリ使用率が高すぎます: {mem:.1f}%")
            elif mem > 75:
                st.warning(f"⚠️ メモリ使用率が高めです: {mem:.1f}%")

        try:
            from src.strategies.risk_manager import CircuitBreaker
            from src.ui.analysis_helpers import cached_regime_detection
            from src.utils.paper_trader import PaperTrader

            regime = cached_regime_detection(None)  # df is None here, will be handled in main
            if regime is None:
                return health

            trader = PaperTrader(results_dir=Path('results'))
            breaker = CircuitBreaker()
            max_dd = breaker.compute_max_drawdown_pct(trader.get_equity_curve())
            triggered, reason = breaker.should_block(
                max_drawdown_pct=max_dd,
                regime_confidence=regime.confidence if regime else None,
                drift_score=health.drift_score_last,
            )
            if triggered:
                st.error(f"🛑 サーキットブレーカー発動中: {reason}")
        except Exception as exc:
            st.info(f"サーキットブレーカー評価をスキップしました: {exc}")

    return health
