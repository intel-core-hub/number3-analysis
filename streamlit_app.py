# streamlit_app.py
import streamlit as st
import pandas as pd
from pathlib import Path

st.title("📊 ナンバーズ3 予測ダッシュボード")

project_root = Path(__file__).resolve().parent
results_dir = project_root / "results"

def _load_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return None

# --- 最新のML予測 ---
st.header("🤖 最新のML予測")

lgbm_path = results_dir / "lightgbm_predictions.csv"
lgbm_df = _load_csv(lgbm_path)

if lgbm_df is not None and not lgbm_df.empty:
    if "date" in lgbm_df.columns:
        lgbm_df = lgbm_df.sort_values("date", ascending=False)

    latest = lgbm_df.iloc[0]
    predicted_number = f"{latest['hundreds']}{latest['tens']}{latest['ones']}"

    st.metric("予測番号", predicted_number)
    if "date" in latest:
        st.caption(f"予測日: {latest['date']}")

    with st.expander("予測履歴を見る"):
        st.dataframe(lgbm_df)
else:
    st.warning("LightGBMの予測結果がまだありません")

# --- バックテスト成績 ---
st.header("📈 バックテスト成績")

perf_path = results_dir / "ml_performance_tracking.csv"
perf_df = _load_csv(perf_path)

if perf_df is not None and not perf_df.empty:
    total_tests = len(perf_df)
    straight_hits = int(perf_df.get("straight_hit", pd.Series([0])).sum())
    box_hits = int(perf_df.get("box_hit", pd.Series([0])).sum())
    total_profit = int(perf_df.get("profit", pd.Series([0])).sum())
    total_cost = int(total_tests * 200)
    roi_percent = (total_profit / total_cost * 100) if total_cost else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("テスト回数", f"{total_tests}回")
    col2.metric("ストレート的中", f"{straight_hits}回")
    col3.metric("ボックス的中", f"{box_hits}回")
    col4.metric("ROI", f"{roi_percent:.2f}%")

    with st.expander("詳細データを見る"):
        display_columns = [
            "timestamp",
            "backtest_id",
            "target_round",
            "actual_number",
            "predicted_number",
            "straight_hit",
            "box_hit",
            "logloss_avg",
            "confidence_avg",
            "profit",
            "cumulative_profit",
        ]
        available_columns = [c for c in display_columns if c in perf_df.columns]
        st.dataframe(perf_df[available_columns])
else:
    st.warning("バックテスト成績がまだありません")
