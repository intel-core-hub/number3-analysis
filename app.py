"""Numbers3 予測・分析システム (簡潔版)

Streamlit ダッシュボード - メインエントリーポイント
"""
import streamlit as st
from src.ui.components.sidebar import render_sidebar, render_health_alert
from src.ui.components.prediction_tab import render_prediction_tab
from src.ui.components.tabs import (
    render_pattern_analysis_tab,
    render_statistical_analysis_tab,
    render_risk_analysis_tab,
    render_data_tab,
    render_system_status_tab,
)

# ページ設定
st.set_page_config(
    page_title="Numbers3 予測・分析システム",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===== サイドバー =====
df, latest_round, latest_date = render_sidebar()
health = render_health_alert()

# ===== メイン =====
st.title("Numbers3 予測・分析システム")

# タブ
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "購入判断",
    "パターン分析",
    "統計分析",
    "リスク分析",
    "データ確認",
    "システムステータス"
])

# 各タブのレンダリング
with tab1:
    render_prediction_tab(df)

with tab2:
    render_pattern_analysis_tab(df)

with tab3:
    render_statistical_analysis_tab(df)

with tab4:
    render_risk_analysis_tab(df, health)

with tab5:
    render_data_tab(df, latest_round, latest_date)

with tab6:
    render_system_status_tab(health)
