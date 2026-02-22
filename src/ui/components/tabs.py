"""その他のタブコンポーネント"""
import pandas as pd
import streamlit as st


def compute_high_low_counts(df: pd.DataFrame, result_col: str = "result") -> pd.Series:
    if df.empty or result_col not in df.columns:
        return pd.Series(dtype="int64")
    return df[result_col].apply(lambda x: 'High' if int(str(x)[:2]) >= 50 else 'Low').value_counts()


def compute_odd_even_counts(df: pd.DataFrame, result_col: str = "result") -> pd.Series:
    if df.empty or result_col not in df.columns:
        return pd.Series(dtype="int64")
    return df[result_col].apply(lambda x: 'Odd' if int(str(x)[:-1]) % 2 == 1 else 'Even').value_counts()


def compute_digit_distribution(df: pd.DataFrame, result_col: str = "result") -> pd.Series:
    if df.empty or result_col not in df.columns:
        return pd.Series(index=range(10), data=[0] * 10, dtype="int64")
    digit_dist = pd.Series(index=range(10), dtype=int)
    for i in range(10):
        count = df[result_col].astype(str).str.count(str(i)).sum()
        digit_dist[i] = int(count)
    return digit_dist


def compute_risk_status(health) -> str:
    if not health:
        return "unknown"
    status = str(getattr(health, "overall_status", "")).lower()
    if status == "critical":
        return "critical"
    if status == "warning":
        return "warning"
    return "ok"


def compute_drawdown_series(df: pd.DataFrame, result_col: str = "result") -> pd.Series:
    if df.empty or result_col not in df.columns:
        return pd.Series(dtype="int64")
    return df[result_col].astype(int).cumsum()


def render_pattern_analysis_tab(df):
    """パターン分析タブ"""
    st.header("パターン分析")
    if df.empty:
        st.warning("データがありません")
        return

    # High/Low分析
    st.subheader("High/Low分析")
    hl_counts = compute_high_low_counts(df)
    st.bar_chart(hl_counts)

    # Odd/Even分析
    st.subheader("Odd/Even分析")
    oe_counts = compute_odd_even_counts(df)
    st.bar_chart(oe_counts)

def render_statistical_analysis_tab(df):
    """統計分析タブ"""
    st.header("統計分析")
    if df.empty:
        st.warning("データがありません")
        return

    st.subheader("基本統計量")
    st.dataframe(df.describe())

    st.subheader("出現分布")
    digit_dist = compute_digit_distribution(df)
    st.bar_chart(digit_dist)

def render_risk_analysis_tab(df, health=None):
    """リスク分析タブ"""
    st.header("リスク分析")

    if health:
        st.subheader("システムヘルス")
        risk_status = compute_risk_status(health)
        if risk_status == "critical":
            st.error("🛑 CRITICAL")
        elif risk_status == "warning":
            st.warning("⚠️ WARNING")
        else:
            st.success("✅ OK")

    if not df.empty:
        st.subheader("ドローダウン")
        cumsum = compute_drawdown_series(df)
        st.line_chart(cumsum)

def render_data_tab(df, latest_round=None, latest_date=None):
    """データ確認タブ"""
    st.header("データ確認")

    col1, col2 = st.columns(2)
    with col1:
        if latest_round:
            st.metric("最新回号", latest_round)
    with col2:
        if latest_date:
            st.metric("最新日時", latest_date.strftime("%Y-%m-%d"))

    st.subheader("直近データ")
    if not df.empty:
        st.dataframe(df.head(20), use_container_width=True)
    else:
        st.info("データなし")

def render_system_status_tab(health=None):
    """システムステータスタブ"""
    st.header("システムステータス")

    if health:
        st.subheader("ヘルスチェック")
        st.json({
            "status": health.overall_status,
            "timestamp": str(health.timestamp),
        })
    else:
        st.info("ヘルス情報取得不可")
