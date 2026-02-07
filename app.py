from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd

from numbers3_logic import (
    Numbers3Backtester,
    Numbers3FeatureEngineer,
    Numbers3IntervalAnalyzer,
    Numbers3PatternAnalyzer,
    Numbers3Predictor,
    Numbers3TrendAnalyzer,
    PREDICTION_MODELS,
    configure_japanese_fonts,
    update_numbers3_clean,
    validate_numbers3,
)

st.set_page_config(page_title="Numbers3 予測・分析システム", layout="wide")

configure_japanese_fonts()

CLEAN_PATH = "numbers3_clean.csv"

st.title("Numbers3 予測・分析システム")

with st.sidebar:
    st.header("データ更新")
    force_full = st.checkbox("初回フル取得", value=False)
    sleep_seconds = st.number_input("取得間隔 (秒)", min_value=0.0, max_value=5.0, value=1.0, step=0.5)
    do_update = st.button("最新データに更新")


@st.cache_data(show_spinner=False)
def load_clean_data(path):
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def run_update(path, force_full_value, sleep_seconds_value):
    df = update_numbers3_clean(
        clean_path=path,
        backup=True,
        sleep_seconds=sleep_seconds_value,
        force_full=force_full_value,
    )
    validate_numbers3(df, strict=True, allow_missing_rounds=True)
    return df


@st.cache_data(show_spinner=False)
def run_prediction(
    features,
    top_n_value,
    model_value,
    recent_window_value,
    interval_weight_value,
    verbose_value,
):
    predictor = Numbers3Predictor(features)
    return predictor.predict(
        top_n=top_n_value,
        model=model_value,
        recent_window=recent_window_value,
        interval_weight=interval_weight_value,
        verbose=verbose_value,
    )


@st.cache_data(show_spinner=False)
def run_model_comparison(features, top_n_value, recent_window_value, interval_weight_value):
    rows = []
    for label, model in PREDICTION_MODELS.items():
        predictor = Numbers3Predictor(features)
        df_pred = predictor.predict(
            top_n=top_n_value,
            model=model,
            recent_window=recent_window_value,
            interval_weight=interval_weight_value,
            verbose=False,
        )
        df_pred = df_pred.copy()
        df_pred.insert(0, "モデル名", label)
        rows.append(df_pred)
    return pd.concat(rows, ignore_index=True)


def get_update_info(path, df):
    last_updated = None
    path_obj = Path(path)
    if path_obj.exists():
        last_updated = datetime.fromtimestamp(path_obj.stat().st_mtime)

    last_round = None
    if "回号" in df.columns:
        rounds = pd.to_numeric(df["回号"], errors="coerce")
        if not rounds.dropna().empty:
            last_round = int(rounds.max())

    return last_updated, last_round


if do_update:
    with st.spinner("データを取得中..."):
        df = run_update(CLEAN_PATH, force_full, sleep_seconds)
    st.sidebar.success("更新完了！")
else:
    try:
        df = load_clean_data(CLEAN_PATH)
    except FileNotFoundError:
        st.warning("numbers3_clean.csv が見つかりません。左の更新ボタンを押してください。")
        st.stop()

fe = Numbers3FeatureEngineer(window=20)
features_df = fe.add_features(df)

last_updated, last_round = get_update_info(CLEAN_PATH, df)
with st.sidebar:
    st.header("更新情報")
    if last_updated is None:
        st.write("最終更新日時: 未取得")
    else:
        st.write(f"最終更新日時: {last_updated:%Y-%m-%d %H:%M:%S}")

    if last_round is None:
        st.write("最終回号: 不明")
    else:
        st.write(f"最終回号: {last_round}")

tab1, tab2, tab3 = st.tabs(["予測", "統計分析", "データ確認"])

with tab1:
    st.header("次回の予測")
    model_label = st.selectbox("モデル", list(PREDICTION_MODELS.keys()))
    model_value = PREDICTION_MODELS[model_label]
    recent_window = 200
    interval_weight = 0.1
    if model_value == "recent_frequency":
        recent_window = st.slider("直近n回", 20, 500, 120, step=10)
    if model_value == "interval_boost":
        interval_weight = st.slider("ハマリ強調係数", 0.1, 0.6, 0.3, step=0.05)
    top_n = st.slider("表示件数", 5, 50, 20)
    verbose_logs = st.checkbox("ログ出力を表示", value=True)
    run_predict = st.button("予測実行")
    compare_top_n = st.slider("比較表示件数(各モデル)", 1, 30, 5)
    run_compare = st.button("全モデル比較")

    if "prediction_df" not in st.session_state:
        st.session_state.prediction_df = None
        st.session_state.prediction_model = None
        st.session_state.prediction_top_n = None
        st.session_state.prediction_recent_window = None
        st.session_state.prediction_interval_weight = None

    if run_predict:
        with st.spinner("予測を計算中..."):
            st.session_state.prediction_df = run_prediction(
                features_df,
                top_n,
                model_value,
                recent_window,
                interval_weight,
                verbose_logs,
            )
            st.session_state.prediction_model = model_label
            st.session_state.prediction_top_n = top_n
            st.session_state.prediction_recent_window = recent_window
            st.session_state.prediction_interval_weight = interval_weight

    if "comparison_df" not in st.session_state:
        st.session_state.comparison_df = None

    if run_compare:
        with st.spinner("全モデルを比較中..."):
            st.session_state.comparison_df = run_model_comparison(
                features_df,
                compare_top_n,
                recent_window,
                interval_weight,
            )

    if st.session_state.prediction_df is None:
        st.info("モデルと表示件数を選んで「予測実行」を押してください。")
    else:
        st.caption(
            f"モデル: {st.session_state.prediction_model} / 表示件数: {st.session_state.prediction_top_n}"
        )
        if st.session_state.prediction_model == "直近n回(頻度+合計)":
            st.caption(f"直近n回: {st.session_state.prediction_recent_window}")
        if st.session_state.prediction_model == "ハマリ強調(合計+ハマリ)":
            st.caption(f"ハマリ強調係数: {st.session_state.prediction_interval_weight:.2f}")
        st.dataframe(st.session_state.prediction_df, use_container_width=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_bytes = st.session_state.prediction_df.to_csv(index=False, encoding="utf-8-sig").encode(
            "utf-8-sig"
        )
        st.download_button(
            label="予測結果CSVをダウンロード",
            data=csv_bytes,
            file_name=f"numbers3_predictions_{timestamp}.csv",
            mime="text/csv",
        )

    if st.session_state.comparison_df is not None:
        st.subheader("全モデル比較ランキング")
        st.dataframe(st.session_state.comparison_df, use_container_width=True)

        summary_df = (
            st.session_state.comparison_df
            .sort_values(["モデル名", "総合スコア"], ascending=[True, False])
            .groupby("モデル名", as_index=False)
            .head(1)
            .reset_index(drop=True)
        )
        st.subheader("モデルごとの1位まとめ")
        st.dataframe(summary_df, use_container_width=True)

with tab2:
    st.header("傾向分析")

    analysis_type = st.selectbox(
        "分析タイプ",
        [
            "インターバル分布",
            "インターバル累積分布",
            "合計値分布",
            "合計値移動平均",
            "遷移確率ヒートマップ",
        ],
    )

    if analysis_type in ["インターバル分布", "インターバル累積分布"]:
        target_col = st.selectbox("分析対象の桁", ["digit_h", "digit_t", "digit_o"])
        analyzer = Numbers3IntervalAnalyzer(features_df)
        fig = (
            analyzer.interval_hist_fig(target_col)
            if analysis_type == "インターバル分布"
            else analyzer.interval_cdf_fig(target_col)
        )
        if fig is None:
            st.info("インターバルが計算できませんでした。")
        else:
            st.pyplot(fig, clear_figure=True)

    elif analysis_type == "合計値分布":
        analyzer = Numbers3TrendAnalyzer(features_df)
        fig = analyzer.sum_distribution_fig()
        st.pyplot(fig, clear_figure=True)

    elif analysis_type == "合計値移動平均":
        window = st.slider("移動平均の窓", 10, 200, 50, step=5)
        analyzer = Numbers3TrendAnalyzer(features_df)
        fig = analyzer.moving_average_fig(window=window)
        st.pyplot(fig, clear_figure=True)

    else:
        analyzer = Numbers3PatternAnalyzer(features_df)
        target = st.selectbox("対象", ["odd_count", "big_count"])
        title = "奇数の個数の遷移確率" if target == "odd_count" else "大きい数字(5-9)の個数の遷移確率"
        fig = analyzer.transition_heatmap_fig(column=target, title=title)
        st.pyplot(fig, clear_figure=True)

with tab3:
    st.header("データ確認")
    st.write("基本情報")
    st.write(f"行数: {len(df)}")

    if "回号" in df.columns:
        rounds = pd.to_numeric(df["回号"], errors="coerce")
        st.write(f"回号範囲: {int(rounds.min())} 〜 {int(rounds.max())}")

    st.subheader("プレビュー")
    st.dataframe(df.head(50), use_container_width=True)

    st.subheader("バックテスト")
    test_rounds = st.slider("検証回数", 10, 200, 50, step=10)
    top_n = st.slider("候補数", 20, 200, 100, step=10)
    backtester = Numbers3Backtester(features_df, fe, window=200, top_n=top_n)
    bt_results, bt_summary = backtester.run(test_rounds=test_rounds, top_k_list=(10, 50, 100))

    st.write("集計")
    st.dataframe(bt_summary, use_container_width=True)

    st.subheader("セット損益サマリー")
    summary_row = bt_summary.iloc[0] if not bt_summary.empty else None
    if summary_row is None:
        st.info("集計データがありません。")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("総投資", f"{int(summary_row.get('total_cost', 0)):,}円")
        col2.metric("総払戻", f"{int(summary_row.get('total_return', 0)):,}円")
        col3.metric("損益", f"{int(summary_row.get('total_profit', 0)):,}円")
        col4.metric("ROI", f"{summary_row.get('roi_pct', 0.0):.2f}%")

        col5, col6 = st.columns(2)
        col5.metric("セット・ストレート", int(summary_row.get("set_straight_hits", 0)))
        col6.metric("セット・ボックス", int(summary_row.get("set_box_hits", 0)))

    st.write("直近の検証結果")
    st.dataframe(bt_results.head(10), use_container_width=True)
