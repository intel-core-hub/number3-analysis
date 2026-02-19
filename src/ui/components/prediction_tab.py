"""購入判断タブコンポーネント"""
import streamlit as st
from pathlib import Path
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt

from src.config import RESULTS_DIR
from src.predictors import PatternPredictor, MultiLabelPredictor
from src.strategies.selective_purchase import SelectivePurchaseStrategy
from src.ui.analysis_helpers import cached_regime_detection
from src.ui.results_loaders import load_latest_backtest_results
from src.analysis.drift_detector import estimate_drift_auc
from src.features.engineer import compute_all_ml_features, ml_feature_columns
from src.models.meta_learner import MetaErrorPredictor, build_meta_training_frame
from src.models.weight_evolver import load_saved_weights
from src.helpers import get_logger

logger = get_logger(__name__)

def load_optimization_results(path):
    """最適化結果をロード"""
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return None

def recommend_ev_threshold(backtest_df):
    """バックテスト結果から推奨EV閾値を計算"""
    try:
        return float(backtest_df["expected_value"].quantile(0.75)) if len(backtest_df) > 0 else 1.0
    except:
        return 1.0

def render_prediction_tab(df):
    """購入判断タブをレンダリング"""
    st.header("次回の購入判断 (Phase 19)")
    st.info("パターン予測 → 信頼度ゲート → 期待値フィルタ → 購入判断")
    
    # パラメータ設定
    col1, col2, col3 = st.columns(3)
    with col1:
        max_tickets = st.slider("最大購入点数", 1, 20, 5, step=1)
    with col2:
        confidence_threshold = st.slider("信頼度閾値", 0.0, 0.5, 0.0, step=0.01)
    with col3:
        top_k_patterns = st.slider("パターン数(top-k)", 1, 5, 2, step=1)
    
    use_meta_gate = st.checkbox("メタゲートを使用", value=False)
    meta_threshold = st.slider("メタ閾値", 0.5, 0.9, 0.65, step=0.01, disabled=not use_meta_gate)
    
    if st.button("購入判断を実行", type="primary"):
        if df.empty:
            st.error("データが空です")
            return
        
        with st.spinner("計算中..."):
            try:
                # 予測
                pattern_pred = PatternPredictor(df)
                pattern_pred.train()
                hl_patterns, oe_patterns = pattern_pred.predict_top_patterns(top_k=top_k_patterns)
                
                ml_pred = MultiLabelPredictor(df)
                ml_pred.train()
                digit_probs = ml_pred.predict_digit_probabilities()
                
                # ドリフト検知
                drift_df = compute_all_ml_features(df)
                feature_cols = [c for c in ml_feature_columns() if c in drift_df.columns]
                drift_score = estimate_drift_auc(drift_df, feature_cols=feature_cols)
                
                # メタモデル
                meta_model = None
                if use_meta_gate:
                    meta_results = load_latest_backtest_results(RESULTS_DIR)
                    if meta_results is not None:
                        meta_model = MetaErrorPredictor()
                        meta_frame = build_meta_training_frame(meta_results)
                        meta_model.fit(meta_frame)
                
                # 戦略
                strategy = SelectivePurchaseStrategy(
                    ev_threshold=0.0,
                    confidence_threshold=confidence_threshold,
                    max_tickets=max_tickets,
                    top_k_patterns=top_k_patterns,
                    payout_history=df,
                    enable_meta_gate=use_meta_gate,
                    meta_threshold=meta_threshold,
                    meta_model=meta_model,
                )
                decision = strategy.decide(
                    hl_probs=pattern_pred.last_hl_probs,
                    oe_probs=pattern_pred.last_oe_probs,
                    digit_probs=digit_probs,
                    candidate_generator=pattern_pred.generate_candidates,
                    payout_history=df,
                    drift_score=drift_score,
                )
                
                # 保存
                st.session_state.decision = decision
                st.session_state.pred_data = {
                    'hl_probs': pattern_pred.last_hl_probs,
                    'oe_probs': pattern_pred.last_oe_probs,
                    'digit_probs': digit_probs,
                }
                st.success("計算完了！")
                st.rerun()
            except Exception as e:
                st.error(f"エラー: {e}")
                logger.exception(f"Prediction error: {e}")
    
    # 結果表示
    if "decision" in st.session_state:
        decision = st.session_state.decision
        pred_data = st.session_state.pred_data
        
        st.divider()
        
        # 購入判定
        if decision.should_buy:
            st.success(f"### 購入推奨\n**{decision.num_tickets}点** (費用: {decision.total_cost:,}円)")
        else:
            st.warning(f"### スキップ\n{decision.reason}")
        
        # メトリクス
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("信頼度", f"{decision.confidence:.3f}")
        col_m2.metric("期待値/枚", f"{decision.expected_value:+,.0f}円")
        col_m3.metric("購入点数", decision.num_tickets)
        col_m4.metric("費用", f"{decision.total_cost:,}円")
        
        # パターン予測
        st.subheader("パターン予測")
        col_hl, col_oe = st.columns(2)
        
        with col_hl:
            hl_sorted = sorted(pred_data['hl_probs'].items(), key=lambda x: x[1], reverse=True)
            hl_df = pd.DataFrame(hl_sorted, columns=["パターン", "確率"])
            hl_df["確率"] = hl_df["確率"].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(hl_df, use_container_width=True)
        
        with col_oe:
            oe_sorted = sorted(pred_data['oe_probs'].items(), key=lambda x: x[1], reverse=True)
            oe_df = pd.DataFrame(oe_sorted, columns=["パターン", "確率"])
            oe_df["確率"] = oe_df["確率"].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(oe_df, use_container_width=True)
        
        # 数字確率
        st.subheader("数字出現確率")
        prob_array = np.array([pred_data['digit_probs'][i] for i in range(10)]).reshape(2, 5)
        fig, ax = plt.subplots(figsize=(8, 3))
        im = ax.imshow(prob_array, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(5))
        ax.set_xticklabels([str(i) for i in range(5)])
        ax.set_yticks(range(2))
        ax.set_yticklabels(["0-4", "5-9"])
        for i in range(2):
            for j in range(5):
                digit = i * 5 + j
                ax.text(j, i, f"{digit}\n{pred_data['digit_probs'][digit]*100:.1f}%",
                       ha="center", va="center", color="black", fontsize=10)
        ax.set_title("数字出現確率ヒートマップ")
        plt.colorbar(im, ax=ax)
        st.pyplot(fig, clear_figure=True)
