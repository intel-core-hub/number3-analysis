from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="numbers3 開発コンソール", layout="wide")


def load_batch_output() -> pd.DataFrame:
    path = Path("data/processed/latest_predictions.parquet")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def main() -> None:
    st.title("🧪 numbers3 開発・実験コンソール")
    st.caption("重い分析処理（シミュレーション・特徴量検証・可視化）はこの環境で実行してください。")

    tab1, tab2 = st.tabs(["Batch出力確認", "実験メモ"])

    with tab1:
        st.subheader("Batch出力データ")
        df = load_batch_output()
        if df.empty:
            st.info("batch_predict.py の出力がまだありません。")
        else:
            st.dataframe(df, width="stretch", hide_index=True)
            st.line_chart(df[["ev_ratio", "confidence", "ruin_probability"]], width="stretch")

    with tab2:
        st.subheader("実験用ワークフロー")
        st.markdown(
            "\n".join(
                [
                    "1. notebooks/ で特徴量・モデルを検証",
                    "2. batch_predict.py で定期実行向け処理を更新",
                    "3. app.py には結果表示と購買判断UIのみ反映",
                ]
            )
        )


if __name__ == "__main__":
    main()