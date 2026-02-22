"""
src.ui.data_loaders — データ取得・キャッシング関数

責務:
    - データベースからの抽選履歴読み込み
    - Streamlit キャッシングによる高速化
    - 最新回号情報の取得
"""
from typing import Optional, Tuple

import pandas as pd
import streamlit as st

from src.database import Numbers3Database
from src.scraper import update_numbers3_database


def load_draws(db: Numbers3Database) -> pd.DataFrame:
    """データベースから抽選履歴を読み込む

    Args:
        db: Numbers3Database インスタンス

    Returns:
        抽選履歴の DataFrame
    """
    return db.load_draws()


@st.cache_data(show_spinner=False)
def cached_load_draws(db_path: str) -> pd.DataFrame:
    """データベースから抽選履歴をキャッシュ付きで読み込む

    Streamlit のキャッシュ機構を使用して、同じデータベースパスに対する
    読み込みを高速化する。

    Args:
        db_path: データベースファイルのパス

    Returns:
        抽選履歴の DataFrame
    """
    db = Numbers3Database(db_path)
    return load_draws(db)


@st.cache_data(show_spinner=False)
def cached_update_draws(
    db_path: str, force_full_value: bool, sleep_seconds_value: float
) -> pd.DataFrame:
    """データベースを更新してデータを読み込む

    スクレイピングによりデータベースを更新し、最新の抽選履歴を返す。
    Streamlit キャッシュにより、同じパラメータでの重複実行を防ぐ。

    Args:
        db_path: データベースファイルのパス
        force_full_value: 全件取得を強制するか
        sleep_seconds_value: スクレイピング間隔（秒）

    Returns:
        更新後の抽選履歴 DataFrame
    """
    db = Numbers3Database(db_path)
    return update_numbers3_database(
        db, force_full=force_full_value, sleep_seconds=sleep_seconds_value
    )


def latest_round_info(
    df: pd.DataFrame,
) -> tuple[int | None, pd.Timestamp | None]:
    """最新回号と抽せん日を取得する

    Args:
        df: 抽選履歴の DataFrame

    Returns:
        (最新回号, 最新抽せん日) のタプル
        データが存在しない場合は (None, None)
    """
    if df.empty or "回号" not in df.columns:
        return None, None

    rounds = pd.to_numeric(df["回号"], errors="coerce").dropna()
    if rounds.empty:
        return None, None

    latest_round = int(rounds.max())
    latest_date = None

    if "抽せん日" in df.columns:
        dates = pd.to_datetime(df["抽せん日"], errors="coerce")
        latest_date = dates.max()

    return latest_round, latest_date
