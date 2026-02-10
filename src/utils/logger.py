"""
src.utils.logger — ログユーティリティ

責務:
    - プロジェクト共通のロガーを取得
    - ログ出力先の統一 (コンソール + ファイル)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

_CONFIGURED = False

LOG_DIR = "logs"
LOG_FILE = "numbers3.log"
DEFAULT_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def get_logger(
    name: str = "numbers3",
    level: int = DEFAULT_LEVEL,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """プロジェクト共通のロガーを返す.

    Parameters
    ----------
    name : str
        ロガー名 (通常はモジュール名)
    level : int
        ログレベル
    log_file : str, optional
        ログファイルパス (デフォルトは logs/numbers3.log)

    Returns
    -------
    logging.Logger
    """
    global _CONFIGURED

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # コンソールハンドラ
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console_handler)

    # ファイルハンドラ (初回のみ)
    if not _CONFIGURED:
        if log_file is None:
            log_file = os.path.join(LOG_DIR, LOG_FILE)
        Path(os.path.dirname(log_file)).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(file_handler)
        _CONFIGURED = True

    return logger
