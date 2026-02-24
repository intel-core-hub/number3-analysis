"""
src.scraper - data retrieval and database update flow.
"""
from __future__ import annotations

from datetime import datetime
from typing import Tuple

import pandas as pd

from src.config import (
    CSV_FALLBACK_PATH,
    DEFAULT_SCRAPE_SLEEP_SECONDS,
    DEFAULT_SCRAPE_START_DAY,
    DEFAULT_SCRAPE_START_MONTH,
    DEFAULT_SCRAPE_START_YEAR,
)
from src.data.fetcher import fetch_numbers3_by_month
from src.database import Numbers3Database
from src.helpers import get_logger

logger = get_logger(__name__)


def fetch_numbers3_data(
    start_date: datetime,
    end_date: datetime,
    sleep_seconds: float = DEFAULT_SCRAPE_SLEEP_SECONDS,
) -> tuple[pd.DataFrame, list[dict]]:
    """Fetch Numbers3 data for a date range.

    Args:
        start_date: First date to fetch.
        end_date: Last date to fetch.
        sleep_seconds: Delay between monthly requests.

    Returns:
        Tuple of (dataframe, failures).
    """
    try:
        return fetch_numbers3_by_month(
            start_date,
            end_date,
            sleep_seconds=sleep_seconds,
            fail_fast=False,
        )
    except Exception as exc:
        logger.exception("Scraping failed: %s", exc)
        raise


def bootstrap_database(db: Numbers3Database) -> bool:
    """Initialize database from legacy CSV if empty.

    Args:
        db: Database adapter.

    Returns:
        True if migration happened.
    """
    if not db.is_empty():
        return False

    if CSV_FALLBACK_PATH.exists():
        migrated = db.migrate_from_csv(CSV_FALLBACK_PATH)
        logger.info("Migrated %s rows from CSV", migrated)
        return migrated > 0

    return False


def update_numbers3_database(
    db: Numbers3Database,
    force_full: bool = False,
    sleep_seconds: float = DEFAULT_SCRAPE_SLEEP_SECONDS,
) -> pd.DataFrame:
    """Update SQLite database with fresh draws.

    Args:
        db: Database adapter.
        force_full: Force full re-fetch.
        sleep_seconds: Delay between monthly requests.

    Returns:
        Updated draws dataframe.
    """
    bootstrap_database(db)

    if force_full or db.is_empty():
        start_date = datetime(
            DEFAULT_SCRAPE_START_YEAR,
            DEFAULT_SCRAPE_START_MONTH,
            DEFAULT_SCRAPE_START_DAY,
        )
    else:
        latest_date = db.get_latest_draw_date()
        if latest_date:
            parsed = pd.to_datetime(latest_date, errors="coerce")
            if pd.isna(parsed):
                start_date = datetime(
                    DEFAULT_SCRAPE_START_YEAR,
                    DEFAULT_SCRAPE_START_MONTH,
                    DEFAULT_SCRAPE_START_DAY,
                )
            else:
                start_date = datetime(parsed.year, parsed.month, 1)
        else:
            start_date = datetime(
                DEFAULT_SCRAPE_START_YEAR,
                DEFAULT_SCRAPE_START_MONTH,
                DEFAULT_SCRAPE_START_DAY,
            )

    end_date = datetime.now()
    df_new, failures = fetch_numbers3_data(
        start_date=start_date,
        end_date=end_date,
        sleep_seconds=sleep_seconds,
    )

    if failures:
        logger.warning("Scraping failures: %s", failures[:3])

    if df_new is not None and not df_new.empty:
        upserted = db.upsert_draws(df_new)
        logger.info("Upserted %s rows into SQLite", upserted)

    return db.load_draws()
