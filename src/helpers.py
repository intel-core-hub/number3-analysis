"""
src.helpers - shared helpers (logging, Slack notifications).
"""
from __future__ import annotations

import logging
from typing import Optional

import requests

from src.utils.logger import get_logger as _get_logger


def get_logger(name: str) -> logging.Logger:
    """Return a configured project logger.

    Args:
        name: Logger name.

    Returns:
        Configured logger instance.
    """
    return _get_logger(name)


def send_slack_message(webhook_url: str | None, text: str) -> None:
    """Send a Slack notification via incoming webhook.

    Args:
        webhook_url: Slack incoming webhook URL.
        text: Message text to send.
    """
    if not webhook_url:
        logging.getLogger(__name__).warning(
            "SLACK_WEBHOOK_URL is not set. Skipping notification."
        )
        return

    payload = {"text": text}
    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
    except Exception as exc:
        logging.getLogger(__name__).error("Failed to send Slack notification: %s", exc)
