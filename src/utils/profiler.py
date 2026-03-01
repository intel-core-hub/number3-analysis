"""Simple profiling utilities (typed shim).

This module provides a lightweight `ProfileContext` context manager and
`profile` decorator used across the codebase. The implementations are
no-ops but provide runtime-safe behavior and static typing for mypy.
"""
from __future__ import annotations

from typing import Any, Callable, ContextManager, TypeVar

T = TypeVar("T")


class ProfileContext(ContextManager[None]):
    def __init__(self, name: str | None = None) -> None:
        self.name = name

    def __enter__(self) -> None:
        return None

    def __exit__(
        self, exc_type: type | None, exc: BaseException | None, tb: Any
    ) -> bool | None:
        return None


def profile() -> Callable[[Callable[..., T]], Callable[..., T]]:
    def _decorator(func: Callable[..., T]) -> Callable[..., T]:
        def _wrapped(*args: Any, **kwargs: Any) -> T:
            return func(*args, **kwargs)

        return _wrapped

    return _decorator
