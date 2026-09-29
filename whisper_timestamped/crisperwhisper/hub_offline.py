"""Offline-aware HuggingFace Hub helpers for CrisperWhisper.

When the Hub is unreachable but weights are already in the local cache,
``call_with_local_files_fallback`` retries with ``local_files_only=True``
instead of aborting.  Honors ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_OFFLINE_ENV_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")

_CONNECTIVITY_MARKERS = (
    "temporary failure in name resolution",
    "name or service not known",
    "nodename nor servname provided",
    "failed to resolve",
    "getaddrinfo failed",
    "connection refused",
    "connection reset",
    "connection aborted",
    "network is unreachable",
    "no route to host",
    "timed out",
    "timeout",
    "cannot send a request, as the client has been closed",
    "offline mode is enabled",
    "we couldn't connect to",
    "max retries exceeded",
)


def hub_force_offline() -> bool:
    """True when HuggingFace / Transformers offline env vars are set."""
    for name in _OFFLINE_ENV_VARS:
        value = os.environ.get(name, "").strip().lower()
        if value in ("1", "true", "yes", "on"):
            return True
    return False


def is_hub_connectivity_error(exc: BaseException) -> bool:
    """True if ``exc`` looks like a Hub / DNS / connection failure."""
    parts: list[str] = [str(exc)]
    cause = getattr(exc, "__cause__", None)
    if cause is not None:
        parts.append(str(cause))
    context = getattr(exc, "__context__", None)
    if context is not None and context is not cause:
        parts.append(str(context))
    message = " ".join(parts).lower()
    if any(marker in message for marker in _CONNECTIVITY_MARKERS):
        return True
    # Common network exception types (stdlib + httpx/requests when present).
    type_name = type(exc).__name__
    if type_name in (
        "ConnectionError",
        "ConnectTimeout",
        "ReadTimeout",
        "TimeoutError",
        "OSError",
        "ProxyError",
        "LocalEntryNotFoundError",
    ):
        # OSError is broad; only treat as connectivity when message matches
        # or errno is typical network failure.
        if type_name == "OSError":
            errno = getattr(exc, "errno", None)
            # EAI_AGAIN (-3), ENETUNREACH, ECONNREFUSED, ETIMEDOUT, etc.
            if errno in (-3, -2, 101, 111, 110, 10051, 10060, 10061):
                return True
            return any(marker in message for marker in _CONNECTIVITY_MARKERS)
        return True
    return False


def call_with_local_files_fallback(
    fn: Callable[..., T],
    *args,
    local_files_only_kw: str = "local_files_only",
    **kwargs,
) -> T:
    """Call ``fn``; on Hub connectivity failure retry with local files only.

    If ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE`` is set, skips the
    network attempt and calls with ``local_files_only=True`` immediately.
    """
    if hub_force_offline():
        kwargs[local_files_only_kw] = True
        return fn(*args, **kwargs)

    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        if kwargs.get(local_files_only_kw) or not is_hub_connectivity_error(exc):
            raise
        logger.warning(
            "HuggingFace Hub unreachable (%s); retrying with local_files_only=True",
            exc,
        )
        kwargs[local_files_only_kw] = True
        return fn(*args, **kwargs)
