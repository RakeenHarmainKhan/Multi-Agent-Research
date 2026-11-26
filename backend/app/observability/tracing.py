"""Langfuse traces without queries, responses, source content or exception messages."""
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from httpx import HTTPError
from langfuse import Langfuse

from app.core.config import settings

logger = logging.getLogger(__name__)
_client: Langfuse | None = None
_current: ContextVar[dict[str, Any] | None] = ContextVar("trace_metadata", default=None)
_SDK_ERRORS = (HTTPError, OSError, RuntimeError, ValueError)


def start_tracing() -> None:
    global _client
    if not settings.LANGFUSE_ENABLED:
        return
    if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
        raise ValueError("LANGFUSE_ENABLED requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
    try:
        _client = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            base_url=settings.LANGFUSE_BASE_URL,
        )
    except _SDK_ERRORS as exc:
        logger.warning("Langfuse initialization failed (%s)", type(exc).__name__)


def stop_tracing() -> None:
    global _client
    client, _client = _client, None
    if client is not None:
        try:
            client.shutdown()
        except _SDK_ERRORS as exc:
            logger.warning("Langfuse shutdown failed (%s)", type(exc).__name__)


def record(**values: Any) -> None:
    metadata = _current.get()
    if metadata is not None:
        metadata.update(values)


def record_error(exc: BaseException) -> None:
    """Include handled failures without exporting their potentially sensitive messages."""
    record(error_type=type(exc).__name__, outcome="error")


@contextmanager
def trace_span(name: str, *, generation: bool = False) -> Iterator[dict[str, Any]]:
    metadata: dict[str, Any] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    if _client is None:
        yield metadata
        return
    try:
        manager = _client.start_as_current_observation(
            name=name, as_type="generation" if generation else "span",
            model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT if generation else None,
        )
        observation = manager.__enter__()
    except _SDK_ERRORS as exc:
        logger.warning("Langfuse span start failed (%s)", type(exc).__name__)
        yield metadata
        return

    parent = _current.get()
    token = _current.set(metadata)
    started = time.perf_counter()
    try:
        yield metadata
    except BaseException as exc:
        record_error(exc)
        raise
    finally:
        _current.reset(token)
        metadata["total_tokens"] = metadata["input_tokens"] + metadata["output_tokens"]
        metadata["latency_seconds"] = round(time.perf_counter() - started, 3)
        if parent is not None:
            for key in ("input_tokens", "output_tokens"):
                parent[key] += metadata[key]
            if metadata.get("error_type") or metadata.get("child_error"):
                parent["child_error"] = True
        try:
            level = "ERROR" if metadata.get("error_type") else "WARNING" if metadata.get("child_error") else "DEFAULT"
            observation.update(metadata=metadata, level=level)
            if generation:
                observation.update(usage_details={
                    "input": metadata["input_tokens"],
                    "output": metadata["output_tokens"],
                    "total": metadata["total_tokens"],
                })
        except _SDK_ERRORS as exc:
            logger.warning("Langfuse span update failed (%s)", type(exc).__name__)
        finally:
            try:
                # Passing the application exception would export its message and stack.
                manager.__exit__(None, None, None)
            except _SDK_ERRORS as exc:
                logger.warning("Langfuse span end failed (%s)", type(exc).__name__)
