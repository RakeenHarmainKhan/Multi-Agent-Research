"""Per-run LLM token accounting."""
from threading import Lock
from typing import Any, Dict, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from app.core.config import settings


class TokenUsageHandler(BaseCallbackHandler):
    """
    Sums token usage over every LLM call in one graph run.

    Unlike LangChain's UsageMetadataCallbackHandler, this does not require the
    provider to return a `model_name`, so it works with any chat model that
    reports `usage_metadata`.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = Lock()
        self.input_tokens = 0
        self.output_tokens = 0
        self.llm_calls = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                with self._lock:
                    self.input_tokens += usage.get("input_tokens", 0)
                    self.output_tokens += usage.get("output_tokens", 0)
        with self._lock:
            self.llm_calls += 1

    def summary(self) -> Dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost": estimate_cost(self.input_tokens, self.output_tokens),
        }


def estimate_cost(input_tokens: int, output_tokens: int) -> Optional[float]:
    """USD cost from configured per-1M-token prices, or None when pricing isn't set."""
    input_price = settings.LLM_INPUT_COST_PER_1M_TOKENS
    output_price = settings.LLM_OUTPUT_COST_PER_1M_TOKENS
    if input_price is None or output_price is None:
        return None
    return round(input_tokens / 1_000_000 * input_price + output_tokens / 1_000_000 * output_price, 6)
