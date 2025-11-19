# src/metrics/llm_counting_wrapper.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import inspect
from .cost_tracker import CostTracker


class CountingLLM:
    """
    Wraps an existing LLM object and logs prompt/completion token counts per call.

    It works in two ways:
      - If the underlying object exposes `last_input_token_count` and
        `last_output_token_count` after a call (like the OAgents classes),
        we use them directly.
      - Otherwise, if a `tokenizer` is available, we compute counts.
    """

    def __init__(
        self,
        base_llm: Any,
        model_name: str,
        cost_tracker: CostTracker,
        tokenizer: Optional[Any] = None,
    ):
        self._llm = base_llm
        self._model_name = model_name
        self._tracker = cost_tracker
        self._tok = tokenizer

    def __getattr__(self, item):
        # forward everything else to underlying llm
        return getattr(self._llm, item)

    def __call__(self, messages: List[Dict[str, str]], **kwargs):
        # Pre-count prompt tokens if we can
        pre_prompt_tokens = None
        if self._tok is not None:
            try:
                t = self._tok.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    return_dict=True,
                    add_generation_prompt=False,
                )
                pre_prompt_tokens = int(t["input_ids"].shape[1])
            except Exception:
                pre_prompt_tokens = None

        # Call the real LLM
        result = self._llm(messages, **kwargs)

        # Prefer counts exposed by the underlying class
        prompt_tokens = getattr(self._llm, "last_input_token_count", None)
        completion_tokens = getattr(self._llm, "last_output_token_count", None)

        # Fallback to what we computed
        if prompt_tokens is None and pre_prompt_tokens is not None:
            prompt_tokens = pre_prompt_tokens
        if completion_tokens is None:
            # try to infer from raw generate output (Transformers)
            raw = getattr(result, "raw", None)
            try:
                out = raw.get("out") if isinstance(raw, dict) else None
                if out is not None and prompt_tokens is not None:
                    # out is a tensor [1, total_len]; completion starts after prompt
                    completion_tokens = int(out[0].shape[0] - prompt_tokens)
            except Exception:
                pass

        # Final defaults
        prompt_tokens = int(prompt_tokens or 0)
        completion_tokens = int(completion_tokens or 0)

        # Log to tracker
        self._tracker.log_call(self._model_name, prompt_tokens, completion_tokens)
        return result
