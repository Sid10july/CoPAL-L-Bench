from src.callbacks.callback import Callback, CallbackArguments
from src.typings import SessionEvaluationOutcome


class CostPrintCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self._printed_calls = 0
        self._step = 0
        self._session_start_call_index: dict[int, int] = {}

    @classmethod
    def is_unique(cls) -> bool:
        return True

    def _get_cost_tracker(self, callback_args: CallbackArguments):
        agent = callback_args.session_context.agent

        # First try agent.cost_tracker
        cost_tracker = getattr(agent, "cost_tracker", None)

        # Fallback: look on the wrapped LLM/model
        if cost_tracker is None:
            llm_obj = getattr(agent, "llm", None) or getattr(agent, "model", None)
            if llm_obj is not None:
                cost_tracker = getattr(llm_obj, "cost_tracker", None)

        return cost_tracker

    def _print_new_calls(self, cost_tracker) -> None:
        calls = getattr(cost_tracker, "calls", None) or []
        if self._printed_calls >= len(calls):
            return

        # Print any new calls since last time
        for call in calls[self._printed_calls :]:
            self._step += 1
            total_cost = sum(c.total_cost_usd for c in calls[: self._step])
            print(f"\n[LLM-COST] Model: {call.model}")
            print(f"[LLM-COST] Step {self._step}:")
            print(f"  Input tokens: {call.prompt_tokens}")
            print(f"  Output tokens: {call.completion_tokens}")
            print(f"  Cost this call: ${call.total_cost_usd:.6f}")
            print(f"  Running total cost: ${total_cost:.6f}")

        self._printed_calls = len(calls)

    def on_agent_inference(self, callback_args: CallbackArguments) -> None:
        cost_tracker = self._get_cost_tracker(callback_args)
        if cost_tracker is None:
            return
        self._print_new_calls(cost_tracker)

    def on_session_create(self, callback_args: CallbackArguments) -> None:
        cost_tracker = self._get_cost_tracker(callback_args)
        if cost_tracker is None:
            return
        calls = getattr(cost_tracker, "calls", None) or []
        self._session_start_call_index[callback_args.current_session.sample_index] = (
            len(calls)
        )

    def on_task_complete(self, callback_args: CallbackArguments) -> None:
        cost_tracker = self._get_cost_tracker(callback_args)
        if cost_tracker is None:
            print("[CostPrintCallback] No cost tracker found on agent/LLM.")
            return

        # Per-sample cost (paper definition)
        calls = getattr(cost_tracker, "calls", None) or []
        start_idx = self._session_start_call_index.get(
            callback_args.current_session.sample_index, None
        )
        if start_idx is not None and start_idx <= len(calls):
            sample_calls = calls[start_idx:]
            sample_in = sum(c.prompt_tokens for c in sample_calls)
            sample_out = sum(c.completion_tokens for c in sample_calls)
            sample_cost = sum(c.total_cost_usd for c in sample_calls)
            outcome = callback_args.current_session.evaluation_record.outcome
            pass_rate = 1.0 if outcome == SessionEvaluationOutcome.CORRECT else 0.0
            cost_of_pass = sample_cost / pass_rate if pass_rate > 0 else float("inf")

            detail = callback_args.current_session.evaluation_record.detail_dict or {}
            detail.update(
                {
                    "cost_input_tokens": sample_in,
                    "cost_output_tokens": sample_out,
                    "cost_usd": sample_cost,
                    "cost_of_pass": cost_of_pass,
                    "pass_rate": pass_rate,
                }
            )
            callback_args.current_session.evaluation_record.detail_dict = detail

            print("\n[CostPrintCallback] === Sample Cost ===")
            print(f"Sample Input Tokens:  {sample_in}")
            print(f"Sample Output Tokens: {sample_out}")
            print(f"Sample Cost ($):      {sample_cost:.6f}")
            print(f"Sample Cost-of-Pass:  {cost_of_pass}")
            print("[CostPrintCallback] ===================\n")

        summary = cost_tracker.summary()
        print("\n[CostPrintCallback] === Cost Metrics ===")
        print(f"Total Input Tokens:   {summary['total_prompt_tokens']}")
        print(f"Total Output Tokens:  {summary['total_completion_tokens']}")
        print(f"Total Cost ($):       {summary['total_cost_usd']:.6f}")
        print("[CostPrintCallback] =====================\n")
