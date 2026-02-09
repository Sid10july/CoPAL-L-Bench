from src.callbacks.callback import Callback, CallbackArguments


class CostPrintCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self._printed_calls = 0
        self._step = 0

    @classmethod
    def is_unique(cls) -> bool:
        return True

    def _get_cost_tracker(self, callback_args: CallbackArguments):
        agent = callback_args.agent

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

    def on_task_complete(self, callback_args: CallbackArguments) -> None:
        cost_tracker = self._get_cost_tracker(callback_args)
        if cost_tracker is None:
            print("[CostPrintCallback] No cost tracker found on agent/LLM.")
            return

        summary = cost_tracker.summary()
        print("\n[CostPrintCallback] === Cost Metrics ===")
        print(f"Total Input Tokens:   {summary['total_prompt_tokens']}")
        print(f"Total Output Tokens:  {summary['total_completion_tokens']}")
        print(f"Total Cost ($):       {summary['total_cost_usd']:.6f}")
        print("[CostPrintCallback] =====================\n")
