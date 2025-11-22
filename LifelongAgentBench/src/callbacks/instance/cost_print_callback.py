from src.callbacks.callback import Callback, CallbackArguments


class CostPrintCallback(Callback):

    @classmethod
    def is_unique(cls) -> bool:
        return True

    def on_task_complete(self, callback_args: CallbackArguments) -> None:
        agent = callback_args.agent

        # First try agent.cost_tracker
        cost_tracker = getattr(agent, "cost_tracker", None)

        # Fallback: look on the wrapped LLM/model
        if cost_tracker is None:
            llm_obj = getattr(agent, "llm", None) or getattr(agent, "model", None)
            if llm_obj is not None:
                cost_tracker = getattr(llm_obj, "cost_tracker", None)

        if cost_tracker is None:
            print("[CostPrintCallback] No cost tracker found on agent/LLM.")
            return

        summary = cost_tracker.summary()
        print("\n[CostPrintCallback] === Cost Metrics ===")
        print(f"Total Input Tokens:   {summary['input_tokens']}")
        print(f"Total Output Tokens:  {summary['output_tokens']}")
        print(f"Total Cost ($):       {summary['total_cost']}")
        print("[CostPrintCallback] =====================\n")
