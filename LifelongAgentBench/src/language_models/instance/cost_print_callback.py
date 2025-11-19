from src.callbacks.callback import Callback, CallbackArguments


class CostPrintCallback(Callback):

    @classmethod
    def is_unique(cls) -> bool:
        return True

    def on_task_complete(self, callback_args: CallbackArguments) -> None:
        agent = callback_args.agent

        # LLM object (CountingLLM wrapper) is usually on agent.llm or agent.model
        llm = getattr(agent, "llm", None) or getattr(agent, "model", None)
        if llm is None:
            print("[CostPrintCallback] No LLM found on agent.")
            return

        cost_tracker = getattr(llm, "cost_tracker", None)
        if cost_tracker is None:
            print("[CostPrintCallback] No cost tracker found on LLM.")
            return

        summary = cost_tracker.summary()

        print("\n=== Cost Metrics (CostPrintCallback) ===")
        print(f"Total Input Tokens:   {summary.get('total_input_tokens')}")
        print(f"Total Output Tokens:  {summary.get('total_output_tokens')}")
        print(f"Total Cost ($):       {summary.get('total_cost')}")
        print("========================================\n")
