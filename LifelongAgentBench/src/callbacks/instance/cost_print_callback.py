from src.callbacks.callback import Callback, CallbackArguments


class CostPrintCallback(Callback):
    @classmethod
    def is_unique(cls) -> bool:
        return True

    def on_task_complete(self, callback_args: CallbackArguments) -> None:
        session = callback_args.current_session
        cost_tracker = getattr(session, "cost", None)

        if cost_tracker is None:
            print("[CostPrintCallback] No cost tracker found.")
            return

        summary = cost_tracker.summary()
        print("\n=== Cost Metrics ===")
        print(f"Total Input Tokens:   {summary['input_tokens']}")
        print(f"Total Output Tokens:  {summary['output_tokens']}")
        print(f"Total Cost ($):       {summary['total_cost']}")
        print("====================\n")
