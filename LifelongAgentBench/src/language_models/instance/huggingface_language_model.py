import os, json, datetime
import torch
from typing import Any, Mapping, Sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.language_models.language_model import LanguageModel
from src.typings import (
    Role,
    ChatHistoryItem,
    LanguageModelContextLimitException,
    LanguageModelOutOfMemoryException,
    ChatHistory,
)


class HuggingfaceLanguageModel(LanguageModel):
    def __init__(
        self,
        model_name_or_path: str,
        role_dict: Mapping[str, str],
        dtype: torch.dtype | str = torch.bfloat16,
        device_map: str | Mapping[str, Any] = "auto",
    ):
        super().__init__(role_dict)
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.io_log_path = os.environ.get("LLM_IO_LOG")

        if isinstance(device_map, str) and device_map == "niuload":
            device_map = "auto"
        if (
            isinstance(device_map, str)
            and device_map == "auto"
            and not torch.cuda.is_available()
        ):
            device_map = (
                "mps"
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                else "cpu"
            )
        if isinstance(device_map, str) and device_map == "mps":
            load_dtype = torch.float16
        else:
            load_dtype = dtype if isinstance(dtype, torch.dtype) else torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            torch_dtype=load_dtype,
            low_cpu_mem_usage=True,
        )

        if self.io_log_path:
            os.makedirs(os.path.dirname(self.io_log_path), exist_ok=True)
            self._log_io(
                {
                    "t": datetime.datetime.now().isoformat(),
                    "phase": "init",
                    "model": model_name_or_path,
                    "device_map": (
                        device_map if isinstance(device_map, str) else "custom"
                    ),
                    "dtype": str(load_dtype),
                }
            )

    def _log_io(self, obj: Mapping[str, Any]) -> None:
        if not self.io_log_path:
            return
        try:
            with open(self.io_log_path, "a") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _convert_message_list_to_model_input_dict(
        self, batch_message_list: Sequence[Sequence[Mapping[str, str]]]
    ) -> Mapping[str, torch.Tensor]:
        batch_input_ids: torch.Tensor = self.tokenizer.apply_chat_template(
            batch_message_list,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)
        no_padding_mask: torch.Tensor = batch_input_ids != self.tokenizer.pad_token_id
        _, first_no_padding_token_index = (
            (no_padding_mask.cumsum(-1) == 1) & no_padding_mask
        ).max(-1)
        row_repeat: torch.Tensor = (
            torch.arange(batch_input_ids.shape[1])
            .unsqueeze(0)
            .repeat(batch_input_ids.shape[0], 1)
        ).to(self.model.device)
        column_repeat: torch.Tensor = first_no_padding_token_index.unsqueeze(1).repeat(
            1, batch_input_ids.shape[1]
        )
        batch_attention_mask_ne: torch.Tensor = row_repeat < column_repeat
        batch_attention_mask: torch.Tensor = ~batch_attention_mask_ne
        return {
            "batch_input_ids": batch_input_ids,
            "batch_attention_mask": batch_attention_mask,
        }

    @staticmethod
    def _is_any_gpu_memory_high() -> bool:
        if not torch.cuda.is_available():
            return False
        for i in list(range(torch.cuda.device_count())):
            device = torch.device(f"cuda:{i}")
            free, total = torch.cuda.mem_get_info(device)
            if free / total < 0.1:
                return True
        return False

    def _inference(
        self,
        batch_chat_history: Sequence[ChatHistory],
        inference_config_dict: Mapping[str, Any],
        system_prompt: str,
    ) -> Sequence[ChatHistoryItem]:
        original_tokenizer_padding_side = self.tokenizer.padding_side
        original_tokenizer_pad_token = self.tokenizer.pad_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if len(system_prompt) > 0:
            message_list_prefix = [{"role": "system", "content": system_prompt}]
        else:
            message_list_prefix = []
        batch_message_list: Sequence[Sequence[Mapping[str, str]]] = [
            message_list_prefix
            + self._convert_chat_history_to_message_list(chat_history)
            for chat_history in batch_chat_history
        ]

        self._log_io(
            {
                "t": datetime.datetime.now().isoformat(),
                "phase": "input",
                "messages": batch_message_list,
            }
        )

        model_input_dict: Mapping[str, torch.Tensor] = (
            self._convert_message_list_to_model_input_dict(batch_message_list)
        )
        batch_input_ids, batch_attention_mask = (
            model_input_dict["batch_input_ids"],
            model_input_dict["batch_attention_mask"],
        )
        del model_input_dict
        if batch_input_ids.shape[-1] >= self.model.config.max_position_embeddings:
            raise LanguageModelContextLimitException(
                f"Input length {batch_input_ids.shape[-1]} exceeds the model's max_position_embeddings "
                f"{self.model.config.max_position_embeddings}."
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        try:
            output_tensor: torch.Tensor = self.model.generate(
                batch_input_ids,
                attention_mask=batch_attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                **inference_config_dict,
            )
        except Exception as e:
            self._log_io(
                {
                    "t": datetime.datetime.now().isoformat(),
                    "phase": "exception",
                    "error": str(e),
                }
            )
            oom = False
            if torch.cuda.is_available() and isinstance(e, torch.cuda.OutOfMemoryError):
                oom = True
            if "MPS" in str(e) and "out of memory" in str(e):
                oom = True
            if oom or HuggingfaceLanguageModel._is_any_gpu_memory_high():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise LanguageModelOutOfMemoryException(str(e)) from e
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        # --- Cost tracking (token counts) ---
        cost_tracker = getattr(self, "cost_tracker", None)
        if cost_tracker is not None:
            try:
                model_name = getattr(self, "model_name_or_path", None) or getattr(
                    self.model, "name_or_path", "unknown-model"
                )
                prompt_token_counts = batch_attention_mask.sum(-1).tolist()
                total_len = int(output_tensor.shape[1])
                for prompt_tokens in prompt_token_counts:
                    completion_tokens = max(0, total_len - int(prompt_tokens))
                    cost_tracker.log_call(
                        str(model_name), int(prompt_tokens), int(completion_tokens)
                    )
            except Exception:
                pass

        output_str_list: Sequence[str] = self.tokenizer.batch_decode(
            output_tensor[:, batch_input_ids.shape[1] :], skip_special_tokens=True
        )

        self._log_io(
            {
                "t": datetime.datetime.now().isoformat(),
                "phase": "output",
                "texts": list(output_str_list),
            }
        )

        output_list: Sequence[ChatHistoryItem] = [
            ChatHistoryItem(role=Role.AGENT, content=output_str)
            for output_str in output_str_list
        ]
        self.tokenizer.padding_side = original_tokenizer_padding_side
        self.tokenizer.pad_token = original_tokenizer_pad_token
        return output_list
