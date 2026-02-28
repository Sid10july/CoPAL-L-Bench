import argparse
import json
import os
import yaml
import copy
import random
from enum import StrEnum
from typing import Any, Mapping, Sequence, Optional
import coredumpy  # type: ignore[import-untyped]

from src.utils import ConfigLoader, SingletonLogger
from src.typings import (
    AssignmentConfig,
    EnvironmentConfig,
    SampleStatus,
    LoggerConfig,
    ContinualAgentBenchException,
    Session,
    SampleIndex,
    PathConfig,
    GeneralInstanceFactory,
    SessionMetricCalculationPartial,
)
from src.tasks import Task, DatasetItem
from src.agents import Agent
from src.language_models import LanguageModel
from src.callbacks import (
    CallbackHandler,
    CallbackConstructor,
    Callback,
    CallbackRestorer,
    CallbackArguments,
)


class ConfigUtilityCaller(StrEnum):
    CLIENT = "client"
    SERVER = "server"
    CLIENT_SIDE_CONTROLLER = "client_side_controller"


class ConfigUtility:
    def __init__(
        self,
        assignment_config: AssignmentConfig,
        environment_config: EnvironmentConfig,
        path_config: PathConfig,
    ):
        self.assignment_config = assignment_config
        self.environment_config = environment_config
        self.path_config = path_config

    def preprocess(self) -> None:
        if self.environment_config.task_client:
            self.assignment_config.task = self.environment_config.task_client

    def construct(self) -> tuple[Task[DatasetItem], Agent, dict[str, Callback]]:
        # Maybe task will be Task or TaskClient, but it doesn't matter!
        task: Task[DatasetItem] = self.assignment_config.task.create()
        # region Construct language_model_dict
        # Here, We actually instantiate the language models.
        # After the code exit construct(), We will never get a chance to get the language model instance.
        # But I think this is good, since this improve the modularity and maintainability of the code.
        language_model_dict: Mapping[str, LanguageModel] = {
            key: value.create()
            for key, value in self.assignment_config.language_model_dict.items()
        }
        agent_instance_factory: GeneralInstanceFactory = self.assignment_config.agent
        if (
            language_model_name := agent_instance_factory.parameters.get(
                "language_model"
            )
        ) is not None:
            agent_instance_factory.parameters["language_model"] = language_model_dict[
                language_model_name
            ]
        # endregion
        agent: Agent = agent_instance_factory.create()
        callback_dict = CallbackConstructor.construct(
            self.assignment_config, task, agent, language_model_dict
        )
        return task, agent, callback_dict

    def validate(self, task: Task[DatasetItem], agent: Agent) -> None:
        sample_index_list = task.get_sample_index_list()
        if self.assignment_config.sample_order == "default":
            return
        # Normalize configured indices to the exact key type used by the task dataset.
        sample_index_set = set(sample_index_list)
        normalized_sample_order = []
        for selected_sample_index in self.assignment_config.sample_order:
            if selected_sample_index in sample_index_set:
                normalized_sample_order.append(selected_sample_index)
                continue
            selected_as_str = str(selected_sample_index)
            if selected_as_str in sample_index_set:
                normalized_sample_order.append(selected_as_str)
                continue
            try:
                selected_as_int = int(selected_sample_index)
            except (TypeError, ValueError):
                selected_as_int = None
            if selected_as_int is not None and selected_as_int in sample_index_set:
                normalized_sample_order.append(selected_as_int)
                continue
            assert selected_sample_index in sample_index_set
        self.assignment_config.sample_order = normalized_sample_order

    def postprocess(self, task: Task[DatasetItem], agent: Agent) -> None:
        if self.assignment_config.sample_order == "default":
            self.assignment_config.sample_order = task.get_sample_index_list()

    def remove_redundant_args(self, raw_config: dict[str, Any]) -> dict[str, Any]:
        # Maybe use `if raw_config["environment_config"]["use_task_client_flag"]` is better, but I use the following
        # condition avoid using dict key directly.
        if not self.environment_config.task_client:
            # If the config file is used to restore the previous incomplete assignment, the `task_client` will be None.
            # Using del to remove the key-value pair will cause an error in this case.
            for key in list(raw_config["environment_config"]):
                if key != "use_task_client_flag":
                    del raw_config["environment_config"][key]
        redundant_key_buffer: set[tuple[str, str]] = set()
        for key in raw_config["task_dict"]:
            if key != raw_config["assignment_config"]["task"]:
                redundant_key_buffer.add(("task_dict", key))
        for key in raw_config["agent_dict"]:
            if key != raw_config["assignment_config"]["agent"]["name"]:
                redundant_key_buffer.add(("agent_dict", key))
        assignment_language_model_name_list: Sequence[str] = [
            language_model_info_dict["name"]
            for language_model_info_dict in raw_config["assignment_config"][
                "language_model_list"
            ]
        ]
        for key in raw_config["language_model_dict"]:
            if key not in assignment_language_model_name_list:
                redundant_key_buffer.add(("language_model_dict", key))
        assignment_callback_name_list: Sequence[str] = [
            callback_info_dict["name"]
            for callback_info_dict in raw_config["assignment_config"][
                "callback_dict"
            ].values()
        ]
        for key in raw_config["callback_dict"]:
            if key not in assignment_callback_name_list:
                redundant_key_buffer.add(("callback_dict", key))
        for info_tuple in redundant_key_buffer:
            del raw_config[info_tuple[0]][info_tuple[1]]
        return raw_config

    @staticmethod
    def _get_custom_instance_info_dict(
        default_instance_info_dict: Mapping[str, Any],
        custom_instance_info_dict: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        module: str = default_instance_info_dict["module"]
        # `... or {}` is used to ensure that both default_parameters and custom_parameters are dict.
        default_parameters: dict[str, Any] = copy.deepcopy(
            default_instance_info_dict.get("parameters") or {}
        )
        custom_parameters = custom_instance_info_dict.get("custom_parameters") or {}
        for parameter_name, custom_parameter_value in custom_parameters.items():
            assert parameter_name in default_parameters  # Do not remove this assertion.
            # Overwrite the default parameter value with the custom parameter value.
            default_parameters[parameter_name] = custom_parameter_value
        return {
            "module": module,
            "parameters": default_parameters,
        }

    @staticmethod
    def read_raw_config(
        raw_config: Mapping[str, Any], caller: ConfigUtilityCaller
    ) -> tuple[AssignmentConfig, EnvironmentConfig, LoggerConfig, PathConfig]:
        raw_config = copy.deepcopy(raw_config)  # Avoid modifying the original config.
        # region Convert raw_config into assignment_config
        # region Construct assignment_language_model_dict
        assignment_language_model_list: Sequence[Mapping[str, Any]] = raw_config[
            "assignment_config"
        ]["language_model_list"]
        assignment_language_model_dict: dict[str, Any] = {}
        for language_model_info_dict in assignment_language_model_list:
            language_model_name = language_model_info_dict["name"]
            default_language_model_info_dict = raw_config["language_model_dict"][
                language_model_name
            ]
            assert language_model_name not in assignment_language_model_dict
            assignment_language_model_dict[language_model_name] = (
                ConfigUtility._get_custom_instance_info_dict(
                    default_language_model_info_dict, language_model_info_dict
                )
            )
        # endregion
        # region Construct assignment_agent
        custom_agent_info_dict = raw_config["assignment_config"]["agent"]
        default_agent_info_dict = raw_config["agent_dict"][
            custom_agent_info_dict["name"]
        ]
        assignment_agent_info_dict = ConfigUtility._get_custom_instance_info_dict(
            default_agent_info_dict, custom_agent_info_dict
        )
        assignment_agent = GeneralInstanceFactory.model_validate(
            assignment_agent_info_dict
        )
        if (
            language_model_name := assignment_agent.parameters.get("language_model")
        ) is not None:
            # Do not replace the language_model in the parameters with the GeneralInstanceFactory instance.
            assert language_model_name in assignment_language_model_dict
        # endregion
        # region Construct assignment_callback_dict
        assignment_callback_dict: dict[str, Any] = raw_config["assignment_config"][
            "callback_dict"
        ]

        # DEBUG PRINTS — add these 4 lines:
        print(
            "Assignment callback names:",
            [cb["name"] for cb in assignment_callback_dict.values()],
        )
        print("Registry callback_dict keys:", list(raw_config["callback_dict"].keys()))
        # END DEBUG

        for callback_key, callback_info_dict in assignment_callback_dict.items():
            default_callback_info_dict = raw_config["callback_dict"][
                callback_info_dict["name"]
            ]
            assignment_callback_dict[callback_key] = (
                ConfigUtility._get_custom_instance_info_dict(
                    default_callback_info_dict, callback_info_dict
                )
            )
        # endregion
        assignment_config = AssignmentConfig(
            task=raw_config["task_dict"][raw_config["assignment_config"]["task"]],
            agent=assignment_agent,
            language_model_dict=assignment_language_model_dict,
            output_dir=raw_config["assignment_config"]["output_dir"],
            sample_order=raw_config["assignment_config"]["sample_order"],
            callback_dict=assignment_callback_dict,
        )
        # endregion
        # region Convert raw_config into environment_config
        if raw_config["environment_config"]["use_task_client_flag"]:
            environment_config = EnvironmentConfig(
                task_client=raw_config["environment_config"]["task_client"],
                chat_history_item_factory_client=raw_config["environment_config"][
                    "chat_history_item_factory_client"
                ],
                server_side_controller_address=raw_config["environment_config"][
                    "server_side_controller_address"
                ],
                interpreter_path=raw_config["environment_config"]["interpreter_path"],
            )
        else:
            environment_config = EnvironmentConfig(
                task_client=None,
                chat_history_item_factory_client=None,
                server_side_controller_address=None,
                interpreter_path=None,
            )
        # endregion
        # region Convert raw_config into logger_config
        if raw_config["logger_config"]["log_file_path"] == "default":
            if raw_config["environment_config"]["use_task_client_flag"]:
                match caller:
                    case ConfigUtilityCaller.CLIENT:
                        log_file_path = os.path.join(
                            assignment_config.output_dir, "singleton_logger_client.log"
                        )
                    case ConfigUtilityCaller.SERVER:
                        log_file_path = os.path.join(
                            assignment_config.output_dir, "singleton_logger_server.log"
                        )
                    case ConfigUtilityCaller.CLIENT_SIDE_CONTROLLER:
                        log_file_path = (
                            "./outputs/singleton_logger_client_side_controller.log"
                        )
                    case _:
                        raise NotImplementedError()
            else:
                log_file_path = os.path.join(
                    assignment_config.output_dir, "singleton_logger.log"
                )
        else:
            log_file_path = raw_config["logger_config"]["log_file_path"]
        logger_config = LoggerConfig(
            level=raw_config["logger_config"]["level"],
            log_file_path=log_file_path,
            logger_name=raw_config["logger_config"]["logger_name"],
        )
        # endregion
        # region Construct path_config from assignment_config
        path_config = PathConfig(
            exception_record_file_path=os.path.join(
                assignment_config.output_dir, "exception.txt"
            ),
            config_output_path=os.path.join(
                assignment_config.output_dir, "config.yaml"
            ),
            session_list_output_path=os.path.join(
                assignment_config.output_dir, "runs.json"
            ),
            metric_output_path=os.path.join(
                assignment_config.output_dir, "metric.json"
            ),
            coredumpy_output_dir=os.path.join(
                assignment_config.output_dir, "coredumpy"
            ),
        )
        # endregion
        return assignment_config, environment_config, logger_config, path_config

    @staticmethod
    def is_raw_config_equal(
        raw_config_1: dict[str, Any], raw_config_2: dict[str, Any]
    ) -> bool:
        raw_config_1 = copy.deepcopy(raw_config_1)
        raw_config_2 = copy.deepcopy(raw_config_2)
        output_dir_1 = raw_config_1["assignment_config"].pop("output_dir")
        output_dir_2 = raw_config_2["assignment_config"].pop("output_dir")
        return raw_config_1 == raw_config_2 and AssignmentConfig.is_output_dir_equal(
            output_dir_1, output_dir_2
        )


def main() -> None:
    # region Prepare variables
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("LAB_RUN_SEED", "42")),
    )
    parser.add_argument(
        "--enable_dbbench_bandit",
        action="store_true",
        default=os.environ.get("DBBENCH_BANDIT_ENABLE", "0") == "1",
    )
    parser.add_argument(
        "--bandit_lambda",
        type=float,
        default=float(os.environ.get("DBBENCH_BANDIT_LAMBDA", "0.3")),
    )
    parser.add_argument(
        "--bandit_alpha",
        type=float,
        default=float(os.environ.get("DBBENCH_BANDIT_ALPHA", "0.5")),
    )
    parser.add_argument(
        "--bandit_budget_target",
        type=float,
        default=float(os.environ.get("DBBENCH_BANDIT_BUDGET_TARGET", "0.00009")),
    )
    parser.add_argument(
        "--bandit_lambda_lr",
        type=float,
        default=float(os.environ.get("DBBENCH_BANDIT_LAMBDA_LR", "0.05")),
    )
    parser.add_argument(
        "--bandit_adaptive_lambda",
        action="store_true",
        default=os.environ.get("DBBENCH_BANDIT_ADAPTIVE_LAMBDA", "1") == "1",
    )
    parser.add_argument(
        "--bandit_two_head",
        action="store_true",
        default=os.environ.get("DBBENCH_BANDIT_TWO_HEAD", "1") == "1",
    )
    parser.add_argument(
        "--bandit_policy",
        type=str,
        default=os.environ.get("DBBENCH_BANDIT_POLICY", "utility"),
        choices=["cheapest_safe", "utility"],
    )
    parser.add_argument(
        "--bandit_correctness_threshold",
        type=float,
        default=float(os.environ.get("DBBENCH_BANDIT_CORRECTNESS_THRESHOLD", "0.70")),
    )
    parser.add_argument(
        "--bandit_adaptive_threshold",
        action="store_true",
        default=os.environ.get("DBBENCH_BANDIT_ADAPTIVE_THRESHOLD", "1") == "1",
    )
    parser.add_argument(
        "--bandit_target_accuracy",
        type=float,
        default=float(os.environ.get("DBBENCH_BANDIT_TARGET_ACCURACY", "0.68")),
    )
    parser.add_argument(
        "--bandit_accuracy_tolerance",
        type=float,
        default=float(os.environ.get("DBBENCH_BANDIT_ACCURACY_TOL", "0.01")),
    )
    parser.add_argument(
        "--bandit_threshold_step",
        type=float,
        default=float(os.environ.get("DBBENCH_BANDIT_THRESHOLD_STEP", "0.005")),
    )
    parser.add_argument(
        "--bandit_cost_ema_decay",
        type=float,
        default=float(os.environ.get("DBBENCH_BANDIT_COST_EMA_DECAY", "0.9")),
    )
    parser.add_argument(
        "--bandit_rescue_enable",
        action="store_true",
        default=os.environ.get("DBBENCH_BANDIT_RESCUE_ENABLE", "1") == "1",
    )
    parser.add_argument(
        "--bandit_rescue_profile_name",
        type=str,
        default=os.environ.get("DBBENCH_BANDIT_RESCUE_PROFILE", "budget_512_r3"),
    )
    parser.add_argument(
        "--bandit_rescue_on_incorrect",
        action="store_true",
        default=os.environ.get("DBBENCH_BANDIT_RESCUE_ON_INCORRECT", "0") == "1",
    )
    parser.add_argument(
        "--bandit_rescue_on_task_limit",
        action="store_true",
        default=os.environ.get("DBBENCH_BANDIT_RESCUE_ON_TASK_LIMIT", "1") == "1",
    )
    parser.add_argument(
        "--bandit_rescue_on_incomplete",
        action="store_true",
        default=os.environ.get("DBBENCH_BANDIT_RESCUE_ON_INCOMPLETE", "0") == "1",
    )
    parser.add_argument(
        "--bandit_threshold_min",
        type=float,
        default=float(os.environ.get("DBBENCH_BANDIT_THRESHOLD_MIN", "0.55")),
    )
    parser.add_argument(
        "--bandit_threshold_max",
        type=float,
        default=float(os.environ.get("DBBENCH_BANDIT_THRESHOLD_MAX", "0.90")),
    )
    parser.add_argument(
        "--bandit_allow_max_arm",
        action="store_true",
        default=os.environ.get("DBBENCH_BANDIT_ALLOW_MAX_ARM", "0") == "1",
    )
    parser.add_argument(
        "--bandit_cost_guardrail_enable",
        action="store_true",
        default=os.environ.get("DBBENCH_BANDIT_COST_GUARDRAIL_ENABLE", "1") == "1",
    )
    parser.add_argument(
        "--bandit_cost_guardrail_band",
        type=float,
        default=float(os.environ.get("DBBENCH_BANDIT_COST_GUARDRAIL_BAND", "0.10")),
    )
    parser.add_argument(
        "--bandit_cost_guardrail_decay",
        type=float,
        default=float(os.environ.get("DBBENCH_BANDIT_COST_GUARDRAIL_DECAY", "0.95")),
    )
    args = parser.parse_args()
    random.seed(args.seed)
    try:
        import numpy as np

        np.random.seed(args.seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass
    raw_config = ConfigLoader().load_from(args.config_path)
    assignment_config, environment_config, logger_config, path_config = (
        ConfigUtility.read_raw_config(raw_config, ConfigUtilityCaller.CLIENT)
    )
    # Prepare the variable that will be used in the following procedures.
    config_utility = ConfigUtility(assignment_config, environment_config, path_config)
    # endregion
    # region write raw_config to disk
    cleaned_config = config_utility.remove_redundant_args(raw_config)
    config_output_path = path_config.config_output_path
    if os.path.exists(config_output_path):
        config_from_disk = yaml.safe_load(open(config_output_path, "r"))
        assert ConfigUtility.is_raw_config_equal(config_from_disk, cleaned_config)
        # The config file already exists, so we don't need to write it again.
    else:
        # Write the config file to the output directory.
        config_output_dir = os.path.dirname(config_output_path)
        if not os.path.exists(config_output_dir):
            os.makedirs(config_output_dir)
        yaml.dump(
            cleaned_config,
            open(config_output_path, "w"),
        )
    # endregion
    # region Initialize logger, Set coredumpy output dir
    logger = SingletonLogger.get_instance(logger_config)
    coredumpy.patch_except(directory=path_config.coredumpy_output_dir)
    # endregion
    # region Construct variable, valid config
    config_utility.preprocess()
    task, agent, callback_dict = config_utility.construct()
    #####
    from src.metrics.cost_tracker import CostTracker, CostRegistry
    from src.metrics.llm_counting_wrapper import CountingLLM

    # create a tracker for this run
    cost_tracker = CostTracker(
        # Optional: define prices here if you’re using a paid provider
        # CostRegistry({"meta-llama/Llama-3.1-8B-Instruct": {"in": 0.0, "out": 0.0}})
    )

    # Figure out model name + tokenizer on the agent/llm object
    llm_obj = getattr(agent, "llm", None) or getattr(agent, "model", None)
    model_name = None
    for attr in ("model_name_or_path", "model_id", "name", "model"):
        if hasattr(llm_obj, attr):
            model_name = getattr(llm_obj, attr)
            break
    model_name = str(model_name or "unknown-model")

    tokenizer = getattr(llm_obj, "tokenizer", None)

    # Wrap
    wrapped = CountingLLM(
        llm_obj, model_name=model_name, tokenizer=tokenizer, cost_tracker=cost_tracker
    )
    if hasattr(agent, "llm"):
        agent.llm = wrapped
    elif hasattr(agent, "model"):
        agent.model = wrapped
    else:
        # Worst case: the agent keeps a callable function attribute — try common names:
        setattr(agent, "llm", wrapped)
    # >>> ADD THIS so callbacks can find the tracker <<<
    setattr(agent, "cost_tracker", cost_tracker)
    # Also attach to language_model (used by LanguageModelAgent)
    lm_obj = getattr(agent, "_language_model", None)
    if lm_obj is not None:
        setattr(lm_obj, "cost_tracker", cost_tracker)
    #####
    config_utility.postprocess(task, agent)
    config_utility.validate(task, agent)
    ContinualAgentBenchException.set_record_file(path_config.exception_record_file_path)
    # endregion
    # region Determine whether to start a new assignment or restore the previous incomplete assignment, based on
    # whether the config file exists.
    session_list_output_path = path_config.session_list_output_path
    assert isinstance(assignment_config.sample_order, list)
    session_list: list[Session]
    unfinished_sample_order: list[SampleIndex]
    if os.path.exists(session_list_output_path):
        # At least one session exists, so we restore the previous incomplete assignment.
        session_list = [
            Session.model_validate(session_info_dict)
            for session_info_dict in json.load(open(session_list_output_path, "r"))
        ]
        unfinished_sample_order = [
            sample_index
            for sample_index in assignment_config.sample_order
            if all(session.sample_index != sample_index for session in session_list)
        ]
        # Previous session may change the state of the callback, restore it here.
        CallbackRestorer.restore(callback_dict)
    else:
        # Start a new assignment.
        session_list = []
        unfinished_sample_order = assignment_config.sample_order
    callback_handler = CallbackHandler(callback_dict)
    # endregion
    # region Run experiment
    bandit_enabled = (
        bool(args.enable_dbbench_bandit) and str(task.task_name) == "db_bench"
    )
    dbbench_bandit = None
    dbbench_feature_adapter = None
    dbbench_budget_primitives = []
    bandit_state_by_sample: dict[int, dict[str, Any]] = {}
    bandit_action_counts: list[int] = []
    bandit_action_cost_ema_normalized: list[Optional[float]] = []
    bandit_action_cost_ema_usd: list[Optional[float]] = []
    rescue_action_counts: list[int] = []
    current_bandit_lambda = float(args.bandit_lambda)
    current_correctness_threshold = float(args.bandit_correctness_threshold)
    bandit_lambda_history: list[float] = []
    bandit_threshold_history: list[float] = []
    bandit_running_correct_count = 0
    bandit_running_sample_count = 0
    bandit_rescue_count = 0
    bandit_rescue_success_count = 0
    bandit_rescue_enabled = False
    running_cost_per_sample_ema_usd: Optional[float] = None
    running_cost_per_sample_ema_normalized: Optional[float] = None
    threshold_min_hit_count = 0
    threshold_max_hit_count = 0
    non_completed_status_count: dict[str, int] = {}
    rescue_profile_idx = 0
    max_budget_profile_idx = 0
    allowed_policy_action_indices: list[int] = []
    bandit_log_path = os.path.join(
        assignment_config.output_dir, "metrics", "dbbench_bandit.jsonl"
    )
    bandit_budget_target = max(float(args.bandit_budget_target), 1e-9)
    if bandit_enabled:
        from src.controllers.dbbench_linucb import (
            LinUCB,
            TwoHeadLinUCB,
            DBBenchFeatureAdapter,
            GENERAL_BUDGET_PRIMITIVES,
        )

        dbbench_budget_primitives = list(GENERAL_BUDGET_PRIMITIVES)
        dbbench_feature_adapter = DBBenchFeatureAdapter()
        bandit_action_counts = [0 for _ in dbbench_budget_primitives]
        bandit_action_cost_ema_normalized = [None for _ in dbbench_budget_primitives]
        bandit_action_cost_ema_usd = [None for _ in dbbench_budget_primitives]
        rescue_action_counts = [0 for _ in dbbench_budget_primitives]
        rescue_profile_idx = next(
            (
                idx
                for idx, profile in enumerate(dbbench_budget_primitives)
                if profile.name == args.bandit_rescue_profile_name
            ),
            max(
                range(len(dbbench_budget_primitives)),
                key=lambda idx: dbbench_budget_primitives[idx].static_cost_proxy,
            ),
        )
        max_budget_profile_idx = max(
            range(len(dbbench_budget_primitives)),
            key=lambda idx: dbbench_budget_primitives[idx].static_cost_proxy,
        )
        allowed_policy_action_indices = list(range(len(dbbench_budget_primitives)))
        if not args.bandit_allow_max_arm:
            allowed_policy_action_indices = [
                idx
                for idx in allowed_policy_action_indices
                if idx != max_budget_profile_idx
            ]
            # Safety fallback to avoid empty action set if config changes.
            if len(allowed_policy_action_indices) == 0:
                allowed_policy_action_indices = [max_budget_profile_idx]
        if args.bandit_two_head:
            dbbench_bandit = TwoHeadLinUCB(
                n_actions=len(dbbench_budget_primitives),
                feature_dim=dbbench_feature_adapter.feature_dim,
                alpha=float(args.bandit_alpha),
            )
        else:
            dbbench_bandit = LinUCB(
                n_actions=len(dbbench_budget_primitives),
                feature_dim=dbbench_feature_adapter.feature_dim,
                alpha=float(args.bandit_alpha),
            )
        os.makedirs(os.path.dirname(bandit_log_path), exist_ok=True)
        # Disable rescue path for stable cost accounting and policy learning.
        bandit_rescue_enabled = False
        logger.info(
            f"[DBBenchBandit] enabled. alpha={args.bandit_alpha}, lambda_init={args.bandit_lambda}, "
            f"two_head={args.bandit_two_head}, adaptive_lambda={args.bandit_adaptive_lambda}, "
            f"budget_target={args.bandit_budget_target}, lambda_lr={args.bandit_lambda_lr}, "
            f"policy={args.bandit_policy}, threshold_init={args.bandit_correctness_threshold}, "
            f"threshold_adaptive={args.bandit_adaptive_threshold}, target_acc={args.bandit_target_accuracy}, "
            f"acc_tol={args.bandit_accuracy_tolerance}, threshold_step={args.bandit_threshold_step}, "
            f"cost_ema_decay={args.bandit_cost_ema_decay}, "
            f"rescue_enable={bandit_rescue_enabled}, "
            f"rescue_profile={dbbench_budget_primitives[rescue_profile_idx].name}, "
            f"rescue_on_incorrect={args.bandit_rescue_on_incorrect}, "
            f"rescue_on_task_limit={args.bandit_rescue_on_task_limit}, "
            f"rescue_on_incomplete={args.bandit_rescue_on_incomplete}, "
            f"threshold_min={args.bandit_threshold_min}, threshold_max={args.bandit_threshold_max}, "
            f"allow_max_arm={args.bandit_allow_max_arm}, "
            f"cost_guardrail_enable={args.bandit_cost_guardrail_enable}, "
            f"cost_guardrail_band={args.bandit_cost_guardrail_band}, "
            f"actions={[p.name for p in dbbench_budget_primitives]}"
        )

    logger.info(
        f"Experiment start. "
        f"Total sample count: {len(assignment_config.sample_order)}. "
        f"Unfinished sample count: {len(unfinished_sample_order)}."
    )
    for sample_index in unfinished_sample_order:
        # region Initialize session
        session = Session(task_name=task.task_name, sample_index=sample_index)
        callback_args = CallbackArguments(
            current_session=session, task=task, agent=agent, session_list=session_list
        )
        callback_handler.on_session_create(callback_args)
        if callback_args.session_controller.should_task_reset:
            task.reset(session)
            callback_handler.on_task_reset(callback_args)
        if (
            bandit_enabled
            and dbbench_bandit is not None
            and dbbench_feature_adapter is not None
        ):
            dataset_item = task._get_current_dataset_item()  # noqa: SLF001
            features = dbbench_feature_adapter.build(dataset_item)
            candidate_indices = list(allowed_policy_action_indices)
            guardrail_over_budget = False
            if (
                args.bandit_cost_guardrail_enable
                and running_cost_per_sample_ema_normalized is not None
                and running_cost_per_sample_ema_normalized
                > (1.0 + float(args.bandit_cost_guardrail_band))
            ):
                guardrail_over_budget = True
                if len(candidate_indices) > 1:
                    max_candidate_idx = max(
                        candidate_indices,
                        key=lambda idx: dbbench_budget_primitives[
                            idx
                        ].static_cost_proxy,
                    )
                    candidate_indices = [
                        idx for idx in candidate_indices if idx != max_candidate_idx
                    ]
            if args.bandit_two_head:
                if args.bandit_policy == "cheapest_safe":
                    min_proxy = min(
                        p.static_cost_proxy for p in dbbench_budget_primitives
                    )
                    # Use observed EMA costs when available; fallback to scaled static proxy.
                    cost_hints = []
                    for idx, primitive in enumerate(dbbench_budget_primitives):
                        ema_cost = bandit_action_cost_ema_normalized[idx]
                        if ema_cost is not None:
                            cost_hints.append(ema_cost)
                        else:
                            proxy_scale = primitive.static_cost_proxy / min_proxy
                            cost_hints.append(proxy_scale)
                    action_idx, action_meta = (
                        dbbench_bandit.select_cheapest_safe_action(
                            features,
                            correctness_threshold=current_correctness_threshold,
                            action_cost_hint=cost_hints,
                            lambda_value=current_bandit_lambda,
                            candidate_action_indices=candidate_indices,
                        )
                    )
                else:
                    action_idx, action_meta = dbbench_bandit.select_action(
                        features,
                        lambda_value=current_bandit_lambda,
                        candidate_action_indices=candidate_indices,
                    )
            else:
                action_idx, ucb_score = dbbench_bandit.select_action(features)
                action_meta = {"score": ucb_score}
            profile = dbbench_budget_primitives[action_idx]
            bandit_action_counts[action_idx] += 1

            # Apply per-sample compute budget.
            if hasattr(agent, "_inference_config_dict"):
                if getattr(agent, "_inference_config_dict") is None:
                    setattr(agent, "_inference_config_dict", {})
                agent._inference_config_dict["max_new_tokens"] = profile.token_budget  # type: ignore[attr-defined]
            task.max_round = profile.max_round
            if hasattr(task, "tool_budget"):
                setattr(task, "tool_budget", int(profile.tool_budget))
            if hasattr(task, "stop_enabled"):
                setattr(task, "stop_enabled", bool(profile.stop_enabled))

            # Save info to update bandit after completion.
            calls = getattr(cost_tracker, "calls", None) or []
            bandit_state_by_sample[int(sample_index)] = {
                "features": features,
                "action_idx": action_idx,
                "profile_name": profile.name,
                "token_budget": profile.token_budget,
                "max_round": profile.max_round,
                "tool_budget": profile.tool_budget,
                "stop_enabled": profile.stop_enabled,
                "start_call_idx": len(calls),
                "action_meta": action_meta,
                "lambda_before": current_bandit_lambda,
                "threshold_before": current_correctness_threshold,
                "cost_hint": (
                    cost_hints[action_idx]
                    if args.bandit_two_head and args.bandit_policy == "cheapest_safe"
                    else None
                ),
                "rescue_triggered": False,
                "rescue_reason": None,
                "rescue_profile_name": None,
                "candidate_indices": candidate_indices,
                "guardrail_over_budget": guardrail_over_budget,
            }
            logger.info(
                f"[DBBenchBandit] sample={sample_index}, action={profile.name}, "
                f"token_budget={profile.token_budget}, max_round={profile.max_round}, "
                f"tool_budget={profile.tool_budget}, stop_enabled={profile.stop_enabled}, "
                f"lambda={current_bandit_lambda:.6f}, threshold={current_correctness_threshold:.4f}, "
                f"guardrail_over_budget={guardrail_over_budget}, "
                f"action_meta={action_meta}"
            )
        logger.info(f"Sample {sample_index} start.")
        # endregion
        # region Run session
        while session.sample_status == SampleStatus.RUNNING:
            if callback_args.session_controller.should_agent_inference:
                agent.inference(session)
                callback_handler.on_agent_inference(callback_args)
            if callback_args.session_controller.should_task_interact:
                task.interact(session)
                callback_handler.on_task_interact(callback_args)
        # endregion
        # region Complete session
        if callback_args.session_controller.should_task_complete:
            task.complete(session)
            callback_handler.on_task_complete(callback_args)
        original_session = session
        original_callback_args = callback_args
        if bandit_enabled and dbbench_bandit is not None:
            bandit_state = bandit_state_by_sample.get(int(sample_index))
            if bandit_state is not None:
                calls = getattr(cost_tracker, "calls", None) or []
                bandit_state["original_end_call_idx"] = len(calls)
                bandit_state["original_status"] = original_session.sample_status.value
                bandit_state["original_outcome"] = (
                    original_session.evaluation_record.outcome.value
                )
        if bandit_enabled and dbbench_bandit is not None:
            should_rescue = False
            rescue_reason: Optional[str] = None
            if original_session.sample_status != SampleStatus.COMPLETED:
                status_key = original_session.sample_status.value
                non_completed_status_count[status_key] = (
                    non_completed_status_count.get(status_key, 0) + 1
                )
            if (
                args.bandit_rescue_on_task_limit
                and original_session.sample_status == SampleStatus.TASK_LIMIT_REACHED
            ):
                should_rescue = True
                rescue_reason = "status=task_limit_reached"
            elif (
                args.bandit_rescue_on_incomplete
                and original_session.sample_status != SampleStatus.COMPLETED
            ):
                should_rescue = True
                rescue_reason = f"status={original_session.sample_status.value}"
            elif (
                args.bandit_rescue_on_incorrect
                and original_session.sample_status == SampleStatus.COMPLETED
                and original_session.evaluation_record.outcome.value != "correct"
            ):
                should_rescue = True
                rescue_reason = f"completed_but_incorrect:{original_session.evaluation_record.outcome.value}"

            if should_rescue and bandit_rescue_enabled:
                rescue_profile = dbbench_budget_primitives[rescue_profile_idx]
                rescue_action_counts[rescue_profile_idx] += 1
                bandit_rescue_count += 1
                bandit_state = bandit_state_by_sample.get(int(sample_index))
                if bandit_state is not None:
                    bandit_state["rescue_triggered"] = True
                    bandit_state["rescue_reason"] = rescue_reason
                    bandit_state["rescue_profile_name"] = rescue_profile.name
                    calls = getattr(cost_tracker, "calls", None) or []
                    bandit_state["rescue_start_call_idx"] = len(calls)
                logger.info(
                    f"[DBBenchBandit] rescue start sample={sample_index}, reason={rescue_reason}, "
                    f"profile={rescue_profile.name}, token_budget={rescue_profile.token_budget}, "
                    f"max_round={rescue_profile.max_round}"
                )

                rescue_session = Session(
                    task_name=task.task_name, sample_index=sample_index
                )
                rescue_callback_args = CallbackArguments(
                    current_session=rescue_session,
                    task=task,
                    agent=agent,
                    session_list=session_list,
                )
                callback_handler.on_session_create(rescue_callback_args)
                if rescue_callback_args.session_controller.should_task_reset:
                    task.reset(rescue_session)
                    callback_handler.on_task_reset(rescue_callback_args)

                if hasattr(agent, "_inference_config_dict"):
                    if getattr(agent, "_inference_config_dict") is None:
                        setattr(agent, "_inference_config_dict", {})
                    agent._inference_config_dict["max_new_tokens"] = rescue_profile.token_budget  # type: ignore[attr-defined]
                task.max_round = rescue_profile.max_round

                while rescue_session.sample_status == SampleStatus.RUNNING:
                    if rescue_callback_args.session_controller.should_agent_inference:
                        agent.inference(rescue_session)
                        callback_handler.on_agent_inference(rescue_callback_args)
                    if rescue_callback_args.session_controller.should_task_interact:
                        task.interact(rescue_session)
                        callback_handler.on_task_interact(rescue_callback_args)

                if rescue_callback_args.session_controller.should_task_complete:
                    task.complete(rescue_session)
                    callback_handler.on_task_complete(rescue_callback_args)

                if rescue_session.evaluation_record.outcome.value == "correct":
                    bandit_rescue_success_count += 1
                if bandit_state is not None:
                    calls = getattr(cost_tracker, "calls", None) or []
                    bandit_state["rescue_end_call_idx"] = len(calls)
                    bandit_state["final_status"] = rescue_session.sample_status.value
                    bandit_state["final_outcome"] = (
                        rescue_session.evaluation_record.outcome.value
                    )
                logger.info(
                    f"[DBBenchBandit] rescue end sample={sample_index}, "
                    f"status={rescue_session.sample_status}, outcome={rescue_session.evaluation_record.outcome}"
                )
                session = rescue_session
                callback_args = rescue_callback_args
            else:
                bandit_state = bandit_state_by_sample.get(int(sample_index))
                if bandit_state is not None:
                    bandit_state["rescue_start_call_idx"] = bandit_state.get(
                        "original_end_call_idx", bandit_state["start_call_idx"]
                    )
                    bandit_state["rescue_end_call_idx"] = bandit_state[
                        "rescue_start_call_idx"
                    ]
                    bandit_state["final_status"] = original_session.sample_status.value
                    bandit_state["final_outcome"] = (
                        original_session.evaluation_record.outcome.value
                    )
        session_list.append(session)
        json.dump(
            [s.model_dump() for s in session_list],
            open(session_list_output_path, "w"),  # noqa
            indent=2,
        )
        logger.info(
            f"Sample {sample_index} end. Session status: {session.sample_status}. "
            f"Evaluation outcome: {session.evaluation_record.outcome}."
        )
        if bandit_enabled and dbbench_bandit is not None:
            bandit_state = bandit_state_by_sample.get(int(sample_index))
            if bandit_state is not None:
                calls = getattr(cost_tracker, "calls", None) or []
                start_idx = int(bandit_state["start_call_idx"])
                original_end_idx = int(
                    bandit_state.get("original_end_call_idx", len(calls))
                )
                rescue_end_idx = int(
                    bandit_state.get("rescue_end_call_idx", len(calls))
                )
                original_calls = calls[start_idx:original_end_idx]
                rescue_calls = calls[original_end_idx:rescue_end_idx]
                original_cost = float(sum(c.total_cost_usd for c in original_calls))
                rescue_cost = float(sum(c.total_cost_usd for c in rescue_calls))
                sample_cost = original_cost + rescue_cost
                original_cost_normalized = original_cost / bandit_budget_target
                rescue_cost_normalized = rescue_cost / bandit_budget_target
                sample_cost_normalized = sample_cost / bandit_budget_target
                original_correct = bandit_state.get("original_outcome") == "correct"
                final_correct = session.evaluation_record.outcome.value == "correct"
                reward_lambda = float(bandit_state["lambda_before"])
                reward = (1.0 if original_correct else 0.0) - (
                    reward_lambda * original_cost_normalized
                )
                if args.bandit_two_head:
                    dbbench_bandit.update(
                        action_idx=int(bandit_state["action_idx"]),
                        x=bandit_state["features"],
                        correct_label=1.0 if original_correct else 0.0,
                        cost_label=original_cost_normalized,
                    )
                else:
                    dbbench_bandit.update(
                        action_idx=int(bandit_state["action_idx"]),
                        x=bandit_state["features"],
                        reward=reward,
                    )
                action_idx = int(bandit_state["action_idx"])
                prev_norm_ema = bandit_action_cost_ema_normalized[action_idx]
                prev_usd_ema = bandit_action_cost_ema_usd[action_idx]
                if prev_norm_ema is None:
                    bandit_action_cost_ema_normalized[action_idx] = (
                        original_cost_normalized
                    )
                else:
                    decay = float(args.bandit_cost_ema_decay)
                    bandit_action_cost_ema_normalized[action_idx] = (
                        decay * prev_norm_ema
                    ) + ((1.0 - decay) * original_cost_normalized)
                if prev_usd_ema is None:
                    bandit_action_cost_ema_usd[action_idx] = original_cost
                else:
                    decay = float(args.bandit_cost_ema_decay)
                    bandit_action_cost_ema_usd[action_idx] = (decay * prev_usd_ema) + (
                        (1.0 - decay) * original_cost
                    )
                if args.bandit_adaptive_lambda:
                    current_bandit_lambda = min(
                        1.0,
                        max(
                            0.0,
                            current_bandit_lambda
                            + (
                                float(args.bandit_lambda_lr)
                                * (sample_cost_normalized - 1.0)
                            ),
                        ),
                    )
                else:
                    current_bandit_lambda = min(1.0, max(0.0, current_bandit_lambda))
                bandit_lambda_history.append(current_bandit_lambda)
                if running_cost_per_sample_ema_usd is None:
                    running_cost_per_sample_ema_usd = sample_cost
                else:
                    guardrail_decay = float(args.bandit_cost_guardrail_decay)
                    running_cost_per_sample_ema_usd = (
                        guardrail_decay * running_cost_per_sample_ema_usd
                    ) + ((1.0 - guardrail_decay) * sample_cost)
                if running_cost_per_sample_ema_normalized is None:
                    running_cost_per_sample_ema_normalized = sample_cost_normalized
                else:
                    guardrail_decay = float(args.bandit_cost_guardrail_decay)
                    running_cost_per_sample_ema_normalized = (
                        guardrail_decay * running_cost_per_sample_ema_normalized
                    ) + ((1.0 - guardrail_decay) * sample_cost_normalized)
                bandit_running_sample_count += 1
                if final_correct:
                    bandit_running_correct_count += 1
                running_accuracy = (
                    float(bandit_running_correct_count)
                    / float(bandit_running_sample_count)
                    if bandit_running_sample_count > 0
                    else 0.0
                )
                if args.bandit_adaptive_threshold:
                    acc_target = float(args.bandit_target_accuracy)
                    acc_tol = float(args.bandit_accuracy_tolerance)
                    threshold_step = float(args.bandit_threshold_step)
                    threshold_min = float(args.bandit_threshold_min)
                    threshold_max = float(args.bandit_threshold_max)
                    if running_accuracy < (acc_target - acc_tol):
                        current_correctness_threshold = min(
                            threshold_max,
                            current_correctness_threshold + threshold_step,
                        )
                    elif running_accuracy > (acc_target + acc_tol):
                        current_correctness_threshold = max(
                            threshold_min,
                            current_correctness_threshold - threshold_step,
                        )
                    if current_correctness_threshold <= threshold_min:
                        threshold_min_hit_count += 1
                    if current_correctness_threshold >= threshold_max:
                        threshold_max_hit_count += 1
                bandit_threshold_history.append(current_correctness_threshold)
                os.makedirs(os.path.dirname(bandit_log_path), exist_ok=True)
                with open(bandit_log_path, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "sample_index": int(sample_index),
                                "correct": final_correct,
                                "original_correct": bool(original_correct),
                                "sample_cost_usd": sample_cost,
                                "original_cost_usd": original_cost,
                                "rescue_cost_usd": rescue_cost,
                                "sample_cost_normalized": sample_cost_normalized,
                                "original_cost_normalized": original_cost_normalized,
                                "rescue_cost_normalized": rescue_cost_normalized,
                                "reward": reward,
                                "action_idx": int(bandit_state["action_idx"]),
                                "profile_name": bandit_state["profile_name"],
                                "token_budget": int(bandit_state["token_budget"]),
                                "max_round": int(bandit_state["max_round"]),
                                "tool_budget": int(bandit_state["tool_budget"]),
                                "stop_enabled": bool(bandit_state["stop_enabled"]),
                                "action_meta": bandit_state.get("action_meta", {}),
                                "lambda_before": float(bandit_state["lambda_before"]),
                                "lambda_after": float(current_bandit_lambda),
                                "threshold_before": float(
                                    bandit_state["threshold_before"]
                                ),
                                "threshold_after": float(current_correctness_threshold),
                                "running_accuracy": float(running_accuracy),
                                "cost_hint": bandit_state.get("cost_hint"),
                                "budget_target": float(args.bandit_budget_target),
                                "candidate_indices": bandit_state.get(
                                    "candidate_indices"
                                ),
                                "guardrail_over_budget": bool(
                                    bandit_state.get("guardrail_over_budget", False)
                                ),
                                "running_cost_per_sample_ema_usd": (
                                    None
                                    if running_cost_per_sample_ema_usd is None
                                    else float(running_cost_per_sample_ema_usd)
                                ),
                                "running_cost_per_sample_normalized_ema": (
                                    None
                                    if running_cost_per_sample_ema_normalized is None
                                    else float(running_cost_per_sample_ema_normalized)
                                ),
                                "rescue_triggered": bool(
                                    bandit_state.get("rescue_triggered", False)
                                ),
                                "rescue_reason": bandit_state.get("rescue_reason"),
                                "rescue_profile_name": bandit_state.get(
                                    "rescue_profile_name"
                                ),
                            }
                        )
                        + "\n"
                    )
                logger.info(
                    f"[DBBenchBandit] sample={sample_index}, final_correct={final_correct}, "
                    f"original_correct={original_correct}, original_cost={original_cost:.6f}, "
                    f"rescue_cost={rescue_cost:.6f}, total_cost={sample_cost:.6f}, "
                    f"total_cost_normalized={sample_cost_normalized:.4f}, reward={reward:.6f}, "
                    f"lambda_after={current_bandit_lambda:.6f}, "
                    f"threshold_after={current_correctness_threshold:.4f}, "
                    f"running_accuracy={running_accuracy:.4f}, "
                    f"running_cost_ema_usd={0.0 if running_cost_per_sample_ema_usd is None else running_cost_per_sample_ema_usd:.6f}, "
                    f"running_cost_ema_normalized={0.0 if running_cost_per_sample_ema_normalized is None else running_cost_per_sample_ema_normalized:.4f}"
                )
        # endregion
        # region Save callback state
        # The state of callback will be used to restore the previous incomplete assignment.
        callback_handler.on_state_save(callback_args)
        # endregion
    # endregion
    # region Evaluate
    session_metric_calculation_partial_list: Sequence[
        SessionMetricCalculationPartial
    ] = [
        SessionMetricCalculationPartial(
            sample_index=session.sample_index,
            evaluation_record=session.evaluation_record,
            sample_status=session.sample_status,
        )
        for session in session_list
    ]
    metric = task.calculate_metric(session_metric_calculation_partial_list)
    cost_summary = cost_tracker.summary()
    sample_count = len(assignment_config.sample_order)
    mean_cost_per_sample = (
        float(cost_summary["total_cost_usd"]) / sample_count
        if sample_count > 0
        else 0.0
    )
    metric["cost"] = {
        "total_prompt_tokens": int(cost_summary["total_prompt_tokens"]),
        "total_completion_tokens": int(cost_summary["total_completion_tokens"]),
        "total_tokens": int(cost_summary["total_tokens"]),
        "total_cost_usd": float(cost_summary["total_cost_usd"]),
        "mean_cost_per_sample_usd": mean_cost_per_sample,
    }
    logger.info(
        f"Experiment end. Metric: {metric}. Total sample count: {len(assignment_config.sample_order)}.",
    )
    logger.info(
        "[RunCostSummary] "
        f"input_tokens={metric['cost']['total_prompt_tokens']}, "
        f"output_tokens={metric['cost']['total_completion_tokens']}, "
        f"total_tokens={metric['cost']['total_tokens']}, "
        f"total_cost_usd={metric['cost']['total_cost_usd']:.6f}, "
        f"mean_cost_per_sample_usd={metric['cost']['mean_cost_per_sample_usd']:.6f}"
    )
    if bandit_enabled:
        try:
            bandit_log_entries = []
            if os.path.exists(bandit_log_path):
                with open(bandit_log_path, "r") as f:
                    bandit_log_entries = [
                        json.loads(line) for line in f if line.strip()
                    ]
            total = len(bandit_log_entries)
            mean_cost = (
                sum(row["sample_cost_usd"] for row in bandit_log_entries) / total
                if total > 0
                else 0.0
            )
            mean_cost_normalized = (
                sum(
                    row.get(
                        "sample_cost_normalized",
                        float(row["sample_cost_usd"]) / bandit_budget_target,
                    )
                    for row in bandit_log_entries
                )
                / total
                if total > 0
                else 0.0
            )
            accuracy = (
                sum(1 for row in bandit_log_entries if row["correct"]) / total
                if total > 0
                else 0.0
            )
            original_accuracy = (
                sum(
                    1
                    for row in bandit_log_entries
                    if row.get("original_correct", False)
                )
                / total
                if total > 0
                else 0.0
            )
            pass_rate = accuracy
            cost_of_pass = (mean_cost / pass_rate) if pass_rate > 0 else None
            metric["dbbench_bandit"] = {
                "enabled": True,
                "alpha": float(args.bandit_alpha),
                "lambda_init": float(args.bandit_lambda),
                "lambda_final": float(current_bandit_lambda),
                "lambda_adaptive": bool(args.bandit_adaptive_lambda),
                "lambda_lr": float(args.bandit_lambda_lr),
                "budget_target": float(args.bandit_budget_target),
                "two_head": bool(args.bandit_two_head),
                "policy": str(args.bandit_policy),
                "correctness_threshold_init": float(args.bandit_correctness_threshold),
                "correctness_threshold_final": float(current_correctness_threshold),
                "threshold_adaptive": bool(args.bandit_adaptive_threshold),
                "threshold_min": float(args.bandit_threshold_min),
                "threshold_max": float(args.bandit_threshold_max),
                "samples": total,
                "action_counts": {
                    dbbench_budget_primitives[idx].name: int(count)
                    for idx, count in enumerate(bandit_action_counts)
                },
                "rescue_action_counts": {
                    dbbench_budget_primitives[idx].name: int(count)
                    for idx, count in enumerate(rescue_action_counts)
                },
                "rescue_count": int(bandit_rescue_count),
                "rescue_success_count": int(bandit_rescue_success_count),
                "rescue_enabled": bool(bandit_rescue_enabled),
                "action_cost_ema_usd": {
                    dbbench_budget_primitives[idx].name: (
                        None if ema is None else float(ema)
                    )
                    for idx, ema in enumerate(bandit_action_cost_ema_usd)
                },
                "action_cost_ema_normalized": {
                    dbbench_budget_primitives[idx].name: (
                        None if ema is None else float(ema)
                    )
                    for idx, ema in enumerate(bandit_action_cost_ema_normalized)
                },
                "mean_cost_usd": mean_cost,
                "mean_cost_normalized": mean_cost_normalized,
                "accuracy": accuracy,
                "original_accuracy": original_accuracy,
                "cost_of_pass": cost_of_pass,
                "running_cost_per_sample_ema_usd_final": (
                    None
                    if running_cost_per_sample_ema_usd is None
                    else float(running_cost_per_sample_ema_usd)
                ),
                "running_cost_per_sample_ema_normalized_final": (
                    None
                    if running_cost_per_sample_ema_normalized is None
                    else float(running_cost_per_sample_ema_normalized)
                ),
                "cost_guardrail_enable": bool(args.bandit_cost_guardrail_enable),
                "cost_guardrail_band": float(args.bandit_cost_guardrail_band),
                "allow_max_arm": bool(args.bandit_allow_max_arm),
                "normal_action_set": [
                    dbbench_budget_primitives[idx].name
                    for idx in allowed_policy_action_indices
                ],
                "mean_lambda": (
                    sum(bandit_lambda_history) / len(bandit_lambda_history)
                    if bandit_lambda_history
                    else float(args.bandit_lambda)
                ),
                "mean_threshold": (
                    sum(bandit_threshold_history) / len(bandit_threshold_history)
                    if bandit_threshold_history
                    else float(args.bandit_correctness_threshold)
                ),
                "threshold_min_hit_rate": (
                    float(threshold_min_hit_count) / float(total) if total > 0 else 0.0
                ),
                "threshold_max_hit_rate": (
                    float(threshold_max_hit_count) / float(total) if total > 0 else 0.0
                ),
                "non_completed_status_count": non_completed_status_count,
            }
        except Exception as e:
            logger.error(f"[DBBenchBandit] failed to append summary metric: {e}")
    json.dump(
        metric,
        open(path_config.metric_output_path, "w"),  # noqa
        indent=2,
    )
    logger.info(f"Metric file has been saved to {assignment_config.output_dir}.")
    # endregion
    # --------------------------------------
    # CostTracker save
    # --------------------------------------
    try:
        from pathlib import Path

        run_dir = Path(
            assignment_config.output_dir
        )  # same directory metrics are already saved to
        (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
        cost_tracker.save(str(run_dir / "metrics" / "token_cost_summary.json"))
        logger.info(
            f"[CostTracker] wrote {run_dir/'metrics'/'token_cost_summary.json'}"
        )
    except Exception as e:
        logger.error(f"[CostTracker] failed to save summary: {e}")
    # --------------------------------------
    # region Release
    task.release()
    # endregion


if __name__ == "__main__":
    main()
