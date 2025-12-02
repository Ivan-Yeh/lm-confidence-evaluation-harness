import argparse
import importlib
import pkgutil
import sys
from hydra import compose, initialize_config_dir
import os

from default_utils.datasets_manager import DatasetsManager
from models.model_manager import ModelManager
from default_utils.utils import import_yaml_lib
from default_utils.custom_types import OrganisedOutputs, ModelOutputs, PromptCollection
from default_utils.registry import METRICS_FUNCTIONS, GRADER_FUNCTIONS, CONFIDENCE_FUNCTIONS, PROMPT_FORMATTER

def _auto_import_modules(): 
    import default_utils
    for package_path in default_utils.__path__:
        for module_info in pkgutil.iter_modules([package_path]):
            module_name = module_info.name
            if module_name.startswith("_"):
                continue
            try:
                importlib.import_module(f"{default_utils.__name__}.{module_name}")
            except ImportError:
                pass
    import confidence_metrics
    for package_path in confidence_metrics.__path__:
        for module_info in pkgutil.iter_modules([package_path]):
            module_name = module_info.name
            if module_name.startswith("_"):
                continue
            try:
                importlib.import_module(f"{confidence_metrics.__name__}.{module_name}")
            except ImportError:
                pass
    import post_processing
    for package_path in post_processing.__path__:
        for module_info in pkgutil.iter_modules([package_path]):
            module_name = module_info.name
            if module_name.startswith("_"):
                continue
            importlib.import_module(f"{post_processing.__name__}.{module_name}")
_auto_import_modules()

def parse_args() -> tuple[str, str, list[str]]:
    parser = argparse.ArgumentParser(description='Run tasks with Hydra config')
    parser.add_argument('args', nargs='*', help='Config overrides in key=value format')
    # Parse arguments
    args = parser.parse_args()
    # Extract task name from overrides
    dataset_name = None
    task_name = None
    other_overrides = []
    for override in args.args:
        if override.startswith('dataset='):
            dataset_name = override.split('=', 1)[1]
        elif override.startswith('task='):
            task_name = override.split('=', 1)[1]
        else:
            other_overrides.append(override)
    if not dataset_name:
        print("Error: You must specify 'dataset=<dataset_name>'")
        sys.exit(1)
    if not task_name:
        print("Error: You must specify 'task=<task_name>'")
        sys.exit(1)
    return dataset_name, task_name, other_overrides


def get_task_yaml() -> tuple[str, str, dict]:
    dataset_name, task_name, overrides = parse_args()
    
    # Get absolute path to config directory
    config_dir = os.path.abspath(f"tasks/{dataset_name}")
    
    # Initialize Hydra with the config directory and tasks search path
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        # Compose config with the task and any overrides
        cfg = compose(
            config_name=f"{task_name}",
            overrides=overrides
        )
        return dataset_name, task_name, cfg


if __name__ == "__main__":
    # obtain config
    dataset_name, task_name, cfg = get_task_yaml()

    # prepare dataset
    dataset_manager: DatasetsManager = DatasetsManager(cfg)

    # format prompts
    prompts: PromptCollection = PROMPT_FORMATTER.get(cfg.get("prompt_formatter", "multiple_choice"))(cfg, dataset_manager)
    if prompts is None:
        prompts = import_yaml_lib(cfg, "prompt_formatter")(cfg, dataset_manager)

    # generate qa outputs
    model_manager: ModelManager = ModelManager(master_cfg=cfg, model_config_type="qa_model")
    if cfg.get("generation_type", "generation") == "generation":
        outputs: list[ModelOutputs] = model_manager.run_generation(prompts)
    elif cfg.get("generation_type") == "continuation":
        outputs: list[ModelOutputs] = model_manager.run_continuation(prompts)
    else:
        raise ValueError(f"Unknown generation type: {cfg.generation_type}")
    
    # extract confidence
    confidence_extraction_func: callable = CONFIDENCE_FUNCTIONS.get(cfg.get("confidence_metrics", "length_normalised_log_likelihood"))
    if confidence_extraction_func is None:
        confidence_extraction_func = import_yaml_lib(cfg, "confidence_metrics")
    extracted_output: OrganisedOutputs = confidence_extraction_func(cfg, outputs, prompts)

    # post process extracted responses and confidences
    if cfg.get("post_processor") is not None:
        post_processor: callable = import_yaml_lib(cfg, "post_processor")
        extracted_output: OrganisedOutputs = post_processor(cfg, outputs, extracted_output, prompts)

    # grade response
    grader_func: callable = GRADER_FUNCTIONS.get(cfg.get("grader", "exact_match"))
    if grader_func is None:
        grader_func = import_yaml_lib(cfg, "grader")
    extracted_output.accuracy_scores = grader_func(cfg, extracted_output, prompts)
    print(extracted_output.accuracy_scores)
    # calculate metrics
    for metric in cfg.get("performance_metrics", []):
        metric_func = METRICS_FUNCTIONS[metric]
        metric_value = metric_func(cfg, extracted_output)
        print(f"{metric}: {metric_value}")