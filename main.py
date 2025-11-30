import argparse
import sys
from hydra import compose, initialize_config_dir
import os

from default_utils.datasets_manager import DatasetsManager
from default_utils.utils import import_yaml_lib
from models.model_manager import ModelManager
from default_utils.custom_types import OrganisedOutputs, ModelOutputs, PromptCollection
from metrics import METRICS_FUNCTIONS

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
    prompts: PromptCollection = import_yaml_lib(cfg, "prompt_formatter")(cfg, dataset_manager)

    # generate qa outputs
    model_manager: ModelManager = ModelManager(master_cfg=cfg, model_config_type="qa_model")
    if cfg.get("generation_type", "generation") == "generation":
        outputs: list[ModelOutputs] = model_manager.run_generation(prompts)
    elif cfg.get("generation_type") == "continuation":
        outputs: list[ModelOutputs] = model_manager.run_continuation(prompts)
    else:
        raise ValueError(f"Unknown generation type: {cfg.generation_type}")
    
    # extract confidence
    confidence_extraction_func: callable = import_yaml_lib(cfg, "confidence_metrics")
    extracted_output: OrganisedOutputs = confidence_extraction_func(cfg, outputs, prompts) # a list of lists of confidence scores, each sublist corresponds to a sampling round

    # post process extracted responses and confidences
    if cfg.get("post_processor") is not None:
        post_processor: callable = import_yaml_lib(cfg, "post_processor")
        extracted_output: OrganisedOutputs = post_processor(cfg, outputs, extracted_output, prompts)

    # grade response
    grader_func: callable = import_yaml_lib(cfg, "grade_response")
    extracted_output.accuracy_scores = grader_func(cfg, extracted_output, prompts) # a list of lists of accuracy scores, each sublist corresponds to a sampling round

    # calculate metrics
    for metric in cfg.get("metrics", []):
        metric_func = METRICS_FUNCTIONS[metric]
        metric_value = metric_func(cfg, extracted_output)
        print(f"{metric}: {metric_value}")