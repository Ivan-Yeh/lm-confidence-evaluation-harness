import argparse
import sys
from hydra import compose, initialize_config_dir
import os

from default_utils.datasets_manager import DatasetsManager
from default_utils.utils import import_yaml_lib
from models.model_manager import ModelManager
from default_utils.custom_types import ModelOutputs, PromptCollection


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
        outputs: ModelOutputs = model_manager.run_generation(prompts)
    elif cfg.get("generation_type") == "continuation":
        outputs: ModelOutputs = model_manager.run_continuation(prompts)
    else:
        raise ValueError(f"Unknown generation type: {cfg.generation_type}")

    print(outputs)
    # process output
    output_processing_func = import_yaml_lib(cfg, "output_processor")
    outputs = output_processing_func(outputs)


    # extract confidence
    confidence_extraction_func: callable = import_yaml_lib(cfg, "confidence_metrics")
    confidence_scores: list = confidence_extraction_func(cfg, outputs)

    # grade response
    # grader_func: callable = import_yaml_lib(cfg, "grade_response")
    # grades: list = grader_func(outputs, dataset_manager)


    # calculate metrics
    # for metric in cfg.metrics:
    #     metric_func = import_yaml_lib(cfg, metric)