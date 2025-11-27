import argparse
import sys
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import os
import importlib

from default_utils.datasets_manager import DatasetsManager
from default_utils.utils import import_yaml_lib
from models.model_manager import ModelManager

def parse_args():
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


def get_task_yaml():
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
    dataset_manager = DatasetsManager(cfg)
    
    # format prompts
    prompts = import_yaml_lib(cfg, "prompt_formatter")(cfg, dataset_manager)
    print(prompts)


    # generate outputs
    # TODO: let model manager handle model and generation types (generative, likelihood) 
    model_manager = ModelManager(cfg)


    # process output
    # output_processing_func = import_yaml_lib(cfg, "output_processor")


    # estimate confidence
    # confidence_estimation_func = import_yaml_lib(cfg, "confidence_metrics")

    # grade response
    # grader_func = import_yaml_lib(cfg, "grade_response")


    # calculate metrics
    # for metric in cfg.metrics:
    #     metric_func = import_yaml_lib(cfg, metric)