from datasets import load_dataset
from .utils import import_yaml_lib


class DatasetsManager:
    def __init__(self, config):
        self.config = config
        self.dataset_name = config.get("dataset_name", None)
        self.dataset_path: str = config.get("dataset_path", None)
        self.subset: str = config.get("subset", None)
        self.split: str = config.get("split", None)
        self.fewshot_split: str = config.get("fewshot_split", None)
        self.fewshot_examples: int = config.get("fewshot_examples", None)
        self.limit: int = config.get("limit", None)
        self.random_state: int = config.get("seed", 42)
        self.preprocess_fn = import_yaml_lib(config, "dataset_preprocessor") if config.get("dataset_preprocessor") else lambda x: x


    def get_dataset(self):
        match self.dataset_path:
            case _:
                ds = load_dataset(self.dataset_path, self.subset)[self.split] if self.subset else load_dataset(self.dataset_path)[self.split]
        if self.limit:
            return self.preprocess_fn(ds.to_pandas().sample(n=self.limit, random_state=self.random_state))
        return self.preprocess_fn(ds.to_pandas())
    

    def get_few_shot_dataset(self):
        match self.dataset_path:
            case _:
                ds = load_dataset(self.dataset_path, self.subset)[self.fewshot_split] if self.subset else load_dataset(self.dataset_path)[self.fewshot_split]
        ds = load_dataset(self.dataset_path, self.subset)[self.fewshot_split] if self.subset else load_dataset(self.dataset_path)[self.fewshot_split]
        return self.preprocess_fn(ds.to_pandas().sample(n=self.fewshot_examples, random_state=self.random_state))