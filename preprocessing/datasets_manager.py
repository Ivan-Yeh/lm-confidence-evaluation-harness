from datasets import load_dataset


class DatasetsManager:
    def __init__(self, config):
        self.config = config
        self.dataset_path: str = config.get("dataset_path", None)
        self.subset: str = config.get("subset", None)
        self.split: str = config.get("split", None)
        self.fewshot_split: str = config.get("fewshot_split", None)
        self.fewshot_examples: int = config.get("fewshot_examples", None)
        self.limit: int = config.get("limit", None)
        self.random_state: int = config.get("seed", 42)

        
    def load_dataset(self):
        match self.dataset_path:
            case _:
                ds = load_dataset(self.dataset_path, self.subset)[self.split] if self.subset else load_dataset(self.dataset_path)[self.split]
        if self.limit:
            return ds.to_pandas().sample(n=self.limit, random_state=self.random_state)
        return ds.to_pandas()
    

    def load_fewshot_dataset(self):
        match self.dataset_path:
            case _:
                ds = load_dataset(self.dataset_path, self.subset)[self.fewshot_split] if self.subset else load_dataset(self.dataset_path)[self.fewshot_split]
        ds = load_dataset(self.dataset_path, self.subset)[self.fewshot_split] if self.subset else load_dataset(self.dataset_path)[self.fewshot_split]
        return ds.to_pandas().sample(n=self.fewshot_examples, random_state=self.random_state)