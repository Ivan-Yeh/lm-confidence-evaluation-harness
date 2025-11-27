from default_utils.custom_types import PromptCollection


class ModelManager:
    def __init__(self, cfg):
        self.cfg = cfg


    def _generate(self, prompt_collection: PromptCollection) -> tuple[list[str], list[dict]]:
        pass
    

    def _conditional_likelihood(self, prompt_collection: PromptCollection) -> tuple[list[str], list[dict]]:
        pass