from default_utils.custom_types import ModelOutputs, PromptCollection, AbstractModel
from .dream import DreamDLM
from .vllm_model import vLLMModel
from .llada import LlaDADLM


class ModelManager:
    def __init__(self, master_cfg: dict, model_config_type="qa_model"):
        self.model_cfg: dict = master_cfg[model_config_type]
        self.model: AbstractModel
        
        match self.model_cfg.get("backend", None):
            case "vllm":
                self.model = vLLMModel(self.model_cfg)
            case "dream":
                self.model = DreamDLM(self.model_cfg)
            case "llada":
                self.model = LlaDADLM(self.model_cfg)
            case "openai_batch":
                raise NotImplementedError("Not implemented yet")
            case "claude_batch":
                raise NotImplementedError("Not implemented yet")
            case "together_ai_batch":
                raise NotImplementedError("Not implemented yet")
            case None:
                raise ValueError(f"Model type not specified in config for {model_config_type}")
            case _:
                raise ValueError(f"Unknown model type: {self.model_cfg.type}")

        self.tokenizer = self.model.tokenizer


    def run_generation(self, prompt_collection: PromptCollection) -> list[ModelOutputs]:
        return self.model.run_generation(prompt_collection)
    

    def run_continuation(self, prompt_collection: PromptCollection) -> list[ModelOutputs]:
        return self.model.run_continuation(prompt_collection)