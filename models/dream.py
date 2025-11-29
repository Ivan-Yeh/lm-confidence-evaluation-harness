from tqdm import tqdm
from default_utils.custom_types import AbstractModel, ModelOutputs, PromptCollection
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


class DreamDLM(AbstractModel):
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name = cfg.get("name", None)
        self.repeat = cfg.get("repeat", 1)
        # Dream models require custom tokenizer/model code from the repo
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        
    
    def run_generation(self, prompt_collection: PromptCollection) -> list[ModelOutputs]:
        pass
        
    
    def run_continuation(self, prompt_collection: PromptCollection) -> list[ModelOutputs]:

        hf_model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,     # model weights can be BF16
            device_map="auto",
            trust_remote_code=True,
        )

        model_outputs_list = []
        
        for _ in range(self.repeat):
            all_output_texts = []
            all_output_tokens = []
            all_output_logprobs = []

            for context in tqdm(prompt_collection.context_texts, desc="Scoring continuations"):
                continuations = prompt_collection.continuation_texts[context]

                # Store all continuations with their scores
                candidates = []

                for continuation in continuations:

                    # 1. prepare full prompt
                    full_prompt = context + " " + continuation

                    # 2. tokenize (keep as Python ints)
                    ctx_ids  = self.tokenizer(context, add_special_tokens=False).input_ids
                    full_ids = self.tokenizer(full_prompt, add_special_tokens=False).input_ids
                    cont_ids = full_ids[len(ctx_ids):]

                    # 3. convert to tensor (must be int64)
                    input_ids = torch.tensor([full_ids], dtype=torch.long).to(hf_model.device)

                    # 4. forward pass
                    with torch.no_grad():
                        outputs = hf_model(input_ids=input_ids)
                        logits = outputs.logits  # [1, seq, vocab]

                    # 5. logprobs
                    logprobs = F.log_softmax(logits, dim=-1)

                    # 6. extract continuation logprobs
                    cont_tokens = []
                    cont_logps = []

                    offset = len(ctx_ids)

                    for i, token_id in enumerate(cont_ids):

                        tok_str = self.tokenizer.decode([token_id])

                        # logprob from the previous position
                        logp = logprobs[0, offset + i - 1, token_id].item()

                        cont_tokens.append(tok_str)
                        cont_logps.append(logp)

                    # Calculate sum of logprobs for this continuation
                    logprob_sum = sum(cont_logps)
                    
                    candidates.append({
                        'text': continuation,
                        'tokens': cont_tokens,
                        'logprobs': cont_logps,
                        'sum': logprob_sum
                    })

                # Select the continuation with maximum sum of logprobs
                best_candidate = max(candidates, key=lambda x: x['sum'])
                
                # 7. store only the best continuation
                all_output_texts.append(best_candidate['text'])
                all_output_tokens.append(best_candidate['tokens'])
                all_output_logprobs.append(best_candidate['logprobs'])

            model_outputs_list.append(
                ModelOutputs(
                    output_texts=all_output_texts,
                    output_tokens=all_output_tokens,
                    output_logprobs=all_output_logprobs,
                )
            )

        return model_outputs_list
