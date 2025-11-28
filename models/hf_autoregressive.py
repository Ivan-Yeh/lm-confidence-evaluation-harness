from tqdm import tqdm
from default_utils.custom_types import AbstractModel, ModelOutputs, PromptCollection
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F


class HFAutoregressiveLLM(AbstractModel):
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name = cfg.get("name", None)
        self.repeat = cfg.get("repeat", 1)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
    
    def run_generation(self, prompt_collection: PromptCollection) -> list[ModelOutputs]:
        # Build chat messages from prompt collection
        messages_list = []
        for context_text in prompt_collection.context_texts:
            messages = []
            if prompt_collection.system_prompt:
                messages.append({"role": "system", "content": prompt_collection.system_prompt})
            messages.append({"role": "user", "content": context_text})
            messages_list.append(messages)
        
        sampling_params = SamplingParams(
            **self.cfg.get("sampling_params", {})
        )
        vllm_model = LLM(model=self.model_name)

        model_outputs_list = []
        for _ in range(self.repeat):
            # Generate responses using vLLM chat
            outputs = vllm_model.chat(messages_list, sampling_params=sampling_params, chat_template_kwargs=self.cfg.get("chat_template_kwargs", None))
            
            # Extract output texts and tokens
            output_texts = []
            output_tokens = []
            output_logprobs = []
            
            for output in outputs:
                # For each prompt, collect all n completions
                for completion in output.outputs:
                    if "assistantfinal" in completion.text:
                        generated_text = completion.text.rsplit("assistantfinal", 1)[1].strip()
                    else:
                        generated_text = completion.text.strip()
                    output_texts.append(generated_text)
                    
                    # Tokenize generated text to get the expected number of tokens
                    generated_token_ids = self.tokenizer.encode(generated_text, add_special_tokens=False)
                    expected_length = len(generated_token_ids)
                    
                    # Extract decoded tokens and logprobs, skipping special tokens
                    tokens = []
                    logprobs = []
                    
                    if completion.logprobs:
                        for lp in completion.logprobs:
                            if lp and len(lp) > 0:
                                tok_info = list(lp.values())[0]
                                decoded_token = tok_info.decoded_token
                                
                                # Skip special tokens
                                if decoded_token not in self.tokenizer.all_special_tokens:
                                    tokens.append(decoded_token)
                                    logprobs.append(tok_info.logprob)
                    
                    # Slice tokens and logprobs to match generated text length
                    tokens = tokens[-expected_length:] if expected_length > 0 else tokens
                    logprobs = logprobs[-expected_length:] if expected_length > 0 else logprobs
                    output_tokens.append(tokens)
                    output_logprobs.append(logprobs)
            
            model_outputs_list.append(ModelOutputs(
                output_texts=output_texts,
                output_tokens=output_tokens,
                output_logprobs=output_logprobs,
            ))
        return model_outputs_list
        
    def run_continuation(self, prompt_collection: PromptCollection) -> list[ModelOutputs]:

        hf_model = AutoModelForCausalLM.from_pretrained(
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
