from tqdm import tqdm
from default_utils.custom_types import AbstractModel, ModelOutputs, PromptCollection
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import gc
import numpy as np

class HFAutoregressiveLLM(AbstractModel):
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name = cfg.get("name", None)
        self.repeat = cfg.get("repeat", 1)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        
    
    def run_generation(self, prompt_collection: PromptCollection) -> list[ModelOutputs]:
        from vllm import LLM, SamplingParams
        # Build chat messages from prompt collection
        messages_list = []
        for context_text in prompt_collection.context_texts:
            messages = []
            if prompt_collection.system_prompt:
                messages.append({"role": "system", "content": prompt_collection.system_prompt})
            messages.append({"role": "user", "content": context_text})
            messages_list.append(messages)
        
        sampling_params = SamplingParams(temperature=self.cfg.get("temperature", 1.0), 
                                         max_tokens=self.cfg.get("max_tokens", 256),
                                         logprobs=1,)
        vllm_model = LLM(model=self.model_name, max_model_len=self.cfg.get("max_model_len", 4096))

        model_outputs_list = []
        for _ in range(self.repeat):
            # Generate responses using vLLM chat
            try:
                outputs = vllm_model.chat(messages_list, sampling_params=sampling_params, chat_template_kwargs=self.cfg.get("chat_template_kwargs", None))
            except:
                outputs = vllm_model.generate(prompt_collection.context_texts, sampling_params=sampling_params)
            
            # Extract output texts and tokens
            output_texts = []
            output_tokens = []
            output_logprobs = []
            
            for output in outputs:
                # For each prompt, collect all n completions
                for completion in output.outputs:
                    if "assistantfinal" in completion.text:
                        generated_text = completion.text.rsplit("assistantfinal", 1)[-1].strip()
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
                context_texts=prompt_collection.context_texts,
                output_texts=output_texts,
                output_tokens=output_tokens,
                output_logprobs=output_logprobs,
            ))
            
        del vllm_model
        torch.cuda.empty_cache()
        return model_outputs_list
        
    def run_continuation(self, prompt_collection: PromptCollection) -> list[ModelOutputs]:
        hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        hf_model.eval()

        tokenizer = self.tokenizer
        model_outputs_list = []

        for _ in range(self.repeat):

            all_output_texts = []
            all_output_tokens = []
            all_output_logprobs = []

            for context in tqdm(prompt_collection.context_texts, desc="Scoring continuations"):
                continuations = prompt_collection.continuation_texts[context]

                # Encode the context once
                ctx_ids = tokenizer(context, add_special_tokens=False).input_ids

                candidates = []

                # ---- Score each continuation C = [c1, c2, ..., ck] ----
                for continuation in continuations:
                    cont_ids = tokenizer(continuation, add_special_tokens=False).input_ids

                    # rolling input: start from context
                    rolling_ids = ctx_ids.copy()

                    cont_logps = []

                    for token_id in cont_ids:

                        # Encode current prefix
                        input_ids = torch.tensor([rolling_ids], dtype=torch.long).to(hf_model.device)

                        # Run model
                        with torch.inference_mode():
                            outputs = hf_model(input_ids)
                            logits = outputs.logits   # [1, seq_len, vocab]

                        # Predict next token (use last position)
                        next_logits = logits[:, -1, :]
                        logprobs = torch.nn.functional.log_softmax(next_logits, dim=-1)

                        # logprob of actual next continuation token
                        lp = logprobs[0, token_id].item()
                        cont_logps.append(lp)

                        # Append this token and move to next
                        rolling_ids.append(token_id)

                    # Sum over continuation
                    logprob_mean = np.mean(cont_logps)

                    candidates.append({
                        "text": continuation,
                        "tokens": tokenizer.decode(cont_ids),
                        "logprobs": cont_logps,
                        "mean": logprob_mean,
                    })

                # ---- Choose best continuation by sum logprob ----
                best = max(candidates, key=lambda x: x["mean"])
                all_output_texts.append(best["text"])
                all_output_tokens.append(best["tokens"])
                all_output_logprobs.append(best["logprobs"])

            model_outputs_list.append(
                ModelOutputs(
                    context_texts=prompt_collection.context_texts,
                    output_texts=all_output_texts,
                    output_tokens=all_output_tokens,
                    output_logprobs=all_output_logprobs,
                )
            )

        hf_model.to("cpu")
        del input_ids
        del outputs
        del logits
        del logprobs
        del hf_model
        gc.collect()
        torch.cuda.empty_cache()
        return model_outputs_list
