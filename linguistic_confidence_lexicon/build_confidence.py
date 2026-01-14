import pandas as pd
import re
from vllm import LLM, SamplingParams
from tqdm import tqdm

# Read the hedging lexicon CSV
df = pd.read_csv("linguistic_confidence_lexicon/hedging_lexicon.csv")

print(f"Loaded {len(df)} sentences from hedging_lexicon.csv")
print(f"Columns: {df.columns.tolist()}")
print(f"Sample:\n{df.head()}\n")

# List of models to use
MODELS = [
    "openai/gpt-oss-20b",
    "meta-llama/Llama-3.1-8B-Instruct",
    "qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

# Sampling parameters for chat
sampling_params = SamplingParams(
    temperature=1,
    max_tokens=100,
)

def extract_confidence_score(response_text):
    """
    Extract confidence score from response using regex, with fallback to any int.
    Returns the score as int or None if not found.
    """
    # Try to extract a number between 0 and 100
    match = re.search(r'\b([0-9]{1,3})\b', response_text)
    if match:
        score = int(match.group(1))
        if 0 <= score <= 100:
            return score / 100
    
    # Fallback: look for any integer in the response
    match = re.search(r'[-+]?\d+', response_text)
    if match:
        score = int(match.group(0))
        if 0 <= score <= 100:
            return score / 100
    
    return None

def evaluate_sentences_with_model(model_name, sentences, num_repeats=10):
    """
    Evaluate sentences with a specific model, repeating each sentence num_repeats times.
    Returns a DataFrame with columns: sentence, hedging_word, model, scores (list of ints)
    """
    
    print(f"\nInitializing model: {model_name}")
    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        max_model_len = 1024
    )
    
    results = []
    
    # Create prompts: for each sentence, we'll evaluate it num_repeats times
    prompt_data = []  # (sentence_text, hedging_word, repeat_idx)
    
    for idx, row in df.iterrows():
        sentence = row['example_sentence']
        hedging_word = row['hedging_word']
        for repeat in range(num_repeats):
            prompt_data.append((sentence, hedging_word, repeat))
    
    # Build chat messages
    messages_batch = []
    for sentence, hedging_word, repeat in prompt_data:
        prompt = f"""Please provide only a confidence score between 0 and 100, based solely on the degree of confidence expressed in the tone and linguistic cues of the following sentence (without using any external or prior knowledge).
                    Please pay attention to the hedging language used and the overall assertiveness of the statement.
                    If the sentence contains random guesses or abstention, the score should be towards 0. If the sentence is stated with strong certainty or no hedging, the score should be towards 100.
                    If the sentence does not contain any linguistic cues or is a succinct, decisive short answer, the score should be towards 100, too.

                    Here is the sentence:
                    {sentence}

                    Confidence Score: [Return only a number between 0 and 100]""".strip()
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        messages_batch.append(messages)
    
    print(f"Evaluating {len(messages_batch)} sentence evaluations ({len(df)} sentences × {num_repeats} repeats)")
    
    # Generate all chat completions at once
    outputs = llm.chat(messages_batch, sampling_params, use_tqdm=True, chat_template_kwargs={"reasoning_effort": "low"})
    
    # Process outputs and group by sentence
    scores_by_sentence = {}
    
    for (sentence, hedging_word, repeat), output in zip(prompt_data, outputs):
        if sentence not in scores_by_sentence:
            scores_by_sentence[sentence] = {
                'hedging_word': hedging_word,
                'scores': []
            }
        
        # Extract score from response
        response_text = output.outputs[0].text.strip()
        score = extract_confidence_score(response_text)
        
        if score is not None:
            scores_by_sentence[sentence]['scores'].append(score)
    
    # Build results dataframe
    for sentence, data in scores_by_sentence.items():
        results.append({
            'sentence': sentence,
            'hedging_word': data['hedging_word'],
            'model': model_name,
            'scores': data['scores'],
            'num_scores': len(data['scores']),
            'mean_score': sum(data['scores']) / len(data['scores']) if data['scores'] else None,
        })
    
    # Clean up
    llm.llm_engine.engine_core.shutdown()
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    all_results = []
    
    for model in MODELS:
        print(f"\n{'='*80}")
        print(f"Evaluating with model: {model}")
        print(f"{'='*80}")
        
        result_df = evaluate_sentences_with_model(model, df['example_sentence'].tolist(), num_repeats=5)
        all_results.append(result_df)
        
        print(f"\nResults for {model}:")
        print(f"  Total sentences evaluated: {len(result_df)}")
        print(f"  Average scores per sentence: {result_df['num_scores'].mean():.2f}")
        print(f"  Sample results:\n{result_df.head()}\n")
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Save to CSV
    output_path = "linguistic_confidence_lexicon/"
    combined_df.to_csv(output_path + "confidence_scores.csv", index=False)
    combined_df.to_pickle(output_path + "confidence_scores.pkl")
    print(f"\nSaved all results to {output_path}")
    
    print(f"\nTotal rows: {len(combined_df)}")
    print(f"Columns: {combined_df.columns.tolist()}")
    print(f"Sample output:\n{combined_df.head(10)}")
