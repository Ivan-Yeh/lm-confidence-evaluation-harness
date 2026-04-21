# Prompt:
"""
Generate a Python list of words or expressions that humans use to convey the level of confidence, certainty, or hedging in their statements (without a subject, only the linguistic cues). These words should include common hedging phrases, adverbs, and qualifiers that indicate varying degrees of certainty or uncertainty, from extremely low confidence (like I do not know, my random guess is, etc) to high confidence (certain, sure, definitely). 

The list should be comprehensive and cover a wide range of expressions used in everyday language as well as in academic and professional contexts.
"""

claude_lexicon = [

    # --- Very low confidence / complete uncertainty ---
    "I have no idea", "I don't know", "I have no clue", "beats me",
    "your guess is as good as mine", "I'm completely in the dark",
    "I haven't the faintest idea", "I'm totally lost on this",
    "my random guess is", "this is pure speculation",
    "I'm just guessing", "wildly guessing", "I'm shooting in the dark",
    "I'm totally uncertain", "not a clue",

    # --- Low confidence / high uncertainty ---
    "perhaps", "possibly", "conceivably", "it could be that",
    "there's a chance", "I suspect", "I vaguely recall",
    "I'm not sure but", "if I had to guess", "tentatively",
    "loosely speaking", "roughly", "something like",
    "I'm inclined to think", "it's not inconceivable",
    "I wouldn't rule out", "at a guess", "very roughly",
    "might", "may well", "could potentially",

    # --- Moderate uncertainty / hedged ---
    "I think", "I believe", "I suppose", "I imagine",
    "as far as I know", "to the best of my knowledge",
    "from what I understand", "it seems", "it appears",
    "seemingly", "apparently", "presumably", "supposedly",
    "if I'm not mistaken", "if memory serves",
    "generally speaking", "broadly speaking", "on the whole",
    "in most cases", "more or less", "more or less likely",
    "I would say", "I'd guess", "from what I can tell",
    "as I understand it", "in my view", "in my opinion",
    "to my mind", "as best as I can tell",

    # --- Moderate confidence / fairly sure ---
    "I'm fairly confident", "I'm fairly certain",
    "I'm reasonably sure", "I'm inclined to believe",
    "there's a good chance", "in all likelihood",
    "probably", "likely", "in all probability",
    "I would expect", "I'd expect", "I'd wager",
    "chances are", "odds are", "for the most part",
    "it's reasonable to assume", "it stands to reason",
    "on balance", "by and large", "largely",
    "it's safe to say", "one could argue", "the evidence suggests",
    "the data suggests", "indications are that", "it looks like",

    # --- High confidence / very sure ---
    "I'm confident", "I'm quite sure", "I'm fairly sure",
    "I'm pretty certain", "I have no doubt",
    "it's highly likely", "almost certainly", "very likely",
    "I strongly believe", "I'm convinced", "I firmly believe",
    "I'm positive", "clearly", "evidently", "obviously",
    "undoubtedly", "unquestionably", "without a doubt",
    "it's clear that", "it's evident that", "it's obvious that",
    "needless to say", "there's little doubt",
    "by all accounts", "demonstrably", "manifestly",

    # --- Very high confidence / certain ---
    "certainly", "definitely", "absolutely", "positively",
    "without question", "for certain", "for sure", "assuredly",
    "I'm certain", "I'm absolutely sure", "I'm 100% sure",
    "I guarantee", "I can confirm", "I can assure you",
    "it is certain that", "it is a fact that", "factually",
    "categorically", "unequivocally", "beyond any doubt",
    "beyond a shadow of a doubt", "beyond dispute",
    "it goes without saying", "indisputably", "incontrovertibly",
    "incontestably", "it is established that",
]

all_hedging_words = list(claude_lexicon)

import pandas as pd
import random
from vllm import LLM, SamplingParams
from tqdm import tqdm

def generate_hedging_examples():
    # Initialize vLLM with gpt-oss-20b
    llm = LLM(
        model="openai/gpt-oss-20b",
        dtype="bfloat16",
        trust_remote_code=True,
    )
    
    sampling_params = SamplingParams(
        temperature=1,
        max_tokens=1024,
        n=20,  # Generate 20 completions per prompt
        # stop=["\n\n", "Example:", "Sentence:"],
    )
    
    results = []
    
    # Build chat messages for all hedging words
    messages_batch = []
    hedging_words_batch = []
    examples_sentences = [
        "There is a correlation between X and Y.",
        "It rains tomorrow.",
        "The experiment shows a significant effect.",
        "The new policy improves the economy.",
        "The medication is effective in treating the disease.",
        "The new product is successful in the market.",
        "The neighbor is home.",
        "The movie is good.",
        "The restaurant serves delicious food.",
        "The city is the oldest in the country.",
        "The book is informative.",
        "The report is not accurate.",
    ]
    for word in tqdm(all_hedging_words, desc="Preparing prompts"):
        selected_sentence = random.choice(examples_sentences)
        messages = [
            {
                "role": "user",
                "content": f"""Given a linguistic cue: "{word}", rewrite one of the following non-verifiable statements to naturally include this cue to convey the intended level of confidence, certainty, or hedging.
                Please do not use other hedging words, hedging phrases or linguistic cues in the sentence other than the specified linguistic cue.
                
                Example sentences to rewrite:
                {selected_sentence}

                Do not use other hedging words or linguistic cues in the sentence. Do not combine linguistic cues. Do not include labels like "Example:" or "Sentence:". 
                Just provide the statement."""
            }
        ]
        messages_batch.append(messages)
        hedging_words_batch.append(word)
    
    print(f"Generating {len(messages_batch) * 5} example sentences...")
    
    # Generate all chat completions at once
    outputs = llm.chat(messages_batch, sampling_params, use_tqdm=True, chat_template_kwargs={"reasoning_effort": "low"})
    
    # Process outputs
    for word, output in zip(hedging_words_batch, outputs):
        for completion in output.outputs:
            sentence = completion.text.strip()
            # Clean up the sentence
            if sentence:
                # Remove any leading labels
                if "assistantfinal" in sentence:
                    sentence = sentence.split("assistantfinal")[-1].strip()
                    sentence = sentence.replace("Statement:", "").strip()
                    sentence = sentence.split("\n")[0].strip()  # Take only first line
                    if len(sentence) > 5:  # Ensure it's a reasonable sentence
                        results.append({
                            "hedging_word": word,
                            "example_sentence": sentence
                        })
    
    # Clean up
    llm.llm_engine.engine_core.shutdown()
    del llm
    return pd.DataFrame(results)

if __name__ == "__main__":
    print(f"Total hedging words: {len(all_hedging_words)}")
    
    # Generate examples
    df = generate_hedging_examples()
    
    print(f"\nGenerated {len(df)} example sentences")
    print(f"Average sentences per hedging word: {len(df) / len(all_hedging_words):.2f}")
    
    # Save to CSV
    output_path = "linguistic_confidence_lexicon/hedging_lexicon.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    
    # Display sample
    print("\nSample examples:")
    print(df.head(10).to_string())

