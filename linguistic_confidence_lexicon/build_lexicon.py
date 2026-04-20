# Prompt:
# Generate a Python list of words and expressions that human use to convey the level of confidence, certainty, or hedging in their statements (without a subject, only the linguistic cues). These words should include common hedging phrases, adverbs, and qualifiers that indicate varying degrees of certainty or uncertainty. 
# The list should be comprehensive and cover a wide range of expressions used in everyday language as well as in academic and professional contexts.

claude_lexicon = [

    # Certainty
    "certainly", "definitely", "absolutely", "undoubtedly", "unquestionably",
    "without a doubt", "for certain", "without question", "clearly", "obviously",
    "evidently", "plainly", "surely", "assuredly", "positively", "categorically",
    "unconditionally", "indisputably", "indubitably", "unequivocally",
    "of course", "naturally", "needless to say", "it goes without saying",
    "there is no doubt that", "it is clear that", "it is obvious that",
    "beyond question", "beyond doubt", "beyond a shadow of a doubt",
    "manifestly", "patently", "demonstrably", "incontrovertibly", "incontestably",

    # High confidence
    "almost certainly", "very likely", "highly likely", "in all likelihood",
    "in all probability", "almost definitely", "without reservation",
    "all but certain", "virtually certain", "as good as certain",
    "overwhelmingly likely", "strongly supported", "well-established",
    "widely confirmed", "conclusively shown",

    # Moderate confidence
    "probably", "likely", "presumably", "apparently", "seemingly",
    "it seems", "it appears", "it looks like", "it would seem",
    "it seems that", "it appears that", "reportedly", "ostensibly",
    "on the face of it", "to all appearances", "by all accounts",
    "on the evidence", "on balance",

    # Weak confidence
    "possibly", "perhaps", "maybe", "conceivably", "potentially",
    "it is possible that", "there is a chance that", "it may be that",
    "it might be that", "it could be that", "there is a possibility that",
    "not impossible", "not out of the question", "feasibly",
    "under some circumstances", "in some cases", "in certain conditions",
    "with some likelihood", "theoretically possible",

    # Uncertainty / doubt
    "it is unclear", "it is uncertain", "it remains to be seen",
    "that is debatable", "it is questionable whether", "it is hard to say",
    "open to debate", "open to question", "yet to be determined",
    "remains unclear", "remains uncertain", "unresolved", "contested",
    "disputed", "not yet established", "inconclusive", "ambiguous",
    "equivocal", "indeterminate", "speculative", "conjectural", "tentative",
    "at issue", "under debate", "in question", "far from settled",
    "by no means certain",

    # Approximation
    "roughly", "approximately", "about", "around", "more or less",
    "give or take", "in the ballpark of", "somewhere around", "or so",
    "in the region of", "of the order of", "in the vicinity of",
    "upwards of", "close to", "nearly", "almost", "practically",
    "effectively", "essentially", "broadly", "loosely",

    # Qualification / degree
    "sort of", "kind of", "somewhat", "rather", "fairly", "quite",
    "to some extent", "to a certain degree", "in a way", "in some ways",
    "largely", "mostly", "mainly", "chiefly", "primarily", "predominantly",
    "entirely", "completely", "wholly", "fully", "partially", "marginally",
    "slightly", "barely", "hardly", "scarcely", "virtually", "nearly",
    "fundamentally", "technically", "in part", "to a degree",
    "for the most part", "by and large", "on the whole", "in general",
    "broadly speaking", "generally speaking", "loosely speaking",
    "as a rule", "typically", "usually", "normally", "ordinarily",
    "in most cases", "more often than not",

    # Academic / formal
    "it is suggested that", "it has been argued that",
    "it could be argued that", "it is worth noting that",
    "it should be noted that", "it is important to note that",
    "it is reasonable to assume that", "there is reason to believe that",
    "preliminary findings suggest", "available evidence points to",
    "it is widely believed that", "it is generally accepted that",
    "it remains unclear whether", "further research is needed",
    "under certain conditions", "depending on", "subject to",
    "contingent upon", "assuming that", "it is worth considering that",
    "it has been observed that", "it is commonly held that",
    "it is often assumed that", "evidence is consistent with",
    "data are consistent with", "it cannot be ruled out that",
    "it would be premature to conclude", "cautious interpretation suggests",
    "tentative evidence points to", "on current evidence",
    "as currently understood", "pending further investigation",
    "with the caveat that", "with the qualification that",
    "notwithstanding", "subject to revision",

    # Conditional / hypothetical
    "in theory", "theoretically", "hypothetically", "in principle",
    "on paper", "ideally", "in an ideal world", "under normal circumstances",
    "all things being equal", "all else being equal",
    "under certain assumptions", "given certain conditions",
    "in certain circumstances", "if conditions hold", "ceteris paribus",
    "in a best-case scenario", "in a worst-case scenario",
    "under optimal conditions", "holding everything else constant",
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

