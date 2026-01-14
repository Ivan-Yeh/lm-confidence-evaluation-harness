# Prompt:
# Generate a Python list of 200+ words and expressions that human use to convey the level of confidence, certainty, or hedging in their statements. These words should include common hedging phrases, adverbs, and qualifiers that indicate varying degrees of certainty or uncertainty. 
# The list should be comprehensive and cover a wide range of expressions used in everyday language as well as in academic and professional contexts.


gpt_lexicon = [
    "perhaps", "maybe", "possibly", "probably", "likely", "unlikely", "certainly",
    "definitely", "undoubtedly", "arguably", "apparently", "seemingly", "presumably",
    "conceivably", "plausibly", "potentially", "theoretically", "practically",
    "ostensibly", "reportedly", "supposedly", "allegedly", "roughly", "approximately",
    "about", "around", "nearly", "almost", "somewhat", "partially", "largely",
    "mostly", "generally", "typically", "often", "sometimes", "rarely", "seldom",
    "frequently", "commonly", "usually", "in general", "by and large", "to some extent",
    "to a degree", "to a certain extent", "to some degree", "in part", "in theory",
    "in practice", "in principle", "in my view", "in my opinion", "from my perspective",
    "as far as I know", "as far as I can tell", "as far as we know", "as far as one can tell",
    "it seems", "it seems that", "it appears", "it appears that", "it may be",
    "it might be", "it could be", "it is possible that", "it is likely that",
    "it is unlikely that", "it is plausible that", "it is conceivable that",
    "it is probable that", "it is doubtful that", "it is unclear whether",
    "it is not clear that", "there is a chance that", "there is a possibility that",
    "there is a likelihood that", "there is reason to believe", "there is reason to doubt",
    "there is evidence that", "there is some evidence that", "there is limited evidence that",
    "the evidence suggests", "the evidence indicates", "the data suggest",
    "the data indicate", "the results suggest", "the findings suggest",
    "the findings indicate", "one might argue", "one could argue", "one may argue",
    "it can be argued", "it could be argued", "it might be argued",
    "I think", "I believe", "I suspect", "I guess", "I assume", "I suppose",
    "I imagine", "I feel", "I would say", "I would argue", "I would suggest",
    "I would expect", "I tend to think", "I am inclined to think",
    "I am confident that", "I am fairly confident that", "I am reasonably confident that",
    "I am not sure", "I am unsure", "I am uncertain", "I am doubtful",
    "I am skeptical", "I am convinced", "I am certain", "I am sure",
    "I am fairly sure", "I am pretty sure", "I am almost sure",
    "with certainty", "with confidence", "with high confidence",
    "with low confidence", "with reasonable confidence",
    "beyond doubt", "without doubt", "without a doubt", "no doubt",
    "little doubt", "some doubt", "reasonable doubt",
    "highly likely", "very likely", "quite likely", "more likely than not",
    "less likely", "highly unlikely", "very unlikely",
    "almost certainly", "most likely", "quite possibly",
    "barely", "hardly", "scarcely",
    "in all likelihood", "in all probability",
    "as a rule", "as a general rule",
    "broadly speaking", "strictly speaking",
    "loosely speaking", "roughly speaking",
    "on balance", "overall", "all things considered",
    "at least", "at most",
    "arguably speaking", "debatably",
    "hypothetically", "tentatively", "provisionally",
    "preliminarily", "exploratorily",
    "empirically", "anecdotally",
    "statistically", "qualitatively",
    "quantitatively", "nominally",
    "relatively", "comparatively",
    "subjectively", "objectively",
    "to the best of my knowledge", "to the best of our knowledge",
    "to my knowledge", "to our knowledge",
    "as I understand it", "as we understand it",
    "if I am not mistaken", "if I am correct",
    "correct me if I am wrong",
    "it is fair to say", "it is safe to say",
    "it stands to reason", "it goes without saying",
    "there is consensus that", "there appears to be consensus that",
    "there is broad agreement that",
    "there is debate about whether",
    "there is uncertainty about whether",
    "there is ambiguity regarding",
    "there is some ambiguity regarding",
    "there is room for doubt",
    "there is room for interpretation"
]

claude_lexicon = [
    # Absolute certainty
    "definitely",
    "certainly",
    "absolutely",
    "undoubtedly",
    "unquestionably",
    "indisputably",
    "without a doubt",
    "beyond doubt",
    "without question",
    "clearly",
    "obviously",
    "evidently",
    "manifestly",
    "plainly",
    "undeniably",
    "incontrovertibly",
    "irrefutably",
    "categorically",
    "positively",
    "assuredly",
    
    # High certainty
    "very likely",
    "highly probable",
    "almost certainly",
    "virtually certain",
    "nearly certain",
    "most definitely",
    "most certainly",
    "in all likelihood",
    "in all probability",
    "no doubt",
    "doubtless",
    "surely",
    "confidently",
    
    # Moderate-high certainty
    "probably",
    "likely",
    "quite likely",
    "quite possibly",
    "reasonably certain",
    "fairly certain",
    "fairly confident",
    "I believe",
    "I'm confident that",
    "I'm certain that",
    "I'm convinced that",
    "I'd say",
    "I would argue",
    
    # Moderate certainty
    "generally",
    "usually",
    "typically",
    "normally",
    "commonly",
    "ordinarily",
    "as a rule",
    "for the most part",
    "by and large",
    "on the whole",
    "in general",
    "broadly speaking",
    "in most cases",
    "more often than not",
    
    # Moderate-low certainty
    "possibly",
    "perhaps",
    "maybe",
    "conceivably",
    "potentially",
    "feasibly",
    "it's possible that",
    "it's conceivable that",
    "there's a chance that",
    "it could be that",
    "it may be that",
    "it might be that",
    
    # Hedging and uncertainty
    "arguably",
    "presumably",
    "supposedly",
    "allegedly",
    "reportedly",
    "seemingly",
    "apparently",
    "ostensibly",
    "purportedly",
    "I think",
    "I believe",
    "I suppose",
    "I imagine",
    "I assume",
    "I presume",
    "I suspect",
    "I reckon",
    "I gather",
    "I would guess",
    "I would estimate",
    "I would venture to say",
    
    # Weak certainty/strong hedging
    "might",
    "may",
    "could",
    "could be",
    "might be",
    "may be",
    "somewhat",
    "rather",
    "relatively",
    "comparatively",
    "to some extent",
    "to a certain extent",
    "to some degree",
    "in some ways",
    "in certain respects",
    "sort of",
    "kind of",
    "more or less",
    
    # Conditional/tentative
    "if I'm not mistaken",
    "if I recall correctly",
    "if memory serves",
    "as far as I know",
    "as far as I can tell",
    "to the best of my knowledge",
    "to my knowledge",
    "to my understanding",
    "in my opinion",
    "in my view",
    "from my perspective",
    "it seems to me",
    "it appears that",
    "it would seem that",
    "it would appear that",
    
    # Approximation
    "approximately",
    "roughly",
    "about",
    "around",
    "circa",
    "nearly",
    "almost",
    "practically",
    "essentially",
    "basically",
    "more or less",
    "give or take",
    "or so",
    "in the ballpark of",
    "in the region of",
    "in the vicinity of",
    
    # Frequency hedges
    "sometimes",
    "occasionally",
    "often",
    "frequently",
    "rarely",
    "seldom",
    "hardly ever",
    "scarcely",
    "at times",
    "from time to time",
    "now and then",
    "once in a while",
    
    # Limiting scope
    "in some cases",
    "in many cases",
    "in certain cases",
    "under certain conditions",
    "under some circumstances",
    "depending on",
    "it depends",
    "largely",
    "mainly",
    "primarily",
    "chiefly",
    "predominantly",
    "mostly",
    "in part",
    "partially",
    "partly",
    
    # Academic/formal hedging
    "tends to",
    "appears to",
    "seems to",
    "suggests that",
    "indicates that",
    "implies that",
    "may indicate",
    "may suggest",
    "could indicate",
    "could suggest",
    "would seem to",
    "would appear to",
    "is thought to",
    "is believed to",
    "is considered to",
    "is regarded as",
    "is seen as",
    
    # Speculation
    "hypothetically",
    "theoretically",
    "speculatively",
    "conjecturally",
    "one might argue",
    "one could argue",
    "it's arguable that",
    "it's debatable whether",
    "it's questionable whether",
    
    # Reservation/caution
    "with reservations",
    "with caution",
    "tentatively",
    "provisionally",
    "subject to",
    "contingent upon",
    "provided that",
    "assuming that",
    "granted that",
    "given that",
    
    # Downplayers
    "just",
    "only",
    "merely",
    "simply",
    "hardly",
    "scarcely",
    "barely",
    "slightly",
    "somewhat",
    "a bit",
    "a little",
    "a touch",
    "marginally",
    "minimally",
    
    # Intensifiers (high confidence)
    "extremely",
    "highly",
    "very",
    "exceptionally",
    "remarkably",
    "particularly",
    "especially",
    "notably",
    "decidedly",
    "distinctly",
    "markedly",
    "significantly",
    "substantially",
    "considerably",
    
    # Epistemic markers
    "arguably",
    "admittedly",
    "granted",
    "naturally",
    "of course",
    "needless to say",
    "it goes without saying",
    "as is well known",
    "as everyone knows",
    
    # Probability expressions
    "odds are",
    "chances are",
    "the likelihood is",
    "in all probability",
    "more likely than not",
    "less likely",
    "unlikely",
    "highly unlikely",
    "improbable",
    
    # Personal attribution
    "I feel",
    "I sense",
    "I have a feeling",
    "I have a hunch",
    "my impression is",
    "my sense is",
    "my understanding is",
    "from what I understand",
    "from what I gather",
    
    # Additional hedges
    "so to speak",
    "as it were",
    "in a sense",
    "in a way",
    "in a manner of speaking",
    "if you will",
    "shall we say",
    "for want of a better word",
    "for lack of a better term"
]

gemini_lexicon = [
    # High Certainty (100% - 90%)
    "absolutely", "certainly", "definitely", "clearly", "obviously", 
    "undeniably", "unquestionably", "without a doubt", "for sure", "positive", 
    "conclusive", "indisputable", "plainly", "strictly", "categorically", 
    "entirely", "fully", "totally", "assuredly", "beyond question", 
    "evident", "manifest", "irrefutable", "confirmed", "guaranteed",

    # Probable / High Confidence (80% - 60%)
    "likely", "probably", "presumably", "expected", "mostly", 
    "generally", "usually", "mainly", "largely", "predominantly", 
    "in all likelihood", "the chances are", "odds are", "strongly suggests", 
    "apparent", "anticipated", "foreseeable", "credible", "reliable", 
    "tends to", "inclined to", "frequent", "regularly", "often",

    # Neutral / Uncertain / Possible (50% - 40%)
    "perhaps", "maybe", "possibly", "potential", "conceivable", 
    "feasible", "could", "might", "may", "can", "uncertain", 
    "unclear", "undetermined", "up in the air", "conditional", 
    "contingent", "open to question", "debatable", "hypothetical", 
    "theoretical", "speculative", "experimental", "provisional",

    # Hedging / Softeners (The "Shields")
    "it seems", "appears", "suggests", "indicates", "supposedly", 
    "allegedly", "reportedly", "rumored", "purported", "intimated", 
    "it is said", "ostensibly", "presumably", "seemingly", "on the surface",
    "at first glance", "superficially", "to some extent", "somewhat", 
    "rather", "quite", "fairly", "reasonably", "relatively",

    # Personal Opinion / Subjective (Subjective Markers)
    "I believe", "I think", "in my view", "from my perspective", 
    "as far as I know", "to my knowledge", "it strikes me", "I feel", 
    "it is my understanding", "I suspect", "I surmise", "I assume", 
    "I guess", "I suppose", "I gather", "I reckon", "personally",

    # Academic & Professional Qualifiers
    "typically", "characteristically", "virtually", "practically", 
    "nearly", "almost", "effectively", "essentially", "basically", 
    "fundamentally", "broadly", "largely", "in general", "as a rule", 
    "by and large", "for the most part", "to a degree", "in effect",

    # Low Certainty / Doubt (10% - 30%)
    "unlikely", "improbable", "doubtful", "dubious", "questionable", 
    "uncertain", "hesitant", "skeptical", "far-fetched", "implausible", 
    "hardly", "scarcely", "barely", "rarely", "seldom", "occasionally", 
    "infrequent", "unconvincing", "not necessarily", "not entirely",

    # Approximators (Vague Language)
    "about", "around", "approximately", "roughly", "nearly", "close to", 
    "somewhere in the region of", "or so", "something like", "more or less", 
    "an estimated", "upwards of", "roughly speaking", "loosely",

    # Contrastive / Concessive (Hedging via balance)
    "however", "nevertheless", "nonetheless", "on the other hand", 
    "alternatively", "conversely", "despite", "although", "even though", 
    "granted", "admittedly", "while it may be", "notwithstanding",

    # Additions for total count (Reaching 200+)
    "consistent with", "corresponds to", "noted", "observed", "detected",
    "widely considered", "often cited", "arguably", "possibly true",
    "tentatively", "preliminary", "subject to change", "evolving", 
    "speculatively", "theoretically possible", "anecdotal", "unverified", 
    "not yet confirmed", "pending", "interim", "pro tem", "ad interim",
    "provisionally", "under consideration", "being explored", "under review",
    "noted for", "associated with", "linked to", "potentially linked", 
    "suspected", "thought to be", "known to be", "widely known", 
    "universally accepted", "rarely seen", "hardly ever", "most likely", 
    "highly probable", "very likely", "extremely likely", "almost certain", 
    "virtually certain", "nearly impossible", "highly unlikely", 
    "completely", "entirely", "wholly", "fully", "partially", "slightly", 
    "minimally", "marginally", "incrementally", "substantially", 
    "significantly", "noticeably", "markedly", "considerably", 
    "vastly", "immensely", "highly", "deeply", "intensely", "strongly",
    "mildly", "vaguely", "faintly", "obscurely", "ambiguously", 
    "equivocally", "univocally", "decidedly", "explicitly", "implicitly",
    "tacitly", "strictly speaking", "broadly speaking", "technically",
    "formally", "informally", "officially", "unofficially", "notably",
    "principally", "primarily", "especially", "particularly", "specifically",
    "mostly", "chiefly", "predominantly", "largely", "altogether"
]

qwen_lexicon = [
    # High certainty / strong assertion
    "certainly",
    "definitely",
    "absolutely",
    "undoubtedly",
    "unquestionably",
    "clearly",
    "obviously",
    "evidently",
    "plainly",
    "surely",
    "indisputably",
    "incontrovertibly",
    "beyond doubt",
    "without a doubt",
    "without question",
    "for sure",
    "no doubt",
    "it is certain that",
    "it is clear that",
    "it is evident that",
    "it stands to reason that",
    "there is no denying that",
    "it goes without saying that",
    "must",
    "will",
    "shall",

    # Moderate certainty / likely
    "probably",
    "likely",
    "presumably",
    "apparently",
    "seemingly",
    "reportedly",
    "allegedly",
    "supposedly",
    "ostensibly",
    "plausibly",
    "reasonably",
    "fairly",
    "quite",
    "rather",
    "more than likely",
    "in all likelihood",
    "it seems that",
    "it appears that",
    "it is probable that",
    "it is likely that",
    "there is reason to believe that",
    "one might argue that",
    "would",
    "should",
    "ought to",

    # Low certainty / uncertainty / possibility
    "possibly",
    "perhaps",
    "maybe",
    "conceivably",
    "theoretically",
    "hypothetically",
    "potentially",
    "might",
    "could",
    "may",
    "it is possible that",
    "it could be that",
    "it might be that",
    "there is a chance that",
    "there is some possibility that",
    "not impossible",
    "cannot be ruled out",
    "remains uncertain",
    "remains unclear",
    "is open to question",
    "is debatable",
    "is speculative",
    "is tentative",

    # Hedging verbs and verb phrases
    "suggest",
    "indicate",
    "imply",
    "point to",
    "appear to",
    "seem to",
    "tend to",
    "may suggest",
    "could indicate",
    "might imply",
    "appears to support",
    "seems consistent with",
    "is consistent with",
    "lends support to",
    "raises the possibility that",
    "hints at",
    "alludes to",
    "proposes",
    "postulates",
    "hypothesizes",
    "assumes",
    "presumes",
    "speculates",
    "conjectures",

    # Qualifiers and downtoners
    "somewhat",
    "slightly",
    "marginally",
    "relatively",
    "comparatively",
    "to some extent",
    "to a certain degree",
    "in part",
    "partially",
    "not entirely",
    "not completely",
    "not fully",
    "hardly",
    "scarcely",
    "barely",
    "almost",
    "nearly",
    "virtually",
    "practically",
    "more or less",
    "roughly",
    "approximately",
    "about",
    "around",
    "circa",
    "give or take",

    # Expressions of caution or limitation
    "with caution",
    "with reservations",
    "with some caveats",
    "subject to",
    "depending on",
    "contingent on",
    "provided that",
    "assuming that",
    "if true",
    "if accurate",
    "as far as we know",
    "to the best of our knowledge",
    "based on current evidence",
    "given the available data",
    "within the limits of",
    "within the scope of",
    "in the context of",
    "under these conditions",
    "at this stage",
    "at present",
    "currently",
    "so far",
    "up to now",
    "preliminarily",
    "tentatively",
    "provisionally",

    # Academic/professional hedging phrases
    "it could be argued that",
    "one interpretation is that",
    "this may be interpreted as",
    "a possible explanation is",
    "an alternative view is",
    "from one perspective",
    "according to some scholars",
    "some researchers maintain that",
    "there is evidence to suggest",
    "data appear to show",
    "findings seem to indicate",
    "results are suggestive of",
    "this lends credence to",
    "while not conclusive",
    "although further research is needed",
    "pending additional evidence",
    "this remains to be seen",
    "the picture is still incomplete",
    "our understanding is limited by",
    "methodological constraints prevent us from concluding",
    "it is premature to conclude",
    "we cannot exclude the possibility that",
    "it is not unreasonable to suppose that",
    "this raises questions about",
    "further investigation is warranted",

    # Modal adverbs and related
    "conceivably",
    "feasibly",
    "imaginably",
    "arguably",
    "admittedly",
    "granted",
    "of course",
    "naturally",
    "understandably",
    "predictably",
    "unsurprisingly",

    # Doubt and skepticism markers
    "doubtful",
    "questionable",
    "uncertain",
    "ambiguous",
    "equivocal",
    "vague",
    "speculative",
    "controversial",
    "disputed",
    "contentious",
    "problematic",
    "unverified",
    "unconfirmed",
    "unsubstantiated",
    "anecdotal",
    "inconclusive",
    "inconsistent",
    "contradictory",
    "it is doubtful that",
    "it is questionable whether",
    "there is little evidence that",
    "no clear evidence supports",
    "fails to demonstrate",
    "does not conclusively show"
]

all_hedging_words = list(set(claude_lexicon + gemini_lexicon + gpt_lexicon + qwen_lexicon))

# Generate example sentences using vLLM
import pandas as pd
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
        max_tokens=512,
        n=20,  # Generate 20 completions per prompt
        stop=["\n\n", "Example:", "Sentence:"],
    )
    
    results = []
    
    # Build chat messages for all hedging words
    messages_batch = []
    hedging_words_batch = []
    
    for word in tqdm(all_hedging_words, desc="Preparing prompts"):
        messages = [
            {
                "role": "user",
                "content": f"""Generate a single non-verifiable opinion or subjective statement that naturally contains the phrase "{word}". 
                The sentence should express confidence, uncertainty, speculation, or personal opinion that cannot be objectively verified. 
                Do not use other hedging words or phrases in the sentence. Do not include labels like "Example:" or "Sentence:". 
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

