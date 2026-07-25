import pandas as pd
from ...default_utils.custom_types import ModelOutputs, OrganisedOutputs, PromptCollection
from ...default_utils.datasets_manager import DatasetsManager
import numpy as np
import sacrebleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

def preprocess_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset["answer"] = dataset["answers"].apply(lambda x: x['text'][0] if isinstance(x, dict) and 'text' in x and len(x['text']) > 0 else "Unanswerable based on the context. If the response implies that it is unable to answer specifically due to a lack of information provided, grade it as CORRECT.")
    return dataset

