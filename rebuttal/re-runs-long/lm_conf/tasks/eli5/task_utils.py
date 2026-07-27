import os
import pandas as pd
from ...default_utils.custom_types import ModelOutputs, OrganisedOutputs, PromptCollection
from ...default_utils.datasets_manager import DatasetsManager

# ELI5 (sentence-transformers/eli5) only ships a "train" split with no
# question/answer partitioning, and it is far too large (325k rows) to run
# in full. DatasetsManager.get_dataset() only random-samples when `limit` is
# set, so we keep `limit: null` in the task yaml and instead truncate here to
# the first N rows in original (HF-stored) order -- i.e. a deterministic
# "first 800 questions" slice rather than a random subsample.
# ELI5_NUM_QUESTIONS env var override is for smoke-testing only; the actual
# rerun always uses the default of 800.
NUM_QUESTIONS = int(os.environ.get("ELI5_NUM_QUESTIONS", "800"))


def preprocess_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the ELI5 (sentence-transformers/eli5) dataset: keep only the
    first NUM_QUESTIONS rows (in original dataset order). Columns are already
    named "question" and "answer", matching direct_free_form_qa's expectations.
    """
    return dataset.iloc[:NUM_QUESTIONS].reset_index(drop=True)
