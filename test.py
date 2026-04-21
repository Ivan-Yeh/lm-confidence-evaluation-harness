import pandas as pd
import os

human_annotated_cues = pd.read_csv(os.path.join("linguistic_confidence_lexicon", "hedging_word_aggregated.csv"))[["hedging_word", "mean", "std"]]
human_annotated_cues["mean"] *= 100.0
human_annotated_cues["std"] *= 100.0
human_annotated_cues = human_annotated_cues.sort_values("mean").round(2).to_dict(orient="records")
print(human_annotated_cues)