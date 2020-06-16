import mwapi
from revscoring import Model
from revscoring.extractors.api.extractor import Extractor
from revscoring.errors import RevisionNotFound
from revscoring.errors import TextDeleted
import pandas as pd
import numpy as np

#enwiki.goodfaith.gradient_boosting.model
#enwiki.damaging.gradient_boosting.model
with open("models/enwiki.damaging.gradient_boosting.model") as f:
       scorer_model = Model.load(f)

extractor = Extractor(mwapi.Session(host="https://en.wikipedia.org",
                                          user_agent="revscoring demo"))

def get_score(rev_id):
    feature_values = list(extractor.extract(rev_id, scorer_model.features))
    results = scorer_model.score(feature_values)
    return results

df = pd.read_csv("data.csv")
df["label_damage"] = ""
df["confidence_damage"] = ""

for i in range(len(df["rev_id"])):
    print(str(i) + "/" + str(len(df["rev_id"])))

    try:
        results = get_score(df["rev_id"][i])
        df["label_damage"][i] = results["prediction"]
        df["confidence_damage"][i] = results["probability"][True]
    except (RevisionNotFound, TextDeleted, KeyError) as err:
        print("revision not found!")
        continue

df.to_csv("data.csv", index=False)