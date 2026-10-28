"""Train both ML parts and save them.   Run:  python -m src.train [--backend classifier|classifier_embed]"""
import argparse

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline, make_union

from . import config
from .matcher import Matcher
from .preprocess import clean


def intent_model():
    feats = make_union(TfidfVectorizer(ngram_range=(1, 2)),
                       TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5)))   # char n-grams handle typos
    return make_pipeline(feats, LogisticRegression(max_iter=2000, class_weight="balanced"))


def main(backend):
    config.MODELS.mkdir(exist_ok=True)
    config.REPORTS.mkdir(exist_ok=True)
    intents = pd.read_csv(config.DATA / "intents.csv")
    q = pd.read_csv(config.DATA / "questions.csv")
    kb = pd.read_csv(config.DATA / "knowledge_base.csv")

    # A. safety / intent classifier.
    # The example questions are also normal SRH questions, so we add them to the "srh_question" class.
    X_int, y_int = intents.text.map(clean).tolist(), intents.label.tolist()
    seen = set(X_int)                                    # do not add the same text twice
    X_srh = [x for x in dict.fromkeys(q.question.map(clean)) if x not in seen]
    X_all, y_all = X_int + X_srh, y_int + ["srh_question"] * len(X_srh)
    n_int = len(X_int)
    # cross-validation: score only the rows of intents.csv; the extra questions are always training data
    pred = [None] * n_int
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X_int, y_int):
        Xtr = [X_int[i] for i in tr] + X_srh
        ytr = [y_int[i] for i in tr] + ["srh_question"] * len(X_srh)
        m = intent_model().fit(Xtr, ytr)
        for i, p in zip(te, m.predict([X_int[i] for i in te])):
            pred[i] = p
    report = classification_report(y_int, pred, digits=2)
    (config.REPORTS / "intent_cv.txt").write_text(report)
    print("5-fold cross-validation of the intent classifier:\n" + report)
    joblib.dump(intent_model().fit(X_all, y_all), config.MODELS / "intent.joblib")

    # B. question matcher: example questions + each entry's title and answer
    matcher = Matcher(backend).fit(q.question, q.entry_id,
                                   extra_texts=list(kb.title) + list(kb.answer), extra_ids=list(kb.id) * 2)
    joblib.dump(matcher, config.MODELS / "matcher.joblib")
    print(f"Saved models to {config.MODELS}  (matcher backend: {backend})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=config.BACKEND)
    main(ap.parse_args().backend)
