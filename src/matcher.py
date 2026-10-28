"""Question matcher: a small supervised classifier that maps a question to a knowledge-base entry.

It is trained on the example questions (questions.csv) plus each entry's title and answer text,
so it also learns the key words of every answer. The predicted probability is the confidence score.

Backends:
  "classifier"        TF-IDF words + character n-grams -> Logistic Regression   (default, works anywhere)
  "classifier_embed"  the same, plus sentence-embedding features (Day 6 upgrade; needs sentence-transformers)
"""
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_union

from .config import EMBED_MODEL
from .preprocess import clean


class Matcher:
    def __init__(self, backend="classifier"):
        self.backend = backend
        self._st = None                          # sentence-transformer, loaded lazily

    def _embed(self, texts):
        if self._st is None:
            from sentence_transformers import SentenceTransformer   # optional dependency
            self._st = SentenceTransformer(EMBED_MODEL)
        return self._st.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def _features(self, texts, fit=False):
        texts = [clean(t) for t in texts]
        X = self.tfidf.fit_transform(texts) if fit else self.tfidf.transform(texts)
        if self.backend == "classifier_embed":
            X = hstack([X, csr_matrix(self._embed(texts))]).tocsr()
        return X

    def fit(self, questions, entry_ids, extra_texts=(), extra_ids=()):
        """questions/entry_ids = example questions; extra_* = titles and answers with their entry ids."""
        texts, ids = list(questions) + list(extra_texts), list(entry_ids) + list(extra_ids)
        self.tfidf = make_union(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
                                TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True))
        self.clf = LogisticRegression(C=10, max_iter=3000)
        self.clf.fit(self._features(texts, fit=True), ids)
        return self

    def top_batch(self, texts, k=3):
        """For each text return the k most likely (entry_id, probability) pairs, best first."""
        P = self.clf.predict_proba(self._features(list(texts)))
        out = []
        for p in P:
            order = np.argsort(-p)[:k]
            out.append([(str(self.clf.classes_[i]), float(p[i])) for i in order])
        return out

    def top(self, text, k=3):
        return self.top_batch([text], k)[0]

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_st"] = None                          # never pickle the big embedding model
        return d
