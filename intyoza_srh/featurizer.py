"""Text -> TF-IDF feature vector, in pure Python + NumPy (no scikit-learn needed at run time).

Features: word 1-2 grams and character 3-5 grams inside word boundaries (char n-grams make the
model robust to typos and SMS spelling). Sublinear term frequency, IDF weighting, L2 norm."""
import json
import math
import re
from collections import Counter

import numpy as np


def clean(text: str) -> str:
    """Lowercase, remove symbols, collapse spaces. Keeps apostrophes and letters."""
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def grams(text: str):
    """All feature strings of one (already cleaned) text."""
    words = text.split()
    out = ["w:" + w for w in words]
    out += ["b:" + a + " " + b for a, b in zip(words, words[1:])]
    for w in words:
        p = f" {w} "
        for n in (3, 4, 5):
            if len(p) < n:
                break
            out += ["c:" + p[i:i + n] for i in range(len(p) - n + 1)]
    return out


class Featurizer:
    def __init__(self, vocab: dict, idf):
        self.vocab = vocab
        self.idf = np.asarray(idf, dtype=np.float32)

    @property
    def dim(self):
        return len(self.vocab)

    @classmethod
    def fit(cls, texts, max_features=40000, min_df=2):
        df = Counter()
        for t in texts:
            df.update(set(grams(clean(t))))
        kept = [g for g, c in df.most_common() if c >= min_df][:max_features]
        n = len(texts)
        vocab = {g: i for i, g in enumerate(sorted(kept))}
        idf = np.zeros(len(vocab), dtype=np.float32)
        for g, i in vocab.items():
            idf[i] = math.log((1 + n) / (1 + df[g])) + 1.0
        return cls(vocab, idf)

    def _row(self, text):
        counts = Counter(i for i in (self.vocab.get(g) for g in grams(clean(text))) if i is not None)
        if not counts:
            return np.zeros(0, np.int32), np.zeros(0, np.float32)
        idx = np.fromiter(counts.keys(), np.int32, len(counts))
        val = np.fromiter((1.0 + math.log(c) for c in counts.values()), np.float32, len(counts)) * self.idf[idx]
        return idx, val / (np.linalg.norm(val) or 1.0)

    def transform_csr(self, texts):
        """Return (indptr, indices, data) of a CSR matrix; used for training on many rows."""
        indptr, indices, data = [0], [], []
        for t in texts:
            idx, val = self._row(t)
            indices.append(idx)
            data.append(val)
            indptr.append(indptr[-1] + len(idx))
        cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt)
        return np.array(indptr, np.int64), cat(indices, np.int32), cat(data, np.float32)

    def transform(self, texts):
        """Dense float32 matrix (rows x dim); fine for a few texts at a time."""
        X = np.zeros((len(texts), self.dim), np.float32)
        for r, t in enumerate(texts):
            idx, val = self._row(t)
            X[r, idx] = val
        return X

    def to_json(self):
        return json.dumps({"vocab": self.vocab, "idf": self.idf.round(5).tolist()})

    @classmethod
    def from_json(cls, s):
        d = json.loads(s)
        return cls(d["vocab"], d["idf"])
