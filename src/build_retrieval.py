"""Add semantic search to the trained model (runs at the end of src/train_keras.py; can also run alone).
Run:  python -m src.build_retrieval

The same pretrained sentence encoder turns every training question and every title into a "meaning" vector.
At chat time the user's question is compared with all of them (cosine similarity). This helps the model with:
  * questions worded in ways it never saw (blended with the classifier: classifier_weight * p + rest * search),
  * crisis messages in new words (nearest examples are crisis messages -> safe reply),
  * broad or unusual questions (the closest answers are always offered as suggestions).
The vectors are saved inside intyoza_srh_model.h5, and the Keras model gets a third output "sentence_vector"."""
import json
import os
import sys
import tempfile

import h5py
import numpy as np
import pandas as pd

from . import config

sys.path.insert(0, str(config.ROOT))
from intyoza_srh.featurizer import clean  # noqa: E402

H5 = config.ROOT / "intyoza_srh" / "model" / "intyoza_srh_model.h5"
CLASSIFIER_WEIGHT, TEMPERATURE = 0.7, 0.04          # chosen on validation questions (see README)
# semantic crisis check: the closest examples must be crisis messages (mean of top k >= floor and above SRH /
# off-topic by margin), and the classifier must give the label at least min_model_prob unless the gap is >= strong_gap.
# Chosen on the red-team set and the validation questions; the test set is untouched.
KNN = {"k": 5, "floor": 0.45, "margin": 0.0, "min_model_prob": 0.002, "strong_gap": 0.05}
GUESS_MIN_ACCURACY = 0.5    # below the answer threshold, show a "best guess" when it is right at least this often
CRISIS = ["self_harm", "abuse_disclosure", "medical_emergency", "diagnosis_request"]


def blend(p_ans, sims, entry_ix, n_ans, w=CLASSIFIER_WEIGHT, tau=TEMPERATURE):
    """Classifier probabilities blended with semantic-search scores (best match per answer)."""
    K = np.full((len(sims), n_ans), -1.0, np.float32)
    for r in range(len(sims)):
        np.maximum.at(K[r], entry_ix, sims[r])
    Ks = np.exp((K - K.max(1, keepdims=True)) / tau)
    Ks /= Ks.sum(1, keepdims=True)
    return w * p_ans + (1 - w) * Ks


def main():
    import tensorflow as tf
    from .train_keras import build_data, encode_all
    from intyoza_srh.chatbot import IntyozaChatbot

    bot = IntyozaChatbot(H5, backend="keras")
    model = bot._net
    enc = tf.keras.Model(model.inputs[1:], model.get_layer("sentence_vector").output)   # inputs[0] is the tf-idf vector
    tok = bot.tokenizer
    labels = bot.answer_labels
    ix = {e: i for i, e in enumerate(labels)}

    q = pd.read_csv(config.DATA / "questions.csv")
    q = q[q.entry_id.isin(ix)].drop_duplicates("question")
    kb = pd.read_csv(config.DATA / "knowledge_base.csv")
    it = pd.read_csv(config.DATA / "intents.csv")

    def vecs(texts):
        return encode_all(enc, tok, [clean(t) for t in texts])

    # 1. answer threshold for the blended score, on validation questions (left out of the search bank here)
    d = build_data(0)
    nq = d["n_val_q"]
    val_txt = list(d["Xv"][:nq])
    qb = q[~q.question.isin(set(val_txt))]
    B = np.concatenate([vecs(qb.question), vecs(kb.title)])
    B_ent = np.array([ix[e] for e in qb.entry_id] + [ix[e] for e in kb.id])
    _, p_val, _ = bot.predict(val_txt)
    s = blend(p_val, vecs(val_txt) @ B.T, B_ent, len(labels))
    top, right = s.max(1), s.argmax(1) == d["yv_ans"][:nq]
    print(f"validation top-1: classifier {np.mean(p_val.argmax(1) == d['yv_ans'][:nq]):.3f} -> blended {right.mean():.3f}")
    best_th, best_score = 0.3, -1e9
    for th in np.arange(0.05, 0.95, 0.05):
        ans = top >= th
        score = (right & ans).sum() - 2 * (~right & ans).sum()     # a wrong answer costs twice a "not sure"
        if score > best_score:
            best_th, best_score = float(th), score
    print(f"blended answer threshold: {best_th:.2f} (answers {np.mean(top >= best_th):.2f} of validation questions, "
          f"{right[top >= best_th].mean():.3f} of them right)")

    # calibration: score -> how often an answer with that score was right on validation questions.
    # This is the "accuracy" shown next to each answer. Bins of equal size, then made non-decreasing.
    order = np.argsort(top)
    bins = np.array_split(order, 12)
    pts = [[float(top[b].mean()), float(right[b].mean())] for b in bins]
    for i in range(1, len(pts)):
        pts[i][1] = max(pts[i][1], pts[i - 1][1])
    calib = [[round(x, 4), round(y, 4)] for x, y in pts]
    guess_th = next((x for x, y in calib if y >= GUESS_MIN_ACCURACY), best_th)
    guess_th = min(guess_th, best_th)
    print("calibration (score -> accuracy):", calib)
    print(f"best-guess threshold: {guess_th:.3f}")

    # 2. final search banks: all training questions + titles; crisis / off-topic examples from intents.csv
    A = np.concatenate([vecs(q.question), vecs(kb.title)]).astype(np.float16)
    A_ent = np.array([ix[e] for e in q.entry_id] + [ix[e] for e in kb.id], np.int32)
    I = vecs(it.text).astype(np.float16)
    I_lab = np.array([bot.intent_labels.index(x) for x in it.label], np.int32)

    cfg = dict(bot.cfg)
    cfg["answer_threshold"] = round(best_th, 2)
    cfg["suggest_threshold"] = round(max(0.05, best_th / 3), 3)
    cfg["search"] = {"classifier_weight": CLASSIFIER_WEIGHT, "temperature": TEMPERATURE, **KNN,
                     "suggest_min_similarity": 0.35}
    cfg["calibration"] = calib
    cfg["guess_threshold"] = round(float(guess_th), 3)

    # 3. Keras model with a third output (the sentence vector), saved with all assets
    full = tf.keras.Model(model.inputs, list(model.outputs[:2]) + [model.get_layer("sentence_vector").output],
                          name="intyoza_srh")
    fd, tmp = tempfile.mkstemp(suffix=".h5", dir=H5.parent)
    os.close(fd)
    full.save(tmp)
    with h5py.File(H5, "r") as src, h5py.File(tmp, "a") as dst:
        src.copy("intyoza", dst)
        g = dst["intyoza"]
        del g["config"]
        g.create_dataset("config", data=np.bytes_(json.dumps(cfg).encode("utf-8")))
        for k, v in (("search_vectors", A), ("search_entry", A_ent), ("intent_vectors", I), ("intent_labels", I_lab)):
            if k in g:
                del g[k]
            g.create_dataset(k, data=v, compression="gzip")
        g.attrs["format"] = "intyoza-srh-3"
    os.replace(tmp, H5)
    (H5.parent / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"Saved {H5} with semantic search ({len(A)} questions, {len(I)} intent examples), "
          f"{H5.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
