"""Train the Keras chatbot model and save it as ONE .h5 file inside the intyoza_srh package.
Run:  python -m src.train_keras            (needs tensorflow; see requirements.txt)

Network (one Keras model, built-in layers only):
  question --WordPiece--> MiniLM sentence encoder (6-layer transformer, pretrained, frozen) -> 384-d vector -> Dense(768)
  question --tf-idf (word 1-2 grams + char 3-5 grams)-------------------------------------------------> Dense(256)
  answer branch: concat -> "answer" softmax (one class per knowledge-base entry)
  intent branch (own layers, class-balanced): sentence vector + Dense(64) on tf-idf -> Dense(256) -> "intent" softmax
     (srh_question / off_topic / self_harm / abuse_disclosure / medical_emergency / diagnosis_request)
The .h5 also stores (group "intyoza") the vocabularies, labels, knowledge base, fixed replies and settings,
so intyoza_srh/model/intyoza_srh_model.h5 is everything another project needs."""
import argparse
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from . import config
from .augment import augment

ROOT = config.ROOT
sys.path.insert(0, str(ROOT))
from intyoza_srh.encoder import ENCODER_INPUTS, HIDDEN, WordPiece, build_keras_encoder, load_hf_weights  # noqa: E402
from intyoza_srh.featurizer import Featurizer, clean  # noqa: E402

PKG_MODEL = ROOT / "intyoza_srh" / "model"
H5 = PKG_MODEL / "intyoza_srh_model.h5"
INTENTS = ["srh_question", "off_topic", "self_harm", "abuse_disclosure", "medical_emergency", "diagnosis_request"]
ENCODER_REPO = "sentence-transformers/all-MiniLM-L6-v2"   # downloaded once, then cached offline
ROWS_PER_ENTRY = 48           # each answer gets about this many training texts after augmentation
STARTERS = ["What is a period", "Pregnancy the first time and every time", "HIV testing", "What is consent",
            "Emergency contraceptive pills", "Signs of an STI"]


def build_data(seed=0):
    rng = random.Random(seed)
    kb = pd.read_csv(config.DATA / "knowledge_base.csv").fillna("")
    q = pd.read_csv(config.DATA / "questions.csv")
    it = pd.read_csv(config.DATA / "intents.csv")
    q = q[q.entry_id.isin(set(kb.id))]
    labels = list(kb.id)
    lab_ix = {e: i for i, e in enumerate(labels)}

    # validation: one original question per entry (never trained on), used for early stopping and thresholds
    val_idx = q.groupby("entry_id").apply(lambda g: g.sample(1, random_state=seed).index[0], include_groups=False)
    val_q, train_q = q.loc[val_idx.values], q.drop(index=val_idx.values)

    ans_texts, ans_y = [], []
    for eid, g in train_q.groupby("entry_id"):
        texts = list(g.question)
        n_aug = max(0, math.ceil((ROWS_PER_ENTRY - len(texts)) / max(1, len(texts))))
        variants = augment(texts, n_aug, seed=rng.randrange(10**9)) if n_aug else [[] for _ in texts]
        rows = texts + [v for vs in variants for v in vs]
        ans_texts += rows
        ans_y += [lab_ix[eid]] * len(rows)
    n_questions = len(ans_texts)
    for _, e in kb.iterrows():
        # the title is a good short query, and each answer sentence carries the words people ask with
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", e.answer) if len(s.split()) >= 4]
        rows = [e.title, e.title.lower() + "?"] + sents
        ans_texts += rows
        ans_y += [lab_ix[e.id]] * len(rows)
    w_text = [1.0] * n_questions + [0.5] * (len(ans_texts) - n_questions)

    it_val = it.sample(frac=0.15, random_state=seed)
    it_tr = it.drop(index=it_val.index)
    int_ix = {k: i for i, k in enumerate(INTENTS)}

    X = ans_texts + list(it_tr.text)
    y_int = [0] * len(ans_texts) + [int_ix[x] for x in it_tr.label]
    y_ans = ans_y + [0] * len(it_tr)
    w_ans = w_text + [0.0] * len(it_tr)                         # intent-only rows teach no answer
    counts = np.bincount(y_int, minlength=len(INTENTS))
    cw = len(y_int) / (len(INTENTS) * np.maximum(counts, 1))
    cw = cw / cw.min()               # full class balance: each crisis class counts as much as all SRH questions
    w_int = [float(cw[c]) for c in y_int]

    Xv = list(val_q.question) + list(it_val.text)
    yv_int = [0] * len(val_q) + [int_ix[x] for x in it_val.label]
    yv_ans = [lab_ix[e] for e in val_q.entry_id] + [0] * len(it_val)
    wv_ans = [1.0] * len(val_q) + [0.0] * len(it_val)
    return dict(kb=kb, labels=labels, X=X, y_int=np.array(y_int), y_ans=np.array(y_ans), w_int=np.array(w_int, np.float32),
                w_ans=np.array(w_ans, np.float32), Xv=Xv, yv_int=np.array(yv_int), yv_ans=np.array(yv_ans),
                wv_ans=np.array(wv_ans, np.float32), n_val_q=len(val_q))


def encoder_files():
    """Original all-MiniLM-L6-v2 files (downloaded once from Hugging Face, then cached offline)."""
    from huggingface_hub import hf_hub_download
    return {f: hf_hub_download(ENCODER_REPO, f) for f in ("vocab.txt", "model.safetensors")}


def encode_all(enc_model, tokz, texts, bs=256):
    """Sentence vectors for many texts; batches of similar length so there is little padding."""
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), HIDDEN), np.float32)
    for i in range(0, len(texts), bs):
        ix = order[i:i + bs]
        out[ix] = np.asarray(enc_model.predict_on_batch(tokz.batch([texts[j] for j in ix])))
    return out


def main(epochs=40, hidden=256, max_features=40000, seed=0):
    import tensorflow as tf
    from scipy.sparse import csr_matrix
    keras = tf.keras
    tf.keras.utils.set_random_seed(seed)

    d = build_data(seed)
    print(f"answers: {len(d['labels'])} | training texts: {len(d['X'])} | validation texts: {len(d['Xv'])}")
    tr_txt, va_txt = [clean(x) for x in d["X"]], [clean(x) for x in d["Xv"]]
    feat = Featurizer.fit(tr_txt, max_features=max_features, min_df=2)
    print(f"tf-idf features: {feat.dim}")

    def csr(texts):
        indptr, indices, data = feat.transform_csr(texts)
        return csr_matrix((data, indices, indptr), shape=(len(texts), feat.dim))

    Xtr, Xva = csr(tr_txt), csr(va_txt)

    # 1. frozen pretrained sentence encoder -> one 384-d vector per text (computed once)
    files = encoder_files()
    vocab_text = Path(files["vocab.txt"]).read_text(encoding="utf-8")
    tokz = WordPiece(vocab_text.splitlines())
    enc_inputs, enc_vec = build_keras_encoder(keras)
    enc_model = keras.Model(enc_inputs, enc_vec)
    load_hf_weights(enc_model, files["model.safetensors"])
    cache = config.MODELS / "sentence_vectors_cache.npz"      # same texts -> reuse the vectors (saves minutes)
    key = hashlib.md5("\n".join(tr_txt + ["---"] + va_txt).encode("utf-8")).hexdigest()
    if cache.exists() and str(np.load(cache)["key"]) == key:
        z = np.load(cache)
        Vtr, Vva = z["tr"], z["va"]
    else:
        print("encoding training texts with the sentence encoder ...", flush=True)
        Vtr, Vva = encode_all(enc_model, tokz, tr_txt), encode_all(enc_model, tokz, va_txt)
        config.MODELS.mkdir(exist_ok=True)
        np.savez(cache, tr=Vtr, va=Vva, key=key)

    def to_sparse(m):
        c = m.tocoo()
        st = tf.SparseTensor(np.stack([c.row, c.col], 1).astype(np.int64), c.data.astype(np.float32), c.shape)
        return tf.sparse.reorder(st)

    class Batches(keras.utils.PyDataset):
        """Sparse batches for one output. With feature_drop > 0, random words / n-grams are removed (input dropout)."""

        def __init__(self, rows, X, V, y, w, bs, feature_drop=0.0):
            super().__init__()
            self.X, self.V, self.y, self.w, self.bs, self.p = X, V, y, w, bs, feature_drop
            self.order = np.array(rows)
            self.on_epoch_end()

        def __len__(self):
            return math.ceil(len(self.order) / self.bs)

        def __getitem__(self, i):
            ix = self.order[i * self.bs:(i + 1) * self.bs]
            m = self.X[ix].copy()
            if self.p:
                m.data *= (np.random.random(m.data.shape) > self.p) / (1 - self.p)
                m.eliminate_zeros()
            return (to_sparse(m), self.V[ix]), self.y[ix], self.w[ix]

        def on_epoch_end(self):
            if self.p:
                np.random.shuffle(self.order)

    n_ans = len(d["labels"])
    L = keras.layers

    # Two separate branches on the same inputs, so neither task weakens the other:
    def answer_branch(f, v):
        h1 = L.Dense(hidden, activation="relu", name="hidden")(f)
        h2 = L.Dense(768, activation="relu", name="embed_hidden")(L.GaussianNoise(0.02)(v))
        h = L.Dropout(0.5)(L.Concatenate()([h1, h2]))
        return L.Dense(n_ans, activation="softmax", name="answer")(h)

    def intent_branch(f, v):
        # mostly the sentence vector (generalises to new wordings); a small tf-idf part helps with SMS spelling
        t = L.Dense(64, activation="relu", name="intent_tfidf")(f)
        h = L.Dense(256, activation="relu", name="intent_hidden")(L.Concatenate()([L.Dropout(0.5)(t), L.GaussianNoise(0.03)(v)]))
        return L.Dense(len(INTENTS), activation="softmax", name="intent")(L.Dropout(0.4)(h))

    def heads_model(branch, name):
        f_in = L.Input((feat.dim,), sparse=True, name="features")
        v_in = L.Input((HIDDEN,), name="sentence_vector")
        m = keras.Model([f_in, v_in], branch(f_in, v_in), name=name)
        m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy")
        return m

    nq = d["n_val_q"]
    val_q = (to_sparse(Xva[:nq]), Vva[:nq])
    rows_all = np.arange(Xtr.shape[0])

    # 2a. answer branch: trained only on rows that belong to an answer
    ans_model = heads_model(answer_branch, "answer_branch")

    class ValTop1(keras.callbacks.Callback):
        """True top-1 accuracy on the held-out validation questions (logged as val_top1)."""

        def on_epoch_end(self, epoch, logs=None):
            p = self.model.predict(val_q, verbose=0, batch_size=1024)
            logs["val_top1"] = float((p.argmax(1) == d["yv_ans"][:nq]).mean())
            print(f"  epoch {epoch + 1}: val_top1 = {logs['val_top1']:.4f}", flush=True)

    ans_rows = rows_all[d["w_ans"] > 0]
    ans_model.fit(Batches(ans_rows, Xtr, Vtr, d["y_ans"], d["w_ans"], 256, feature_drop=0.25), epochs=epochs, verbose=2,
                  callbacks=[ValTop1(),
                             keras.callbacks.EarlyStopping(monitor="val_top1", mode="max", patience=5, restore_best_weights=True),
                             keras.callbacks.ReduceLROnPlateau(monitor="val_top1", mode="max", factor=0.5, patience=2)])

    # 2b. intent branch: all rows, each class balanced (crisis classes count as much as all SRH questions)
    int_model = heads_model(intent_branch, "intent_branch")
    int_model.fit(Batches(rows_all, Xtr, Vtr, d["y_int"], d["w_int"], 256, feature_drop=0.25), epochs=6, verbose=2)

    # confidence threshold from validation questions: answer only when the top probability is high enough
    p_ans = ans_model.predict(val_q, verbose=0, batch_size=1024)
    top, best = p_ans.max(1), p_ans.argmax(1)
    right = best == d["yv_ans"][:nq]
    print(f"validation top-1 accuracy: {right.mean():.3f}")
    print("threshold | answered | right among answered")
    best_th, best_score = 0.3, -1e9
    for th in np.arange(0.05, 0.95, 0.05):
        ans = top >= th
        score = (right & ans).sum() - 2 * (~right & ans).sum()     # a wrong answer costs twice a "not sure"
        print(f"  {th:.2f}    | {ans.mean():.2f}     | {right[ans].mean() if ans.any() else 0:.3f}")
        if score > best_score:
            best_th, best_score = float(th), score
    print(f"chosen answer threshold: {best_th:.2f}")

    # crisis thresholds (a false alarm shows a caring message with help lines; a miss can cost a life):
    #  main bar      = the highest bar (never above config.SAFE_MIN_PROB) that catches >= 97% of held-out crisis rows
    #  uncertain bar = a lower bar used only when no answer is confident; kept so that it adds at most
    #                  0.5% false alarms per class on the hand-written validation SRH questions
    P_int = int_model.predict((to_sparse(Xva), Vva), verbose=0, batch_size=1024)
    yv = d["yv_int"]
    unsure = np.zeros(len(yv), bool)
    unsure[:nq] = top < best_th
    safe_min, safe_unsure = {}, {}
    for lab, cap in config.SAFE_MIN_PROB.items():
        c = INTENTS.index(lab)
        p_pos = P_int[yv == c, c]
        th = float(np.clip(np.quantile(p_pos, 0.03), 0.05, cap)) if len(p_pos) else cap
        tu = th
        for cand in np.arange(0.02, th, 0.01):
            if np.mean(unsure[:nq] & (P_int[:nq, c] >= cand) & (P_int[:nq, c] < th)) <= 0.005:
                tu = float(cand)
                break
        safe_min[lab], safe_unsure[lab] = round(th, 3), round(tu, 3)
        print(f"  {lab}: bar {th:.3f} (recall {np.mean(p_pos >= th):.3f}, false alarms {np.mean(P_int[:nq, c] >= th):.3f})"
              f" | bar when unsure {tu:.3f}")

    # development check on the red-team set (the test set stays untouched for the final report)
    from intyoza_srh.safety import rule_intent, wants_refusal
    red = pd.read_csv(config.DATA / "redteam_prompts.csv")
    red = red[red.expected_intent.isin(list(config.SAFE_MIN_PROB))]
    rt = [clean(x) for x in red.prompt]
    keep = [i for i, t in enumerate(rt) if not rule_intent(t) and not wants_refusal(t)]
    rt = [rt[i] for i in keep]
    exp = red.expected_intent.values[keep]
    Xr = to_sparse(csr(rt))
    Vr = encode_all(enc_model, tokz, rt)
    pi = int_model.predict((Xr, Vr), verbose=0, batch_size=1024)
    pa = ans_model.predict((Xr, Vr), verbose=0, batch_size=1024).max(1)
    got = []
    for k in range(len(rt)):
        lab = next((l for l in safe_min if pi[k, INTENTS.index(l)] >= safe_min[l]), None)
        if lab is None and pa[k] < best_th:
            lab = next((l for l in safe_unsure if pi[k, INTENTS.index(l)] >= safe_unsure[l]), None)
        got.append(lab)
    print("red-team crisis prompts not caught by keyword rules (dev check): "
          + ", ".join(f"{l} {np.mean([g == l for g, e in zip(got, exp) if e == l]):.3f}" for l in safe_min))

    # 3. one complete model: question inputs -> encoder + tf-idf -> both branches (dense tf-idf input, loads anywhere)
    f_dense = L.Input((feat.dim,), name="features")
    enc_inputs, enc_vec = build_keras_encoder(keras)
    full = keras.Model([f_dense] + [enc_inputs[k] for k in ENCODER_INPUTS],
                       [intent_branch(f_dense, enc_vec), answer_branch(f_dense, enc_vec)], name="intyoza_srh")
    load_hf_weights(full, files["model.safetensors"])
    for src_model, names in ((ans_model, ("hidden", "embed_hidden", "answer")),
                             (int_model, ("intent_tfidf", "intent_hidden", "intent"))):
        for name in names:
            full.get_layer(name).set_weights(src_model.get_layer(name).get_weights())

    cfg = {"answer_threshold": round(best_th, 2), "suggest_threshold": round(max(0.05, best_th / 3), 3),
           "safe_min_prob": safe_min, "safe_min_prob_when_unsure": safe_unsure,
           "offtopic_min_prob": config.OFFTOPIC_MIN_PROB, "require_reviewed": False,
           "starter_questions": STARTERS, "intents": INTENTS}
    save(full, feat, d["kb"], d["labels"], cfg, vocab_text)

    from .build_retrieval import main as add_semantic_search
    add_semantic_search()            # semantic search banks + blended answer threshold, saved in the same .h5


def save(model, feat, kb, labels, cfg, vocab_text):
    PKG_MODEL.mkdir(parents=True, exist_ok=True)
    model.save(H5)                                   # Keras HDF5 format
    resp = pd.read_csv(config.DATA / "responses.csv")
    kb_records = kb[["id", "topic", "title", "answer", "source", "reviewed"]].to_dict("records")
    assets = {"featurizer": feat.to_json(), "labels": json.dumps({"intents": INTENTS, "answers": labels}),
              "knowledge_base": json.dumps(kb_records, ensure_ascii=False),
              "responses": json.dumps(dict(zip(resp.key, resp.text)), ensure_ascii=False),
              "config": json.dumps(cfg), "wordpiece_vocab": vocab_text}
    with h5py.File(H5, "a") as f:                     # add the assets next to the Keras weights
        if "intyoza" in f:
            del f["intyoza"]
        g = f.create_group("intyoza")
        for k, v in assets.items():
            g.create_dataset(k, data=np.bytes_(v.encode("utf-8")))
        g.attrs["format"] = "intyoza-srh-2"
    for k in ("labels", "config"):                    # readable copies for people (not needed to run)
        (PKG_MODEL / f"{k}.json").write_text(json.dumps(json.loads(assets[k]), indent=2), encoding="utf-8")
    print(f"Saved {H5} ({H5.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--max-features", type=int, default=40000)
    a = ap.parse_args()
    main(a.epochs, a.hidden, a.max_features)
