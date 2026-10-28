"""Intyoza SRH chatbot: safety rules -> Keras model (intent head + answer head) -> confidence gate.

Everything the bot needs is inside ONE file, model/intyoza_srh_model.h5:
  * the Keras network: MiniLM sentence encoder + tf-idf branch -> "intent" and "answer" heads,
  * an extra HDF5 group "intyoza" with the vocabulary, labels, knowledge base, fixed replies and settings.

    from intyoza_srh import IntyozaChatbot
    bot = IntyozaChatbot()                  # or IntyozaChatbot("path/to/intyoza_srh_model.h5")
    print(bot.ask("can i get pregnant the first time")["answer"])

backend="auto" uses TensorFlow when it is installed, otherwise pure NumPy (only numpy + h5py needed).
The bot never writes new health facts: it only returns reviewed answers from the knowledge base."""
import json
import os
import re
from pathlib import Path

import h5py
import numpy as np

from .encoder import ENCODER_LAYERS, WordPiece, numpy_encode
from .featurizer import Featurizer, clean
from .safety import rule_intent, wants_refusal

DEFAULT_MODEL = Path(__file__).resolve().parent / "model" / "intyoza_srh_model.h5"
HEAD_LAYERS = ("hidden", "embed_hidden", "answer", "intent_tfidf", "intent_hidden", "intent")

SMALL_TALK = [  # (pattern on the WHOLE cleaned message, response key)
    (r"(hi|hello|hey|helo|hallo|good (morning|afternoon|evening)|muraho|mwiriwe|mwaramutse|bite|amakuru)( there| intyoza| madam| sir)?",
     "greeting"),
    (r"(thank you|thanks|thank u|thx|thanks a lot|murakoze|ok thanks|okay thank you)( so much| very much)?( intyoza)?", "thanks"),
    (r"(bye|goodbye|see you|good night|ok bye|murabeho)", "goodbye"),
    (r"(who are you|what are you|what is intyoza|what can you do|are you a (robot|bot|human|doctor))", "about"),
]
SMALL_TALK_TEXT = {
    "greeting": "Hello! I am Intyoza. You can ask me anything about growing up, periods, relationships, "
                "consent, contraception, pregnancy, STIs or HIV. Nobody will judge you. What would you like to know?",
    "thanks": "You are welcome! Do you have another question? I am here to help.",
    "goodbye": "Take care! Come back any time you have a question. For personal advice you can visit your "
               "nearest health center or call 114 for free.",
    "about": "I am Intyoza, an information tool about sexual and reproductive health for young people in Rwanda. "
             "I answer with information written from trusted health guides. I am not a doctor, I do not ask your "
             "name and I do not save your questions.",
}


class _NumpyNet:
    """Runs the saved Keras network with NumPy only:
    question -> MiniLM sentence vector -> embed_hidden ; tf-idf -> hidden ; concat -> intent / answer softmax."""

    def __init__(self, f):
        self.w = {}
        for name in HEAD_LAYERS + tuple(ENCODER_LAYERS):
            g = f["model_weights"][name]
            names = [n.decode() if isinstance(n, bytes) else n for n in g.attrs["weight_names"]]
            self.w[name] = [np.asarray(g[n]) for n in names]

    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, inputs):
        relu = lambda x, layer: np.maximum(x @ self.w[layer][0] + self.w[layer][1], 0.0)
        vec = numpy_encode(self.w, inputs)
        f = inputs["features"]
        h = np.concatenate([relu(f, "hidden"), relu(vec, "embed_hidden")], axis=1)               # answer branch
        hi = relu(np.concatenate([relu(f, "intent_tfidf"), vec], axis=1), "intent_hidden")       # intent branch
        return [self._softmax(hi @ self.w["intent"][0] + self.w["intent"][1]),
                self._softmax(h @ self.w["answer"][0] + self.w["answer"][1]), vec]


class IntyozaChatbot:
    def __init__(self, model_path=None, backend="auto"):
        self.model_path = str(model_path or os.getenv("INTYOZA_MODEL", DEFAULT_MODEL))
        with h5py.File(self.model_path, "r") as f:
            a = f["intyoza"]
            read = lambda k: a[k][()].decode() if isinstance(a[k][()], bytes) else a[k][()]
            self.featurizer = Featurizer.from_json(read("featurizer"))
            self.tokenizer = WordPiece(read("wordpiece_vocab").splitlines())
            labels = json.loads(read("labels"))
            self.kb = {e["id"]: e for e in json.loads(read("knowledge_base"))}
            self.resp = json.loads(read("responses"))
            self.cfg = json.loads(read("config"))
            self.intent_labels, self.answer_labels = labels["intents"], labels["answers"]
            if backend == "auto":
                try:
                    import tensorflow  # noqa: F401
                    backend = "keras"
                except ImportError:
                    backend = "numpy"
            self.backend = backend
            if backend == "numpy":
                self._net = _NumpyNet(f)
            # semantic search banks (meaning vectors of training questions and of intent examples)
            self.search = "search_vectors" in a
            if self.search:
                self.search_vec = np.asarray(a["search_vectors"], np.float32)
                self.search_entry = np.asarray(a["search_entry"])
                self.intent_vec = np.asarray(a["intent_vectors"], np.float32)
                self.intent_lab = np.asarray(a["intent_labels"])
        if backend == "keras":
            import tensorflow as tf
            net = tf.keras.models.load_model(self.model_path, compile=False)
            if len(net.outputs) == 2:                 # older file: add the sentence vector as an output
                net = tf.keras.Model(net.inputs, list(net.outputs) + [net.get_layer("sentence_vector").output])
            self._net = net
        self.require_reviewed = os.getenv("INTYOZA_REQUIRE_REVIEWED", "1" if self.cfg.get("require_reviewed") else "0") == "1"
        self._small_talk = [(re.compile(rf"^{p}$"), k) for p, k in SMALL_TALK]

    # ------------------------------------------------------------------ model
    def predict(self, texts):
        """Return (intent_probs, answer_probs, sentence_vectors) arrays for a list of texts."""
        texts = [clean(t) for t in texts]
        outs = []
        for i in range(0, len(texts), 256):          # in chunks: the tf-idf matrix is wide
            chunk = texts[i:i + 256]
            inputs = self.tokenizer.batch(chunk)
            inputs["features"] = self.featurizer.transform(chunk)
            if self.backend == "keras":
                outs.append(self._net.predict(inputs, verbose=0, batch_size=256))
            else:
                outs.append(self._net.predict(inputs))
        return tuple(np.concatenate([np.asarray(o[k]) for o in outs]) for k in (0, 1, 2))

    def answer_scores(self, p_ans, vec):
        """Classifier probabilities blended with semantic search (best-matching training question per answer)."""
        if not self.search:
            return p_ans, None
        c = self.cfg["search"]
        sims = vec @ self.search_vec.T
        K = np.full(p_ans.shape, -1.0, np.float32)
        for r in range(len(sims)):
            np.maximum.at(K[r], self.search_entry, sims[r])
        Ks = np.exp((K - K.max(1, keepdims=True)) / c["temperature"])
        Ks /= Ks.sum(1, keepdims=True)
        return c["classifier_weight"] * p_ans + (1 - c["classifier_weight"]) * Ks, K

    def semantic_crisis(self, vec, probs):
        """Crisis label if the closest examples to this message are crisis messages, else None.
        When the classifier is sure the message is NOT that crisis, a clearer gap is required."""
        if not self.search:
            return None
        c = self.cfg["search"]
        k = c["k"]
        sims = self.intent_vec @ vec
        mean_top = lambda x: float(np.sort(x)[-k:].mean()) if len(x) else -1.0
        score = {lab: mean_top(sims[self.intent_lab == self.intent_labels.index(lab)])
                 for lab in self.cfg["safe_min_prob"]}
        other = max(mean_top(sims[self.intent_lab == self.intent_labels.index("srh_question")]),
                    mean_top(sims[self.intent_lab == self.intent_labels.index("off_topic")]),
                    mean_top(self.search_vec @ vec))
        best = max(score, key=score.get)
        gap = score[best] - other
        if score[best] < c["floor"] or gap < c["margin"]:
            return None
        if probs[best] < c.get("min_model_prob", 0.0) and gap < c.get("strong_gap", 0.0):
            return None
        return best

    def accuracy(self, score):
        """Estimated chance that an answer with this score is right (measured on validation questions)."""
        pts = self.cfg.get("calibration")
        if not pts:
            return None
        xs, ys = zip(*pts)
        return float(np.interp(score, xs, ys))

    def top_answers(self, texts, k=3):
        """For each text, the k best (entry_id, score) pairs, best first."""
        _, P, V = self.predict(texts)
        S, _ = self.answer_scores(P, V)
        order = np.argsort(-S, axis=1)[:, :k]
        return [[(self.answer_labels[j], float(s[j])) for j in row] for s, row in zip(S, order)]

    # ------------------------------------------------------------------ chat
    def ask(self, text: str) -> dict:
        t = clean(text)
        if not t:
            return self._reply("off_topic", "off_topic")
        for pattern, key in self._small_talk:
            if pattern.match(t):
                return {"type": "small_talk", "intent": "small_talk", "answer": SMALL_TALK_TEXT[key],
                        "disclaimer": self.resp["disclaimer"], "suggestions": self._starter_titles() if key in ("greeting", "about") else []}
        # 1. keyword backup for crisis messages (runs first, cannot be skipped)
        label = rule_intent(t)
        if label:
            return self._reply("safe", label, source="rule")
        if wants_refusal(t):
            return self._reply("refuse", "srh_question")
        # 2. intent head, then semantic crisis check (closest examples are crisis messages)
        p_int, p_ans, vec = self.predict([t])
        probs = dict(zip(self.intent_labels, p_int[0].tolist()))
        for label, bar in self.cfg["safe_min_prob"].items():
            if probs[label] >= bar:
                return self._reply("safe", label, prob=round(probs[label], 3), source="model")
        S, K = self.answer_scores(p_ans, vec)
        order = np.argsort(-S[0])[:4]
        top = [(self.answer_labels[j], float(S[0][j])) for j in order]
        confident = top[0][1] >= self.cfg["answer_threshold"]
        # clearly not about SRH (and no confident answer): say so before any closer look
        if not confident and probs["off_topic"] >= self.cfg["offtopic_min_prob"]:
            return self._reply("off_topic", "off_topic", prob=round(probs["off_topic"], 3))
        label = self.semantic_crisis(vec[0], probs)
        if label:
            return self._reply("safe", label, source="semantic")
        if not confident:
            # no confident answer: when in doubt, a weaker crisis signal is enough for the safe reply
            for label, bar in self.cfg.get("safe_min_prob_when_unsure", {}).items():
                if probs[label] >= bar:
                    return self._reply("safe", label, prob=round(probs[label], 3), source="model_unsure")
        # 3. answer + confidence gate; "best guess" when fairly but not fully sure, and only for real SRH questions
        allowed = [(i, s) for i, s in top if not self.require_reviewed or self.kb[i]["reviewed"] == "yes"]
        min_sim = self.cfg.get("search", {}).get("suggest_min_similarity", 1.0)
        close = lambda i, s: s >= self.cfg["suggest_threshold"] or (K is not None and K[0][self.answer_labels.index(i)] >= min_sim)
        guess = (top[0][1] >= self.cfg.get("guess_threshold", self.cfg["answer_threshold"])
                 and probs["srh_question"] >= 0.5 and probs["off_topic"] < 0.2)
        if (confident or guess) and allowed and allowed[0][0] == top[0][0]:
            e = self.kb[top[0][0]]
            related = [self.kb[i]["title"] for i, s in top[1:] if close(i, s)]
            text = f"{e['answer']}\n\n{self.resp['srh_referral_footer']}"
            if not confident:
                text = "I think this is what you are asking about:\n\n" + text
            acc = self.accuracy(top[0][1])
            return {"type": "answer", "intent": "srh_question", "entry_id": e["id"], "topic": e["topic"],
                    "title": e["title"], "answer": text, "best_guess": not confident,
                    "confidence": None if acc is None else round(100 * acc),
                    "source": e["source"], "reviewed": e["reviewed"], "disclaimer": self.resp["disclaimer"],
                    "score": round(top[0][1], 3), "candidates": top[:3], "suggestions": related}
        # 4. not sure: never a dead end. Offer the answers closest in meaning (semantic search).
        maybe = [self.kb[i]["title"] for i, s in allowed if close(i, s)]
        return self._reply("not_sure", "srh_question", score=round(top[0][1], 3), candidates=top[:3], suggestions=maybe)

    def answer_by_title(self, title: str) -> dict:
        """Used when the user taps a suggestion chip: returns that exact entry."""
        for e in self.kb.values():
            if e["title"].lower() == title.strip().lower():
                if self.require_reviewed and e["reviewed"] != "yes":
                    break
                return {"type": "answer", "intent": "srh_question", "entry_id": e["id"], "topic": e["topic"],
                        "title": e["title"], "answer": f"{e['answer']}\n\n{self.resp['srh_referral_footer']}",
                        "source": e["source"], "reviewed": e["reviewed"], "disclaimer": self.resp["disclaimer"],
                        "score": 1.0, "suggestions": []}
        return self.ask(title)

    def chat(self, text: str) -> str:
        """Convenience: just the reply text."""
        return self.ask(text)["answer"]

    def topics(self):
        """{topic: [titles]} for menus."""
        out = {}
        for e in self.kb.values():
            out.setdefault(e["topic"], []).append(e["title"])
        return out

    def _starter_titles(self):
        return [t for t in self.cfg.get("starter_questions", []) if any(e["title"] == t for e in self.kb.values())]

    def _reply(self, kind, intent, **extra):
        key = {"safe": intent, "off_topic": "off_topic", "not_sure": "not_sure", "refuse": "refuse_medicine"}[kind]
        out = {"type": kind, "intent": intent, "answer": self.resp[key], "disclaimer": self.resp["disclaimer"]}
        if kind == "not_sure" and extra.get("suggestions"):
            out["answer"] = ("That is a good question. I do not have one exact answer for it, but these answers are "
                             "close to what you asked. Tap one to read it, or ask again in other words.\n\n"
                             "For personal advice, visit your nearest health center or call 114 (free). If you are in "
                             "danger or feeling unsafe, call 112, 116 (child helpline) or 3512 (gender-based violence).")
        out.setdefault("suggestions", [])
        out.update(extra)
        return out
