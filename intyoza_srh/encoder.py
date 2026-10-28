"""Sentence encoder: all-MiniLM-L6-v2 (6-layer BERT, 384-d; trained for paraphrase matching, Apache-2.0).

It turns a question into a 384-number "meaning" vector, so questions with the same meaning but
different words ("jab" / "injection", "pee" / "urine") land close together.
Three parts, all offline:
  * WordPiece tokenizer in pure Python,
  * the network built from built-in Keras layers (saved inside intyoza_srh_model.h5),
  * the same network in NumPy (used when TensorFlow is not installed)."""
import unicodedata

import numpy as np

MAX_TOKENS = 64
HIDDEN, HEADS, LAYERS_N, FFN = 384, 12, 6, 1536
EPS = 1e-12


# ----------------------------------------------------------------------------- tokenizer
class WordPiece:
    def __init__(self, vocab_lines):
        self.vocab = {t: i for i, t in enumerate(vocab_lines)}
        self.cls, self.sep, self.unk = self.vocab["[CLS]"], self.vocab["[SEP]"], self.vocab["[UNK]"]

    @staticmethod
    def _basic(text):
        text = unicodedata.normalize("NFD", text.lower())
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")
        out, word = [], ""
        for c in text:
            cp = ord(c)
            if c.isspace() or cp == 0 or cp == 0xFFFD or unicodedata.category(c) in ("Cc", "Cf"):
                if word:
                    out.append(word)
                word = ""
            elif (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126) \
                    or unicodedata.category(c).startswith("P"):
                if word:
                    out.append(word)
                out.append(c)
                word = ""
            else:
                word += c
        if word:
            out.append(word)
        return out

    def _wordpiece(self, word):
        if len(word) > 100:
            return [self.unk]
        ids, start = [], 0
        while start < len(word):
            end, cur = len(word), None
            while start < end:
                piece = ("##" if start else "") + word[start:end]
                if piece in self.vocab:
                    cur = self.vocab[piece]
                    break
                end -= 1
            if cur is None:
                return [self.unk]
            ids.append(cur)
            start = end
        return ids

    def encode(self, text):
        ids = [i for w in self._basic(text) for i in self._wordpiece(w)]
        return [self.cls] + ids[:MAX_TOKENS - 2] + [self.sep]

    def batch(self, texts):
        """Model inputs for a list of texts, padded to the longest one."""
        seqs = [self.encode(t) for t in texts]
        n = max(len(s) for s in seqs)
        tok = np.zeros((len(seqs), n), np.int32)
        mask = np.zeros((len(seqs), n), np.float32)
        for r, s in enumerate(seqs):
            tok[r, :len(s)] = s
            mask[r, :len(s)] = 1.0
        pos = np.tile(np.arange(n, dtype=np.int32), (len(seqs), 1))
        pool = mask / mask.sum(1, keepdims=True)
        return {"token_ids": tok, "position_ids": pos, "attention_mask": mask, "pool_weights": pool}


ENCODER_INPUTS = ("token_ids", "position_ids", "attention_mask", "pool_weights")


# ----------------------------------------------------------------------------- keras
def build_keras_encoder(keras):
    """Return (inputs dict, sentence-vector tensor) built only from built-in Keras layers."""
    L = keras.layers
    tok = keras.Input((None,), dtype="int32", name="token_ids")
    pos = keras.Input((None,), dtype="int32", name="position_ids")
    mask = keras.Input((None,), name="attention_mask")
    pool = keras.Input((None,), name="pool_weights")
    x = L.Add()([L.Embedding(30522, HIDDEN, name="enc_word")(tok), L.Embedding(512, HIDDEN, name="enc_pos")(pos)])
    x = L.LayerNormalization(epsilon=EPS, name="enc_ln")(x)
    att_mask = L.Reshape((1, -1))(mask)
    for i in range(LAYERS_N):
        a = L.MultiHeadAttention(HEADS, HIDDEN // HEADS, name=f"enc{i}_att")(x, x, attention_mask=att_mask)
        x = L.LayerNormalization(epsilon=EPS, name=f"enc{i}_ln1")(L.Add()([x, a]))
        f = L.Dense(FFN, activation="gelu", name=f"enc{i}_ff1")(x)
        f = L.Dense(HIDDEN, name=f"enc{i}_ff2")(f)
        x = L.LayerNormalization(epsilon=EPS, name=f"enc{i}_ln2")(L.Add()([x, f]))
    vec = L.Flatten()(L.Dot(axes=1)([x, L.Reshape((-1, 1))(pool)]))        # masked mean pooling
    vec = L.UnitNormalization(name="sentence_vector")(vec)
    return {"token_ids": tok, "position_ids": pos, "attention_mask": mask, "pool_weights": pool}, vec


def load_hf_weights(model, safetensors_path):
    """Copy the original all-MiniLM-L6-v2 weights into the Keras layers (once, at training time)."""
    from safetensors.numpy import load_file
    w = load_file(safetensors_path)
    g = lambda k: w[k]
    model.get_layer("enc_word").set_weights([g("embeddings.word_embeddings.weight")])
    # token type is always 0, so its vector is folded into the position vectors
    model.get_layer("enc_pos").set_weights([g("embeddings.position_embeddings.weight") + g("embeddings.token_type_embeddings.weight")[0]])
    model.get_layer("enc_ln").set_weights([g("embeddings.LayerNorm.weight"), g("embeddings.LayerNorm.bias")])
    hd = HIDDEN // HEADS
    for i in range(LAYERS_N):
        p = f"encoder.layer.{i}."
        qkv = []
        for n in ("query", "key", "value"):
            qkv += [g(p + f"attention.self.{n}.weight").T.reshape(HIDDEN, HEADS, hd), g(p + f"attention.self.{n}.bias").reshape(HEADS, hd)]
        out = [g(p + "attention.output.dense.weight").T.reshape(HEADS, hd, HIDDEN), g(p + "attention.output.dense.bias")]
        model.get_layer(f"enc{i}_att").set_weights(qkv + out)
        model.get_layer(f"enc{i}_ln1").set_weights([g(p + "attention.output.LayerNorm.weight"), g(p + "attention.output.LayerNorm.bias")])
        model.get_layer(f"enc{i}_ff1").set_weights([g(p + "intermediate.dense.weight").T, g(p + "intermediate.dense.bias")])
        model.get_layer(f"enc{i}_ff2").set_weights([g(p + "output.dense.weight").T, g(p + "output.dense.bias")])
        model.get_layer(f"enc{i}_ln2").set_weights([g(p + "output.LayerNorm.weight"), g(p + "output.LayerNorm.bias")])


# ----------------------------------------------------------------------------- numpy
def _erf(x):
    # Abramowitz & Stegun 7.1.26 (max error 1.5e-7), enough for float32
    s = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return s * y


def _ln(x, gamma, beta):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + EPS) * gamma + beta


def numpy_encode(W, inputs):
    """W: {layer name: [weights]} as saved in the .h5. Returns (n, 384) unit vectors."""
    tok, pos, mask, pool = (inputs[k] for k in ENCODER_INPUTS)
    x = W["enc_word"][0][tok] + W["enc_pos"][0][pos]
    x = _ln(x, *W["enc_ln"])
    neg = (1.0 - mask)[:, None, None, :] * -1e9
    hd = HIDDEN // HEADS
    for i in range(LAYERS_N):
        qk, qb, kk, kb, vk, vb, ok, ob = W[f"enc{i}_att"]
        q = np.einsum("bld,dhk->bhlk", x, qk) + qb[None, :, None, :]
        k = np.einsum("bld,dhk->bhlk", x, kk) + kb[None, :, None, :]
        v = np.einsum("bld,dhk->bhlk", x, vk) + vb[None, :, None, :]
        s = np.einsum("bhqk,bhsk->bhqs", q, k) / np.sqrt(hd) + neg
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        a = np.einsum("bhqs,bhsk->bqhk", s, v)
        a = np.einsum("bqhk,hkd->bqd", a, ok) + ob
        x = _ln(x + a, *W[f"enc{i}_ln1"])
        f = x @ W[f"enc{i}_ff1"][0] + W[f"enc{i}_ff1"][1]
        f = 0.5 * f * (1.0 + _erf(f / np.sqrt(2.0)))
        f = f @ W[f"enc{i}_ff2"][0] + W[f"enc{i}_ff2"][1]
        x = _ln(x + f, *W[f"enc{i}_ln2"])
    vec = np.einsum("bld,bl->bd", x, pool)
    return vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12)


ENCODER_LAYERS = ["enc_word", "enc_pos", "enc_ln"] + [f"enc{i}_{n}" for i in range(LAYERS_N)
                                                       for n in ("att", "ln1", "ff1", "ff2", "ln2")]
