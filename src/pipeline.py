"""The full model: safety rules -> intent classifier -> matcher -> confidence gate -> answer."""
import joblib
import pandas as pd

from . import config
from .preprocess import clean
from .safety_rules import rule_intent, wants_refusal


class Intyoza:
    def __init__(self):
        self.intent = joblib.load(config.MODELS / "intent.joblib")
        self.matcher = joblib.load(config.MODELS / "matcher.joblib")
        kb = pd.read_csv(config.DATA / "knowledge_base.csv").fillna("")
        self.kb = kb.set_index("id").to_dict("index")
        resp = pd.read_csv(config.DATA / "responses.csv")
        self.resp = dict(zip(resp.key, resp.text))
        self.threshold = config.THRESHOLDS[self.matcher.backend]

    def ask(self, text: str) -> dict:
        t = clean(text)
        if not t:
            return self._reply("off_topic", "off_topic")
        # 1. keyword backup for crisis messages (runs first, cannot be skipped)
        label = rule_intent(t)
        if label:
            return self._reply("safe", label, source="rule")
        if wants_refusal(t):
            return self._reply("refuse", "srh_question")
        # 2. ML intent classifier
        probs = dict(zip(self.intent.classes_, self.intent.predict_proba([t])[0]))
        for label in config.SAFE_INTENTS:
            if probs[label] >= config.SAFE_MIN_PROB[label]:
                return self._reply("safe", label, prob=round(probs[label], 3), source="model")
        if probs["off_topic"] >= config.OFFTOPIC_MIN_PROB:
            return self._reply("off_topic", "off_topic", prob=round(probs["off_topic"], 3))
        # 3. match the question to the reviewed knowledge base
        top = self.matcher.top(t, k=3)
        entry_id, score = top[0]
        if score < self.threshold:
            return self._reply("not_sure", "srh_question", score=round(score, 3), candidates=top)
        entry = self.kb[entry_id]
        if config.REQUIRE_REVIEWED and entry["reviewed"] != "yes":      # launch gate
            return self._reply("not_sure", "srh_question", score=round(score, 3), candidates=top)
        return {"type": "answer", "intent": "srh_question", "entry_id": entry_id, "title": entry["title"],
                "answer": f"{entry['answer']}\n\n{self.resp['srh_referral_footer']}",
                "disclaimer": self.resp["disclaimer"], "score": round(score, 3), "candidates": top}

    def _reply(self, kind, intent, **extra):
        key = {"safe": intent, "off_topic": "off_topic", "not_sure": "not_sure", "refuse": "refuse_medicine"}[kind]
        out = {"type": kind, "intent": intent, "answer": self.resp[key], "disclaimer": self.resp["disclaimer"]}
        out.update(extra)
        return out
