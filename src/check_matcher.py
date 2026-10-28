"""Day 4 quick check of the matcher alone.   Run:  python -m src.check_matcher"""
import pandas as pd

from . import config
from .matcher import Matcher

q = pd.read_csv(config.DATA / "questions.csv")
kb = pd.read_csv(config.DATA / "knowledge_base.csv")
test = pd.read_csv(config.DATA / "test_set.csv")
m = Matcher().fit(q.question, q.entry_id, extra_texts=list(kb.title) + list(kb.answer), extra_ids=list(kb.id) * 2)

ins = test[(test.expected_entry_id != "NOT_SURE") & (test.expected_intent == "srh_question")]
top = m.top_batch(ins.question, 3)
print("in-scope test questions:", len(ins))
print("top-1 accuracy:", round(sum(t[0][0] == e for t, e in zip(top, ins.expected_entry_id)) / len(ins), 2))
print("top-3 accuracy:", round(sum(e in [x[0] for x in t] for t, e in zip(top, ins.expected_entry_id)) / len(ins), 2))
print("\nFirst 15 misses (question | expected | model's top 3):")
shown = 0
for x, e, t in zip(ins.question, ins.expected_entry_id, top):
    if e not in [y[0] for y in t] and shown < 15:
        print(f"- {x} | {e} | {[y[0] for y in t]}")
        shown += 1
