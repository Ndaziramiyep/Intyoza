"""Check the CSV files for mistakes.   Run:  python -m src.validate_data"""
import re
import sys

import pandas as pd

from . import config

D = config.DATA
kb, q = pd.read_csv(D / "knowledge_base.csv"), pd.read_csv(D / "questions.csv")
it, te = pd.read_csv(D / "intents.csv"), pd.read_csv(D / "test_set.csv")
rt, rs = pd.read_csv(D / "redteam_prompts.csv"), pd.read_csv(D / "responses.csv")
ps = pd.read_csv(D / "source_passages.csv")
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", str(s).lower())).strip()
errors = []

if kb.id.duplicated().any():
    errors.append("duplicate ids in knowledge_base.csv")
if kb.answer.isna().any():
    errors.append("empty answers in knowledge_base.csv")
if not set(q.entry_id) <= set(kb.id):
    errors.append(f"questions.csv has unknown entry ids: {set(q.entry_id) - set(kb.id)}")
few = [e for e, n in q.groupby("entry_id").size().items() if n < 3] + list(set(kb.id) - set(q.entry_id))
if few:
    errors.append(f"entries with fewer than 3 example questions: {few}")
ok_expected = set(kb.id) | {"NOT_SURE", "CRISIS"}
if not set(te.expected_entry_id) <= ok_expected:
    errors.append(f"test_set has unknown expected_entry_id: {set(te.expected_entry_id) - ok_expected}")
bad_lab = set(it.label) - (set(config.SAFE_INTENTS) | {"off_topic", "srh_question"})
if bad_lab:
    errors.append(f"intents.csv has unknown labels: {bad_lab}")
for name, df, col in [("questions", q, "question"), ("intents", it, "text"), ("test_set", te, "question"), ("redteam", rt, "prompt")]:
    if df[col].map(norm).duplicated().any():
        errors.append(f"{name}.csv has duplicate rows")
train = {norm(x) for x in q.question} | {norm(x) for x in it.text}
for name, texts in [("test_set", te.question), ("redteam", rt.prompt)]:
    leak = [x for x in texts if norm(x) in train]
    if leak:
        errors.append(f"{len(leak)} {name} rows are also in the training files, e.g. {leak[:3]}")
if {norm(x) for x in te.question} & {norm(x) for x in rt.prompt}:
    errors.append("test_set and redteam share rows")
need = set(config.SAFE_INTENTS) | {"off_topic", "not_sure", "disclaimer", "srh_referral_footer", "refuse_medicine"}
if need - set(rs.key):
    errors.append(f"responses.csv missing keys: {need - set(rs.key)}")
if (kb.reviewed != "yes").any():
    print(f"NOTE: {(kb.reviewed != 'yes').sum()} of {len(kb)} entries are not yet marked reviewed = yes")

print(f"knowledge_base={len(kb)} questions={len(q)} intents={len(it)} test_set={len(te)} redteam={len(rt)} source_passages={len(ps)}")
print("intent classes:", it.label.value_counts().to_dict())
print("writing styles in questions:", q["style"].value_counts().to_dict())
if errors:
    print("\nPROBLEMS:")
    for e in errors:
        print(" -", e)
    sys.exit(1)
print("\nData OK")
