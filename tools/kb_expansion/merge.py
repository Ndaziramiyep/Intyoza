"""Merge data/kb_expansion/*.jsonl into the project CSV files. Safe to run again (idempotent).
Run:  python tools/kb_expansion/merge.py

  knowledge_base.csv  original entries + expansion entries (reviewed = pending)
  questions.csv       + each expansion entry's training questions (style "seed")
  test_set.csv        + each expansion entry's held-out test questions (source "expansion")
Questions that appear in two entries, or in both training and test, are dropped."""
import csv
import glob
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from check_part import check  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "data")
PREFIXES = ("MEN", "PUB", "CSA", "CSB", "PRA", "PRB", "STI", "HIV", "REL", "CON", "WEL", "SRV")
MAX_TOTAL = 1500              # answers in knowledge_base.csv (original + expansion)
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", str(s).lower())).strip()


def read_csv(name):
    with open(os.path.join(DATA, name), encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def write_csv(name, fields, rows):
    with open(os.path.join(DATA, name), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    files = sorted(glob.glob(os.path.join(DATA, "kb_expansion", "*.jsonl")))
    if not all(check(p) for p in files):
        print("Fix the errors above first (entries with errors are still merged if they parse).")
    entries = []
    for p in files:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    kb_f, kb = read_csv("knowledge_base.csv")
    q_f, qs = read_csv("questions.csv")
    t_f, ts = read_csv("test_set.csv")
    is_exp = lambda i: str(i).split("-")[0] in PREFIXES
    kb = [r for r in kb if not is_exp(r["id"])]
    qs = [r for r in qs if not (is_exp(r["entry_id"]) or r["style"] == "expansion")]
    ts = [r for r in ts if not (is_exp(r["expected_entry_id"]) or r.get("source") == "expansion")]

    # duplicates.json {dropped_id: kept_id}: entries that answer the same question as another entry.
    # The dropped entry's questions move to the kept entry, so no training data is lost.
    dup_path = os.path.join(DATA, "kb_expansion", "duplicates.json")
    dups = json.load(open(dup_path, encoding="utf-8")) if os.path.exists(dup_path) else {}
    by_id = {e.get("id"): e for e in entries}
    orig_ids = {r["id"] for r in kb}
    to_orig = []                                  # (original id, entry) when an expansion entry repeats an original one
    for gone, kept in dups.items():
        if gone not in by_id or gone == kept:
            continue
        if kept in by_id:
            for k in ("questions", "test"):           # combine, keeping each question text once
                seen = set()
                both = by_id[kept][k] + by_id[gone][k]
                by_id[kept][k] = [x for x in both if not (norm(x) in seen or seen.add(norm(x)))]
        elif kept in orig_ids:
            to_orig.append((kept, by_id[gone]))
    valid_target = lambda i: dups[i] in by_id or dups[i] in orig_ids
    entries = [e for e in entries if e.get("id") not in dups or not valid_target(e["id"])]
    print(f"merged {len(dups)} duplicate entries into the entry that answers the same question")

    # drop entries with a missing field, a repeated id, or a title already used
    seen_titles = {norm(r["title"]) for r in kb}
    seen_ids = {r["id"] for r in kb}
    keep = []
    for e in entries:
        if not all(k in e for k in ("id", "topic", "title", "answer", "source", "questions", "test")):
            continue
        if e["id"] in seen_ids or norm(e["title"]) in seen_titles:
            print(f"  skipped duplicate: {e['id']} {e['title']}")
            continue
        seen_ids.add(e["id"])
        seen_titles.add(norm(e["title"]))
        keep.append(e)

    # cap the total size: drop the last-written entries of the largest topics first
    while len(kb) + len(keep) > MAX_TOTAL:
        sizes = Counter(e["topic"] for e in keep)
        big = max(sizes, key=lambda t: (sizes[t] + sum(r["topic"] == t for r in kb), t))
        last = max((i for i, e in enumerate(keep) if e["topic"] == big), key=lambda i: keep[i]["id"])
        keep.pop(last)

    # a question text may belong to one entry only; test texts must never be training texts
    owner = Counter()
    for r in qs:
        owner[norm(r["question"])] += 1
    for e in keep:
        for x in set(map(norm, e["questions"] + e["test"])):
            owner[x] += 1
    for oid, e in to_orig:
        for x in set(map(norm, e["questions"] + e["test"])):
            owner[x] += 1
    train_norm = {norm(r["question"]) for r in qs}
    n_q = n_t = dropped = 0
    for oid, e in to_orig:                        # questions of a duplicate go to the original entry
        for x in e["questions"]:
            if owner[norm(x)] == 1:
                qs.append({"question": x, "entry_id": oid, "style": "expansion", "district": ""})
                train_norm.add(norm(x))
                n_q += 1
        for x in e["test"]:
            if owner[norm(x)] == 1 and norm(x) not in train_norm:
                ts.append({"question": x, "expected_entry_id": oid, "expected_intent": "srh_question",
                           "style": "seed", "district": "", "source": "expansion"})
                n_t += 1
    for e in keep:
        kb.append({"id": e["id"], "topic": e["topic"], "title": e["title"], "answer": e["answer"],
                   "source": e["source"], "reviewed": "pending", "reviewer_note": ""})
        for x in e["questions"]:
            if owner[norm(x)] == 1:
                qs.append({"question": x, "entry_id": e["id"], "style": "seed", "district": ""})
                train_norm.add(norm(x))
                n_q += 1
            else:
                dropped += 1
    for e in keep:
        for x in e["test"]:
            if owner[norm(x)] == 1 and norm(x) not in train_norm:
                ts.append({"question": x, "expected_entry_id": e["id"], "expected_intent": "srh_question",
                           "style": "seed", "district": "", "source": "expansion"})
                n_t += 1
            else:
                dropped += 1
    # older test rows (e.g. NOT_SURE questions the new answers now cover) must not repeat a training question
    n_before = len(ts)
    ts = [r for r in ts if r.get("source") == "expansion" or norm(r["question"]) not in train_norm]
    if len(ts) < n_before:
        print(f"removed {n_before - len(ts)} old test rows that are now training questions")

    write_csv("knowledge_base.csv", kb_f, kb)
    write_csv("questions.csv", q_f, qs)
    write_csv("test_set.csv", t_f, ts)
    print(f"knowledge_base.csv: {len(kb)} entries ({len(keep)} from expansion)")
    print(f"questions.csv: +{n_q} training questions; test_set.csv: +{n_t} test questions; {dropped} ambiguous questions dropped")
    print("Topics:", dict(Counter(r["topic"] for r in kb)))


if __name__ == "__main__":
    main()
