"""Check one knowledge-base expansion file (JSON Lines). Standard library only.
Run:  python tools/kb_expansion/check_part.py data/kb_expansion/MEN.jsonl"""
import json
import re
import sys

TOPICS = {"menstruation", "puberty", "contraception", "pregnancy", "stis", "hiv",
          "relationships", "consent_safety", "wellbeing", "services_rights"}
SOURCES = {
    "UNICEF/IMC Growing Healthy booklet 2015", "WHO/JHU Family Planning Handbook 2022",
    "WHO Adolescent Contraception 2017", "WHO Adolescent SRHR 2018",
    "Rwanda HIV Guidelines 2022", "REB/UNFPA CSE reference book 2018",
    "UNESCO Technical Guidance on Sexuality Education 2018", "WHO fact sheet",
    "Rwanda Ministry of Health / RBC",
}
# Things an answer must never contain (doses, abortion methods).
BANNED = re.compile(r"\b\d+\s?(mg|milligram|ml)\b|misoprostol|mifepristone|tablets? every \d+ hours", re.I)
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", str(s).lower())).strip()


def check(path):
    errors, ids, titles, qs = [], set(), set(), set()
    n = 0
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            n += 1
            try:
                e = json.loads(line)
            except json.JSONDecodeError as ex:
                errors.append(f"line {ln}: bad JSON ({ex})")
                continue
            for k in ("id", "topic", "title", "answer", "source", "questions", "test"):
                if k not in e:
                    errors.append(f"line {ln}: missing '{k}'")
            if errors and errors[-1].startswith(f"line {ln}:"):
                continue
            if e["id"] in ids:
                errors.append(f"line {ln}: duplicate id {e['id']}")
            ids.add(e["id"])
            if norm(e["title"]) in titles:
                errors.append(f"line {ln}: duplicate title '{e['title']}'")
            titles.add(norm(e["title"]))
            if e["topic"] not in TOPICS:
                errors.append(f"line {ln}: topic '{e['topic']}' not in {sorted(TOPICS)}")
            if e["source"] not in SOURCES:
                errors.append(f"line {ln}: source '{e['source']}' not allowed")
            w = len(e["answer"].split())
            if not 30 <= w <= 130:
                errors.append(f"line {ln}: answer has {w} words (need 30-130)")
            if BANNED.search(e["answer"]):
                errors.append(f"line {ln}: answer contains a dose or abortion-method term")
            if not (isinstance(e["questions"], list) and len(e["questions"]) >= 6):
                errors.append(f"line {ln}: need at least 6 training questions")
            if not (isinstance(e["test"], list) and len(e["test"]) >= 2):
                errors.append(f"line {ln}: need at least 2 test questions")
            for q in e.get("questions", []) + e.get("test", []):
                k = norm(q)
                if k in qs:
                    errors.append(f"line {ln}: question repeated in this file: '{q}'")
                qs.add(k)
            if set(map(norm, e.get("questions", []))) & set(map(norm, e.get("test", []))):
                errors.append(f"line {ln}: a test question is also a training question")
    print(f"{path}: {n} entries, {len(errors)} errors")
    for x in errors[:50]:
        print("  " + x)
    return not errors


if __name__ == "__main__":
    ok = all([check(p) for p in sys.argv[1:]])
    sys.exit(0 if ok else 1)
