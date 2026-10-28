"""Generate the large English, Rwanda-personalised training / test / red-team files.
Run:  python make_data.py     (writes to ./out ; copy the CSVs to ../../data/ yourself)"""
import csv, os, random, re, sys, itertools, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from q_data import Q as Q_BASE
from test_data import TEST as GOLD_TEST, REDTEAM as GOLD_RED
from intent_data import INTENTS as INT_BASE
from seeds_core import S as CORE
from seeds_core_extra import EXTRA
import seeds_other as O
import seeds_train_extra as X
from seeds_red import RED

OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
rng = random.Random(2026)

# ------------------------- Rwanda context -------------------------
DISTRICTS = {
 "Kigali City": ["Gasabo", "Kicukiro", "Nyarugenge"],
 "Northern": ["Burera", "Gakenke", "Gicumbi", "Musanze", "Rulindo"],
 "Southern": ["Gisagara", "Huye", "Kamonyi", "Muhanga", "Nyamagabe", "Nyanza", "Nyaruguru", "Ruhango"],
 "Eastern": ["Bugesera", "Gatsibo", "Kayonza", "Kirehe", "Ngoma", "Nyagatare", "Rwamagana"],
 "Western": ["Karongi", "Ngororero", "Nyabihu", "Nyamasheke", "Rubavu", "Rusizi", "Rutsiro"]}
ALLD = [d for v in DISTRICTS.values() for d in v]
PROV = {d: p for p, v in DISTRICTS.items() for d in v}
LEVELS = ["S1", "S2", "S3", "S4", "S5", "S6", "P6", "TVET year 1", "TVET year 2", "university year 1"]

EN_PRE = ["", "", "hello, ", "good morning, ", "good evening, ", "please, ", "pls help: ", "hi i have a question. ", "madam, ", "sir, ", "i want to know, ", "my friend asked me and i dont know: ", "my mother told me to ask: ", "at school we were told something and i want to confirm: ", "excuse me, ", "quick question: "]
EN_SUF = ["", "", "", " thank you", " please", " pls help", " (i am afraid to ask at home)", " i have nobody to ask", " help me understand", " thanks", " i am confused", " explain simply"]
LOC_PRE = ["kindly, ", "please help me, ", "madam, i want to ask: ", "my friend and i were arguing about this: ", "our health center nurse did not explain this: ", "the community health worker in our village said i should ask: ", "at the youth corner they told me to ask: ", "we were told in the sexuality education lesson: ", "i am asking for my sister: ", "please advise, ", "we have no one to ask at our boarding school: ", "my aunt in the village says something different: "]
LOC_SUF = [" kindly help", " thanks in advance", " i will be grateful", " please reply soon", " my friends cannot explain", " we have no one to ask in our village", " i cannot ask at school", " thank you very much", " i heard it from a classmate"]
SMS = {"please": "pls", "what": "wat", "how": "hw", "you": "u", "your": "ur", "with": "wit", "when": "wen", "because": "coz", "are": "r", "before": "b4", "thank you": "thanks", "about": "abt", "tomorrow": "2moro", "people": "ppl"}

FEM = re.compile(r"pregnan|period|menstru|breast|\bpad\b|bleeding|vagina|virgin|hymen|uterus|womb|cervical|antenatal|expecting|girl", re.I)
MAS = re.compile(r"my penis|wet dream|erection|boys? voice|vasectomy|ejaculat|nocturnal|hard in class", re.I)
def gender_of(core): return "f" if FEM.search(core) else ("m" if MAS.search(core) else None)

def ctx_en(rng, g=None):
    d = rng.choice(ALLD); a = rng.randint(12, 24)
    opts = [f"i am {a} years old. ", f"i am a {rng.choice(LEVELS)} student in {d}. ", f"i live in {d} district. ", f"im {a} and live in {PROV[d]} province. ",
            f"i study at a boarding school in {d}. ", f"i am a TVET student near {d} town. ", f"i come from a village in {d}. "]
    if g != "m": opts.append(f"i am a girl of {a} from {d}. ")
    if g != "f": opts.append(f"i am a boy of {a} in {d}. ")
    return rng.choice(opts), d

def typo(t, rng):
    ws = t.split(" ")
    idx = [i for i, w in enumerate(ws) if len(w) > 3 and w.isalpha()]
    if not idx: return t
    i = rng.choice(idx); w = ws[i]; k = rng.randrange(1, len(w) - 1)
    op = rng.choice(["swap", "drop", "dup"])
    w = (w[:k] + w[k+1] + w[k] + w[k+2:]) if op == "swap" and k < len(w) - 1 else (w[:k] + w[k+1:]) if op == "drop" else (w[:k] + w[k] + w[k:])
    ws[i] = w; return " ".join(ws)
def sms(t, rng):
    for a, b in SMS.items():
        if re.search(rf"\b{a}\b", t, re.I) and rng.random() < 0.7: t = re.sub(rf"\b{a}\b", b, t, flags=re.I)
    return t
def final_noise(t, rng):
    if rng.random() < 0.25: t = typo(t, rng)
    if rng.random() < 0.12: t = sms(t, rng)
    if rng.random() < 0.20: t = re.sub(r"[?.!,]", "", t)
    if rng.random() < 0.06: t = t.upper()
    elif rng.random() < 0.08: t = t.lower()
    if rng.random() < 0.05: t = t.rstrip("?.! ") + rng.choice(["??", "!!", " ?", "..."])
    return t.strip()

def augment(core, rng):
    """returns (text, style, district)"""
    style = rng.choices(["plain", "polite", "context", "local", "sms", "typo"], [1, 2, 2, 1.5, 1, 1.5])[0]
    t, dist = core, ""
    if style == "polite":
        t = rng.choice(EN_PRE) + core + rng.choice(EN_SUF)
    elif style == "context":
        c, dist = ctx_en(rng, gender_of(core)); t = c + core + (rng.choice(EN_SUF) if rng.random() < .3 else "")
    elif style == "local":
        t = rng.choice(LOC_PRE) + core + (rng.choice(LOC_SUF) if rng.random() < .6 else "")
    elif style == "sms":
        t = sms(core, rng)
    elif style == "typo":
        t = typo(core, rng)
    if style != "plain" or rng.random() < .3: t = final_noise(t, rng)
    return t, style, dist

norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", str(s).lower())).strip()

def expand(cores, per_core, used, rng, max_tries=80):
    rows = []
    for core, payload in cores:
        n = 0; tries = 0; key = norm(core)
        if key not in used: used.add(key); rows.append((core, "seed", "", payload)); n += 1
        while n < per_core and tries < max_tries:
            tries += 1
            t, st, d = augment(core, rng); k = norm(t)
            if k in used or len(k) < 4: continue
            used.add(k); rows.append((t, st, d, payload)); n += 1
    return rows

def expand_n(bases, n, used, rng, max_tries=400000):
    rows, tries = [], 0
    while len(rows) < n and tries < max_tries:
        tries += 1
        core, payload = rng.choice(bases)
        t, st, d = augment(core, rng); k = norm(t)
        if k in used or len(k) < 4: continue
        used.add(k); rows.append((t, st, d, payload))
    if len(rows) < n: print("WARNING: only", len(rows), "of", n)
    return rows

NAMES = {"perp": "p", "act": "a", "ctx": "c", "expr": "e", "trig": "t", "sym": "s", "dis": "d"}
def combos(d):
    keys = [k for k in d if k != "form"]; out = []
    for vals in itertools.product(*[d[k] for k in keys]):
        kw = {NAMES[k]: v for k, v in zip(keys, vals)}
        for f in d["form"]:
            try:
                if kw.get("c") and "when" in kw.get("a", "") and kw["c"].startswith("when"): continue
                x = " ".join(f.format(**kw).split()); x = re.sub(r"^[,.\s]+", "", x); x = re.sub(r"\s+([,.])", r"\1", x); x = re.sub(r",\s*,", ",", x).strip(" ,")
                out.append(x)
            except KeyError: pass
    return [(x, None) for x in dict.fromkeys(out)]

# ------------------------- questions.csv (training) -------------------------
test_cores = [(q, e) for (e, a, b) in CORE for q in b] + [(q, e) for (e, tr, te) in EXTRA for q in te]
train_cores = [(q, e) for e, qs in Q_BASE.items() for q in qs] + [(q, e) for (e, a, b) in CORE for q in a] + [(q, e) for (e, tr, te) in EXTRA for q in tr]
used_tr = {norm(r[0]) for r in GOLD_TEST} | {norm(c[0]) for c in test_cores}
q_rows = [[t, p, st, d] for (t, st, d, p) in expand(train_cores, 10, used_tr, rng)]
rng.shuffle(q_rows)

# ------------------------- intents.csv (training) -------------------------
used_int = set(used_tr); intents = []
byent = collections.defaultdict(list)
for t, p, st, d in q_rows: byent[p].append((t, st, d))
srh = []
for e, lst in byent.items(): rng.shuffle(lst); srh += lst[:19]
rng.shuffle(srh); srh = srh[:1250]
intents += [[t, "srh_question", st, d] for t, st, d in srh]
for t, *_ in srh: used_int.add(norm(t))
def add_class(label, bases, n):
    intents.extend([[t, label, st, d] for (t, st, d, _) in expand_n(bases, n, used_int, rng)])
add_class("abuse_disclosure", combos(O.ABUSE_EN_A) + [(x, None) for x in X.ABUSE_X_EN] * 40, 1250)
add_class("self_harm", combos(O.SELF_EN_A) + [(x, None) for x in X.SELF_X_EN] * 40, 1250)
add_class("medical_emergency", combos(O.EMERG_EN_A) + [(x, None) for x in X.EMERG_X_EN] * 40, 1250)
add_class("diagnosis_request", combos(O.DIAG_EN_A) + [(x, None) for x in X.DIAG_X_EN] * 40, 1250)
add_class("off_topic", [(x, None) for x in O.OFF_EN_A], 1250)
for lab, lst in INT_BASE.items():
    for t in lst:
        if norm(t) not in used_int: used_int.add(norm(t)); intents.append([t, lab, "seed", ""])
rng.shuffle(intents)

# ------------------------- test_set.csv (held out) -------------------------
used_te = set(used_tr) | used_int
tests = [[t, e, i, "gold", "", "gold"] for t, e, i in GOLD_TEST]
for t, e, i in GOLD_TEST: used_te.add(norm(t))
tests += [[t, p, "srh_question", st, d, "synthetic"] for (t, st, d, p) in expand(test_cores, 15, used_te, rng)]
def add_test(bases, n, entry, intent):
    tests.extend([[t, entry, intent, st, d, "synthetic"] for (t, st, d, _) in expand_n(bases, n, used_te, rng)])
add_test(combos(O.ABUSE_EN_B), 300, "CRISIS", "abuse_disclosure")
add_test(combos(O.SELF_EN_B), 300, "CRISIS", "self_harm")
add_test(combos(O.EMERG_EN_B), 300, "CRISIS", "medical_emergency")
add_test(combos(O.DIAG_EN_B), 300, "CRISIS", "diagnosis_request")
add_test([(x, None) for x in O.NS_EN], 500, "NOT_SURE", "srh_question")
add_test([(x, None) for x in O.OFF_EN_B], 500, "NOT_SURE", "off_topic")
rng.shuffle(tests)

# ------------------------- redteam_prompts.csv (challenge set) -------------------------
used_rt = set(used_te) | {norm(t[0]) for t in tests}
red = [[p, i, b, "gold", "", "gold"] for (p, i, b) in GOLD_RED if norm(p) not in used_rt]
for p, i, b in GOLD_RED: used_rt.add(norm(p))
alloc = {"self_harm_indirect": 1000, "abuse_indirect": 1000, "emergency_indirect": 700, "diagnosis_demand": 600, "refuse_dosage": 800, "refuse_jailbreak": 1100}
for cat, intent, beh, seeds in RED:
    rows = expand_n([(s, None) for s in seeds], alloc[cat], used_rt, rng)
    red += [[t, intent, beh, cat, d, "synthetic"] for (t, st, d, _) in rows]
rng.shuffle(red)

def w(name, header, rows):
    with open(f"{OUT}/{name}", "w", newline="", encoding="utf-8") as f:
        c = csv.writer(f); c.writerow(header); c.writerows(rows)
w("questions.csv", ["question", "entry_id", "style", "district"], q_rows)
w("intents.csv", ["text", "label", "style", "district"], intents)
w("test_set.csv", ["question", "expected_entry_id", "expected_intent", "style", "district", "source"], tests)
w("redteam_prompts.csv", ["prompt", "expected_intent", "expected_behavior", "category", "district", "source"], red)
import pandas as pd
for n in ["questions", "intents", "test_set", "redteam_prompts"]:
    print(n, len(pd.read_csv(f"{OUT}/{n}.csv")))
print(pd.read_csv(f"{OUT}/intents.csv").label.value_counts().to_dict())
print(pd.read_csv(f"{OUT}/test_set.csv").expected_intent.value_counts().to_dict())
