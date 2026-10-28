"""Make realistic variants of a training question (Rwandan context, greetings, SMS spelling, typos).
Same ideas as tools/data_generation/make_data.py, applied at training time to the seed questions."""
import random
import re

DISTRICTS = ["Gasabo", "Kicukiro", "Nyarugenge", "Burera", "Gakenke", "Gicumbi", "Musanze", "Rulindo", "Gisagara",
             "Huye", "Kamonyi", "Muhanga", "Nyamagabe", "Nyanza", "Nyaruguru", "Ruhango", "Bugesera", "Gatsibo",
             "Kayonza", "Kirehe", "Ngoma", "Nyagatare", "Rwamagana", "Karongi", "Ngororero", "Nyabihu",
             "Nyamasheke", "Rubavu", "Rusizi", "Rutsiro"]
LEVELS = ["S1", "S2", "S3", "S4", "S5", "S6", "P6", "TVET", "university"]
PRE = ["hello, ", "good morning, ", "good evening, ", "please, ", "pls help: ", "hi i have a question. ", "madam, ",
       "sir, ", "i want to know, ", "my friend asked me and i dont know: ", "excuse me, ", "quick question: ",
       "kindly, ", "please help me, ", "our health center nurse did not explain this: ",
       "at the youth corner they told me to ask: ", "i am asking for my sister: ", "please advise, "]
SUF = [" thank you", " please", " pls help", " (i am afraid to ask at home)", " i have nobody to ask", " thanks",
       " i am confused", " explain simply", " kindly help", " thanks in advance", " please reply soon",
       " i cannot ask at school", " i heard it from a classmate"]
SMS = {"please": "pls", "what": "wat", "how": "hw", "you": "u", "your": "ur", "with": "wit", "when": "wen",
       "because": "coz", "are": "r", "before": "b4", "about": "abt", "people": "ppl", "is it": "is t", "okay": "ok"}


def _ctx(rng):
    d, a = rng.choice(DISTRICTS), rng.randint(12, 24)
    return rng.choice([f"i am {a} years old. ", f"i am a {rng.choice(LEVELS)} student in {d}. ",
                       f"i live in {d} district. ", f"im {a} from {d}. ", f"i study at a boarding school in {d}. ",
                       f"i come from a village in {d}. "])


def typo(t, rng):
    ws = t.split(" ")
    idx = [i for i, w in enumerate(ws) if len(w) > 3 and w.isalpha()]
    if not idx:
        return t
    i = rng.choice(idx)
    w = ws[i]
    k = rng.randrange(1, len(w) - 1)
    op = rng.choice(["swap", "drop", "dup"])
    if op == "swap":
        w = w[:k] + w[k + 1] + w[k] + w[k + 2:]
    elif op == "drop":
        w = w[:k] + w[k + 1:]
    else:
        w = w[:k] + w[k] + w[k:]
    ws[i] = w
    return " ".join(ws)


def sms(t, rng):
    for a, b in SMS.items():
        if re.search(rf"\b{a}\b", t, re.I) and rng.random() < 0.7:
            t = re.sub(rf"\b{a}\b", b, t, flags=re.I)
    return t


def variant(q, rng: random.Random):
    t = q.strip().rstrip("?").lower() if rng.random() < 0.7 else q.strip()
    r = rng.random()
    if r < 0.25:
        t = rng.choice(PRE) + t
    elif r < 0.45:
        t = _ctx(rng) + t
    if rng.random() < 0.3:
        t = t + rng.choice(SUF)
    if rng.random() < 0.35:
        t = typo(t, rng)
    if rng.random() < 0.25:
        t = sms(t, rng)
    if rng.random() < 0.05:
        t = t.upper()
    if rng.random() < 0.3:
        t += rng.choice(["?", "??", "!", ""])
    return t


def augment(questions, n, seed=0):
    """Return n variants for each question (originals not included)."""
    rng = random.Random(seed)
    return [[variant(q, rng) for _ in range(n)] for q in questions]
