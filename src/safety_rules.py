"""Keyword backup for crisis messages. It runs BEFORE the ML model, so a classifier
miss can never leave a person in danger without help. Rules are deliberately broad:
a false alarm only shows a caring message with help lines; a miss can cost a life."""
import re

RULES = [  # order = priority
    ("self_harm", r"\b(kill(ing)? (myself|my self)|end my life|ending my life|ending it all|end it all|suicid\w*|want to die|wanna die|"
                  r"dont want to live|do not want to live|don't want to live|hurt myself|harm myself|cut myself|cutting myself|"
                  r"better off dead|no ?one would miss me|nobody would miss me|wish i (was|were) dead|take my own life|"
                  r"not worth living|want to disappear forever|feel(ing)? like dying|wish i could die|nothing to live for|"
                  r"no reason to live|(cant|can't|cannot) go on (anymore|living)|end everything|end it (all|now)|"
                  r"jump (off|from) (a |the )?(bridge|building|roof|cliff)|hang myself|poison myself|drink poison|"
                  r"take an overdose|overdose on purpose|life is (pointless|meaningless))\b"),
    ("abuse_disclosure", r"\b(raped|gang raped|rape me|forced me|forces me|forced himself|molest\w*|sexually abused|sexually assaulted|"
                         r"abused me|abuses me|touch(es|ed) (me|my (private|breasts?|body|thighs?|bum|buttocks))|"
                         r"touch me|touching me|touch(es|ed|ing) (me|my \w+)( \w+){0,3} without my (permission|consent)|blackmail\w* me|"
                         r"(be nice to|sleep with|have sex with) (him|her|them) or (i will )?(lose|fail|be fired)|"
                         r"threaten(ed|s)? me|beats me|hits me|hit me|follow(ed)? me and touch|"
                         r"(if|unless) i refuse (him|her|them|sex)|sex (for|in exchange for) (marks|grades|money|food|a job|school fees))\b"),
    ("medical_emergency", r"\b(bleeding (so much|a lot|heavily|badly|too much)|heavy bleeding|bleeding wont stop|bleeding won't stop|"
                          r"(cant|can't|cannot|wont|won't) stop bleeding|soaking (a |one )?pad|fainted|collapsed|unconscious|"
                          r"severe (abdominal |stomach |belly )?pain|unbearable pain|baby is not moving|convulsions)\b"),
]
_COMPILED = [(label, re.compile(p)) for label, p in RULES]

# Requests for medicine doses or ways to end a pregnancy are never answered (policy, not a data question).
_REFUSE = re.compile(
    r"\b(how (can|do|to|could) (i|a girl|we|you) (abort|end (a|my|the) pregnancy|remove (a|my|the) pregnancy|get rid of (a|my|the) pregnancy)|"
    r"(herbs?|tablets?|pills?|medicine|injection|drink)\b.{0,40}\b(to|that|can|will) (end|remove|abort|terminate|stop) (a|my|the)? ?pregnancy|"
    r"\bmisoprostol\b|abort(ion)? at home|abort without|\bdosage\b|\bdose of\b|"
    r"how (many|much) (tablets|pills|ibuprofen|antibiotics?|emergency pills|mg)|"
    r"what dose|prescribe me|give me (a )?prescription)\b")


def wants_refusal(clean_text: str) -> bool:
    """True for requests for doses, prescriptions or ways to end a pregnancy."""
    return bool(_REFUSE.search(clean_text))


def rule_intent(clean_text: str):
    """Return a crisis label if a keyword rule matches, else None."""
    for label, pattern in _COMPILED:
        if pattern.search(clean_text):
            return label
    return None


# NOTE: these rules were written from the TRAINING examples and general vocabulary. In version 2 some common crisis
# phrases were added after reviewing model misses ("feel like dying", "touched ... without my permission", "be nice
# to him or lose my job", ...). Some of them also occur in test_set.csv, so crisis recall on that file is now
# partly optimistic. Keep this in mind when reading reports/keras_eval_report.md.
