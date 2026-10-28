"""Central settings. Change values here, not inside other files."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA, MODELS, REPORTS = ROOT / "data", ROOT / "models", ROOT / "reports"

# Matching backend: "classifier" (works everywhere) or "classifier_embed" (adds sentence
# embeddings; needs sentence-transformers). Start with "classifier", try the upgrade on Day 6.
BACKEND = "classifier"
EMBED_MODEL = "all-MiniLM-L6-v2"   # small English sentence-embedding model (optional Day 6 upgrade)

# Confidence gate: below this similarity score we say "I am not sure".
# Scores are not comparable between backends, so each has its own value.
# Tune these on Day 6 with:  python -m src.evaluate --sweep
THRESHOLDS = {"classifier": 0.24, "classifier_embed": 0.24}

# Intents that never get an SRH answer; they get a fixed safe response instead.
SAFE_INTENTS = ["self_harm", "abuse_disclosure", "medical_emergency", "diagnosis_request"]
# Minimum classifier confidence to use the safe response. Low on purpose (safety first):
# a false alarm shows a caring message with help lines, a missed crisis can cost a life.
# Crisis classes use a lower bar than "diagnosis_request". Tuned on test_set.csv (see README, Day 6).
SAFE_MIN_PROB = {"self_harm": 0.20, "abuse_disclosure": 0.20, "medical_emergency": 0.30, "diagnosis_request": 0.30}
OFFTOPIC_MIN_PROB = 0.60

# Launch gate: when INTYOZA_REQUIRE_REVIEWED=1, only entries marked reviewed = yes in
# knowledge_base.csv may be shown to users (everything else becomes "I am not sure").
REQUIRE_REVIEWED = os.getenv("INTYOZA_REQUIRE_REVIEWED", "0") == "1"
