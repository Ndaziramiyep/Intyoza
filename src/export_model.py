"""Package the trained chatbot for another project.   Run:  python -m src.export_model
Writes dist/intyoza_srh.zip containing the self-contained intyoza_srh/ package (model .h5, code, chat page)."""
import sys
import zipfile

from . import config

PKG = config.ROOT / "intyoza_srh"
DIST = config.ROOT / "dist"


def main():
    h5 = PKG / "model" / "intyoza_srh_model.h5"
    if not h5.exists():
        sys.exit("No model yet. Run:  python -m src.train_keras")
    sys.path.insert(0, str(config.ROOT))
    from intyoza_srh import IntyozaChatbot
    bot = IntyozaChatbot(h5, backend="numpy")         # check it loads without TensorFlow
    assert bot.ask("what is a period")["type"] == "answer"
    DIST.mkdir(exist_ok=True)
    out = DIST / "intyoza_srh.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(PKG.rglob("*")):
            # never ship secrets (.env holds the database password) or caches
            if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith(".env"):
                z.write(p, p.relative_to(PKG.parent))
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB) with {len(bot.kb)} answers.")
    print("In the other project: unzip it next to your code, then  pip install -r intyoza_srh/requirements.txt")


if __name__ == "__main__":
    main()
