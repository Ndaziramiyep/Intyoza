import re, csv, hashlib, pathlib, subprocess, sys
SRC = {
 "ITGSE":"UNESCO International Technical Guidance on Sexuality Education",
 "WHO-JHU-FPHandbook-2022Ed-v221114":"WHO/JHU Family Planning Handbook 2022",
 "comprehensive_sexuality_education_cse_final_july162019_1":"REB/UNFPA CSE Reference Book 2018",
 "final-guidelines-for-hiv-prevention-treatment-and-care-in-rwanda-2022-c":"Rwanda HIV Guidelines 2022",
 "who-recommendations-adolescent-sexual-and-reproductive-health-and-rights-2018":"WHO Adolescent SRHR 2018",
 "WHO-recommendations-adolescent-contraception-2017":"WHO Adolescent Contraception 2017",
}
TOPICS = {
 "menstruation": r"menstrua|period\b|periods\b|pad\b|pads\b|menarche|cramp",
 "puberty": r"pubert|adolescen\w+ (changes|development)|wet dream|erection|breast",
 "contraception": r"contracept|condom|pill\b|pills\b|implant|iud\b|injectable|family planning|emergency contracept",
 "pregnancy": r"pregnan|antenatal|birth\b|childbirth|fertil|ovulat",
 "stis": r"\bsti\b|\bstis\b|sexually transmitted|syphilis|gonorrh|chlamydia|herpes|hpv",
 "hiv": r"\bhiv\b|\baids\b|\bart\b|antiretroviral|prep\b|\bpep\b|viral load|pmtct",
 "relationships": r"relationship|peer pressure|gender norm|communication skills|partner",
 "consent_safety": r"consent|abuse|violence|rape|assault|harass|coerc|gbv|protection",
}
RW = re.compile(r"rwanda|rwandan|rbc\b|migeprof|isange|community health insurance|\bREB\b", re.I)
JUNK = re.compile(r"(\.{5,})|^\s*(contents|table of contents|acknowledg|references|abbreviations|annex|index)\b|http|www\.|©|ISBN|doi:", re.I)

def paras(page):
    # join hard-wrapped lines; split on blank lines
    blocks = re.split(r"\n\s*\n", page)
    out = []
    for b in blocks:
        t = re.sub(r"-\n(?=[a-z])", "", b)            # de-hyphenate
        t = re.sub(r"\s*\n\s*", " ", t).strip()
        t = re.sub(r"\s+", " ", t)
        if t: out.append(t)
    return out

# usage: python passages.py <folder with the PDFs>   (needs the pdftotext tool from poppler-utils)
PDFS = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(".")
pathlib.Path("raw").mkdir(exist_ok=True)
for stem in SRC:
    if (PDFS / f"{stem}.pdf").exists():
        subprocess.run(["pdftotext", str(PDFS / f"{stem}.pdf"), f"raw/{stem}.txt"], check=True)
rows, seen = [], set()
for fn, label in SRC.items():
    if not pathlib.Path(f"raw/{fn}.txt").exists():
        continue
    txt = pathlib.Path(f"raw/{fn}.txt").read_text(errors="ignore")
    for pno, page in enumerate(txt.split("\f"), 1):
        buf = []
        def flush():
            global buf
            text = " ".join(buf).strip(); buf = []
            w = text.split()
            if len(w) < 20: return
            alpha = sum(c.isalpha() for c in text) / max(1, len(text))
            if alpha < 0.75 or JUNK.search(text[:200]) or text.count("  ") > 3: return
            key = hashlib.md5(re.sub(r"\W+","",text.lower())[:200].encode()).hexdigest()
            if key in seen: return
            seen.add(key)
            low = text.lower()
            tags = [t for t, p in TOPICS.items() if re.search(p, low)]
            rows.append([label, pno, text, len(w), ";".join(tags[:3]), "yes" if RW.search(text) else "no"])
        for p in paras(page):
            w = p.split()
            if len(w) > 100:   # split long paragraphs on sentence boundaries into ~60-word chunks
                sents, cur = re.split(r"(?<=[.!?])\s+", p), []
                for s in sents:
                    cur.append(s)
                    if len(" ".join(cur).split()) >= 60:
                        buf = [" ".join(cur)]; flush(); cur = []
                if cur: buf = [" ".join(cur)]; flush()
                continue
            buf.append(p)
            if len(" ".join(buf).split()) >= 38: flush()
        if buf: flush()
print(len(rows))
import collections; print(collections.Counter(r[0] for r in rows))
with open("source_passages.csv","w",newline="",encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["passage_id","source","page","text","word_count","topic_tags","mentions_rwanda"])
    for i, r in enumerate(rows, 1):
        w.writerow([f"P-{i:05d}"] + r)
