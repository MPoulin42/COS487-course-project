"""Cleaned document texts for the neural models.

Neural encoders use their own subword tokenizers, so pre-processing here is
limited to fixing the data noise documented in the Part 1 report:

* Titles: template metadata glued onto ~12% of titles ("...AnsweringDOI: XXXX",
  "Preprint: APS/123-QED", "Thanks: <e-mails>", "Conference: ...", "CCS: ...")
  is cut off at the first such marker.
* All text: HTML entities unescaped, stray tags removed, whitespace collapsed.
* Paragraphs: near-empty paragraphs (< MIN_CHARS characters, e.g. "Proof.",
  equation residue; measured in characters, not words, because some extracted
  paragraphs lost all their spaces) are left out of the paragraph index; none of them is judged
  relevant in the Train or Study qrels. Duplicate boilerplate (text occurring
  >= BOILERPLATE_MIN times, e.g. NeurIPS checklist answers) is counted but KEPT:
  some Train queries ask about exactly that content (e.g. negative societal
  impacts) and have checklist copies judged fully relevant.

Paper-level documents are "title. abstract" (same unit as Part 1); paragraph-level
documents are single paragraphs.

    python -m neural.corpus --data-dir "data/JSON Files" --out neural_cache/corpus.pkl
"""

from __future__ import annotations

import argparse
import html
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

BOILERPLATE_MIN = 5
MIN_CHARS = 25

_TITLE_META = re.compile(
    r"(?:Preprint|Thanks|Conference|DOI|CCS|Keywords|Journal|Submitted|Accepted|Copyright|Equal contribution)\s*:.*$",
    re.S,
)
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = html.unescape(text or "")
    if "<" in text:
        text = _TAG.sub(" ", text)
    return _WS.sub(" ", text).strip()


def clean_title(title: str) -> str:
    title = clean_text(title)
    cut = _TITLE_META.sub("", title).strip()
    return cut or title  # never blank out a title entirely


def build_corpus(records: Dict[str, dict]) -> dict:
    """records: paper_id -> {"title", "abstract", "paragraphs": {para_id: text}} (pipeline.load_records)."""
    paper_ids: List[str] = []
    paper_texts: List[str] = []
    titles: Dict[str, str] = {}
    for pid in sorted(records):
        r = records[pid]
        title = clean_title(r["title"])
        titles[pid] = title
        paper_ids.append(pid)
        paper_texts.append(f"{title}. {clean_text(r['abstract'])}".strip())

    paras = [(para_id, clean_text(text)) for pid in sorted(records) for para_id, text in records[pid]["paragraphs"].items()]
    counts = Counter(text for _, text in paras)
    para_ids, para_texts = [], []
    dropped = {"short": 0, "boilerplate_kept": 0}
    for para_id, text in paras:
        if len(text) < MIN_CHARS:
            dropped["short"] += 1
            continue
        if counts[text] >= BOILERPLATE_MIN:
            dropped["boilerplate_kept"] += 1
        para_ids.append(para_id)
        para_texts.append(text)
    return {
        "paper": {"ids": paper_ids, "texts": paper_texts},
        "paragraph": {"ids": para_ids, "texts": para_texts},
        "titles": titles,
        "dropped": dropped,
        "total_paragraphs": len(paras),
    }


def load_corpus(path: str | Path) -> dict:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def get_corpus(data_dir: str | Path, cache: str | Path, rebuild: bool = False) -> dict:
    """Load the cleaned corpus from ``cache`` or build it from the per-paper JSON files."""
    cache = Path(cache)
    if cache.exists() and not rebuild:
        return load_corpus(cache)
    import pipeline  # Part 1 loader (reads every per-paper JSON)

    corpus = build_corpus(pipeline.load_records(data_dir))
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as f:
        pickle.dump(corpus, f, protocol=pickle.HIGHEST_PROTOCOL)
    return corpus


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="neural_cache/corpus.pkl")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    c = get_corpus(args.data_dir, args.out, args.rebuild)
    print(f"{len(c['paper']['ids'])} papers, {len(c['paragraph']['ids'])} of {c['total_paragraphs']} paragraphs kept "
          f"(dropped {c['dropped']['short']} under {MIN_CHARS} characters; {c['dropped']['boilerplate_kept']} boilerplate copies kept) "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
