"""Build fine-tuning triples (anchor=query, positive, negative) from the Train split only.

Train.json's 2,000 queries are split 90/10 *by query* into train / validation, so
no validation query is seen in training. Study.json and Test.json are never read.

Positives: every judged document (relevance 1 or 2) from Train_paper_qrel.tsv
(paper text = "title. abstract") and Train_paragraph_qrel.tsv (paragraph text).
Both levels are mixed into one set so a single model serves both tasks.

Hard negatives (one per positive):
* paragraph positives -> another paragraph of the same paper that is not judged
  relevant for the query: topically close, so the model must learn finer distinctions
  than in-batch negatives alone teach it;
* paper positives -> a paper ranked 10-50 for the query by the *base* dense model
  (skipping the top 10, where unjudged relevant papers are most likely) that is not
  judged relevant. ``--paper-negatives random`` uses a random paper instead (no GPU needed).

    python -m neural.build_training_data --corpus neural_cache/corpus.pkl --queries-dir data --out neural_cache/triples
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from neural.corpus import load_corpus


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 4 and int(parts[3]) > 0:
            qrels[parts[0]][parts[2]] = int(parts[3])
    return qrels


def mine_paper_negatives(queries: dict[str, str], qrels, ids, texts, model_name: str, lo: int, hi: int, seed: int):
    """qid -> list of hard-negative paper ids (ranks lo..hi under the base dense model, judged positives removed)."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = 256
    doc = model.encode_document(texts, batch_size=128, convert_to_tensor=True, normalize_embeddings=True,
                                show_progress_bar=True)
    qids = list(queries)
    q = model.encode_query([queries[x] for x in qids], batch_size=128, convert_to_tensor=True, normalize_embeddings=True)
    out = {}
    for start in range(0, len(qids), 256):
        top = torch.topk(q[start:start + 256] @ doc.T, k=hi, dim=1).indices.cpu().tolist()
        for qid, row in zip(qids[start:start + 256], top):
            out[qid] = [ids[i] for i in row[lo:] if ids[i] not in qrels.get(qid, {})]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="neural_cache/corpus.pkl")
    ap.add_argument("--queries-dir", required=True, help="folder with Train.json and the Train_*_qrel.tsv files")
    ap.add_argument("--out", default="neural_cache/triples")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--paper-negatives", choices=["mined", "random"], default="mined")
    ap.add_argument("--mining-model", default="sentence-transformers/all-distilroberta-v1")
    ap.add_argument("--mine-range", type=int, nargs=2, default=(10, 50), metavar=("LO", "HI"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    qdir = Path(args.queries_dir)
    corpus = load_corpus(args.corpus)
    paper_text = dict(zip(corpus["paper"]["ids"], corpus["paper"]["texts"]))
    para_text = dict(zip(corpus["paragraph"]["ids"], corpus["paragraph"]["texts"]))
    paras_of = defaultdict(list)
    for pid in para_text:
        paras_of[pid.rsplit("_", 1)[0]].append(pid)

    queries = {str(q["query_id"]): q["query"] for q in json.loads((qdir / "Train.json").read_text(encoding="utf-8"))}
    paper_qrels = read_qrels(qdir / "Train_paper_qrel.tsv")
    para_qrels = read_qrels(qdir / "Train_paragraph_qrel.tsv")

    qids = sorted(queries, key=int)
    rng.shuffle(qids)
    n_val = int(len(qids) * args.val_ratio)
    split = {"val": set(qids[:n_val]), "train": set(qids[n_val:])}

    if args.paper_negatives == "mined":
        mined = mine_paper_negatives(queries, paper_qrels, corpus["paper"]["ids"], corpus["paper"]["texts"],
                                     args.mining_model, *args.mine_range, args.seed)
    else:
        mined = {}
    all_papers = corpus["paper"]["ids"]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, keep in split.items():
        rows, skipped = [], 0
        for qid in sorted(keep, key=int):
            q = queries[qid]
            for doc_id in paper_qrels.get(qid, {}):
                if doc_id not in paper_text:
                    skipped += 1
                    continue
                pool = mined.get(qid) or [p for p in rng.sample(all_papers, 20) if p not in paper_qrels[qid]]
                rows.append({"qid": qid, "level": "paper", "anchor": q, "positive": paper_text[doc_id],
                             "negative": paper_text[rng.choice(pool)]})
            for doc_id in para_qrels.get(qid, {}):
                if doc_id not in para_text:
                    skipped += 1
                    continue
                siblings = [p for p in paras_of[doc_id.rsplit("_", 1)[0]] if p not in para_qrels[qid]]
                neg = rng.choice(siblings) if siblings else rng.choice(corpus["paragraph"]["ids"])
                rows.append({"qid": qid, "level": "paragraph", "anchor": q, "positive": para_text[doc_id],
                             "negative": para_text[neg]})
        rng.shuffle(rows)
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[name] = {"queries": len(keep), "triples": len(rows), "missing_docs": skipped}
        print(f"{name}: {len(keep)} queries, {len(rows)} triples ({skipped} judged docs not in corpus)")
    (out / "split.json").write_text(json.dumps({k: sorted(v, key=int) for k, v in split.items()} | {"stats": stats},
                                               indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
