# Quick start

```bash
bash install.sh                                   # conda env "ir-part1" + NLTK stopwords
conda activate ir-part1

# Search interface (http://127.0.0.1:5000). First start indexes the collection and writes index_cache.pkl.
bash run.sh app /mnt/netstore1_datasets/IR26/<paper_json_dir>

# All eight Study/Test result files (+ metrics and significance tests if qrels are given)
bash run.sh experiments DATA_DIR QUERIES_DIR TEAM [Study_paper_qrel.tsv Study_paragraph_qrel.tsv]
```

Result files go to `Retrieval Results/` using the exact names from the assignment. Average search time per
query is printed and saved to `search_times.json`.

To label the top paragraph of each Test query (the `TEAM_qrel_part1_test.tsv` deliverable), start the app with
`python app.py --data-dir DATA --queries-dir QUERIES --team TEAM --best-model bm25` and open `/annotate`.
Labels are saved on every click.

| File | Role |
| --- | --- |
| `tokenizationandtfidf.py` | T1 tokenizer (`tokenize`) |
| `t2_framework/` | T2 TF-IDF / BM25 scoring and the public `search` API |
| `pipeline.py` | Loads per-paper JSON, builds paper/paragraph indexes (cached), configures the API |
| `app.py`, `templates/`, `static/` | Flask search interface and annotation page |
| `run_experiments.py`, `eval_metrics.py` | Result-file generation and evaluation |

Paper-level documents index title + abstract; paragraph-level documents index each paragraph.

# COS487 T2 Retrieval API

T2 owns the scoring and ranking layer. T1 owns document extraction, tokenization, and index/statistics construction. T4 imports the public interface directly from `t2_framework/retrieval.py`; a separate `api.py` wrapper is unnecessary.

## Public interface T4 calls

```python
from t2_framework.retrieval import search, get_paper, get_paragraph, tokenize

results = search("visual diffusion model replication", model="bm25", level="paper", k=100)
# [{"id": "2408.00001", "score": 12.3, "rank": 1}, ...]
paper = get_paper("2408.00001")
# {"paper_id": "2408.00001", "title": "...", "abstract": "..."}
paragraph = get_paragraph("2408.00001_1")
# {"para_id": "2408.00001_1", "paper_id": "2408.00001", "text": "..."}
tokens = tokenize("Query text")  # calls the configured T1 tokenizer
```

`configure_retrieval(...)` is an application-startup function, not a per-query call. The application must call it once with T1's real `CollectionStats` for both `paper` and `paragraph`, the full per-paper JSON records, and T1's tokenizer. The four public functions above do not guess paths or load mock data implicitly.

## T1 → T2 startup configuration

T1's collection export must provide both `paper` and `paragraph`. Each collection uses this exact schema:

```json
{
  "inverted_index": {"token": {"document_id": 2}},
  "df": {"token": 4},
  "doc_lengths": {"document_id": 120},
  "num_docs": 100,
  "avg_doc_length": 120.5
}
```

Each per-paper source JSON should use this schema (matching the supplied `2408.00001.json`):

```json
{
  "paper_id": "2408.00001",
  "title": "Paper title",
  "abstract": "Paper abstract",
  "paragraphs": {
    "2408.00001_1": "Paragraph text..."
  }
}
```

At startup, convert each collection payload to `CollectionStats` and load the source records by paper ID, then configure the API:

```python
import json
from pathlib import Path
from t2_framework import CollectionStats
from t2_framework.retrieval import configure_retrieval
from t1_preprocessing import tokenize  # replace with T1's actual import

bundle = json.loads(Path("t1_index_bundle.json").read_text(encoding="utf-8"))
paper_collection = CollectionStats.from_mapping(bundle["paper"])
paragraph_collection = CollectionStats.from_mapping(bundle["paragraph"])

records = {}
for path in Path("paper_json").glob("*.json"):
    record = json.loads(path.read_text(encoding="utf-8"))
    records[str(record["paper_id"])] = record

configure_retrieval(
    paper_collection=paper_collection,
    paragraph_collection=paragraph_collection,
    paper_records=records,
    tokenizer=tokenize,
)
```

The `t1_preprocessing` module name and paths above are placeholders to replace with T1's actual names. This configuration intentionally raises errors if files, fields, IDs, or tokenizer do not match; it does not substitute mock data.

## Scoring details

- `t2_framework/scoring.py` implements TF-IDF and BM25 directly. It does not use `scikit-learn`, `PyTerrier`, or `rank-bm25`.
- TF-IDF uses the course-demo weighting `tf_weight = 1 + log10(tf)`, `idf = log10(N / df)`, and cosine normalization.
- BM25 defaults to `k1=1.5`, `b=0.75` and positive RSJ IDF `ln(1 + (N-df+0.5)/(df+0.5))`.
- If the instructor/team specifies a different exact formula variant, update and document it before official runs.
- Query tokenization must be T1's exact tokenizer, including stemming, stopwords, case folding, and punctuation rules.

## Errors are intentional

- API search before `configure_retrieval`: `RuntimeError` explains startup configuration is missing.
- Missing collection fields or mismatched IDs: `ValueError` names the missing/mismatched data.
- Unknown model/level or negative `k`: `ValueError`.
- Unknown paper/paragraph IDs: `KeyError`.
- Missing JSON paths: `FileNotFoundError`.

No public API method automatically loads mock data. Mock data exists only for explicitly running tests.

## Run tests

From the project directory:

```bash
python -m unittest discover -s tests -v
```

The tests use small declared fixtures to test contracts and file formats. They do not validate retrieval quality over the real corpus

## TREC output

` t2_framework/io_utils.py ` writes six-column TREC runs: `qid Q0 docid rank score run_id`. Generate the eight official files only after configuring the API with T1's real collections and tokenizer.
