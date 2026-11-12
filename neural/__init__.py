"""Part 2: neural retrieval (fine-tuned dense bi-encoder and fine-tuned SPLADE sparse encoder).

Modules
    corpus.py               cleaned paper / paragraph texts, cached as one pickle
    build_training_data.py  (query, positive, hard negative) triples from the Train qrels, 90/10 query split
    train.py                fine-tunes the dense or sparse model; writes loss history + loss plot
    encode.py               encodes the whole collection into an on-disk index (sharded, resumable)
    retrievers.py           DenseRetriever / SparseRetriever: top-k search over those indexes
"""
