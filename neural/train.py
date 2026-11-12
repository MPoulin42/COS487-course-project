"""Fine-tune the dense or the sparse model on the Train triples.

Dense : sentence-transformers bi-encoder (default all-distilroberta-v1, mean pooling,
        cosine similarity), MultipleNegativesRankingLoss: for each query the positive
        must outscore its hard negative and every other document in the batch.
Sparse: SPLADE (default naver/splade_v2_max: BERT MLM head, log(1+ReLU) max pooling
        over the vocabulary), SpladeLoss = SparseMultipleNegativesRankingLoss + FLOPS
        regularisation on query and document vectors to keep them sparse.

Training and validation loss are logged every --log-steps steps; the checkpoint with
the lowest validation loss is kept (saved to OUT/final). The full log goes to
OUT/loss_history.json and the plot to OUT/loss.png (train + validation per step).

    python -m neural.train --model dense  --triples neural_cache/triples --out models/dense
    python -m neural.train --model sparse --triples neural_cache/triples --out models/sparse
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULTS = {
    "dense": {"base": "sentence-transformers/all-distilroberta-v1", "lr": 2e-5, "batch": 32},
    "sparse": {"base": "naver/splade_v2_max", "lr": 2e-5, "batch": 16},
}


def load_triples(path: Path, limit: int = 0):
    from datasets import Dataset

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit:
        rows = rows[:limit]
    return Dataset.from_dict({k: [r[k] for r in rows] for k in ("anchor", "positive", "negative")})


def plot_loss(history: list[dict], out_png: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tr = [(h["step"], h["loss"]) for h in history if "loss" in h]
    va = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    if tr:
        ax.plot(*zip(*tr), label="training loss", linewidth=1.6)
    if va:
        ax.plot(*zip(*va), label="validation loss", marker="o", markersize=3.5, linewidth=1.6)
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["dense", "sparse"], required=True)
    ap.add_argument("--base-model", help="Hugging Face id (default depends on --model)")
    ap.add_argument("--triples", default="neural_cache/triples")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--max-seq-length", type=int, default=256)
    ap.add_argument("--log-steps", type=int, default=20, help="training-loss logging and validation interval")
    ap.add_argument("--doc-reg", type=float, default=3e-5, help="SPLADE document FLOPS weight")
    ap.add_argument("--query-reg", type=float, default=5e-5, help="SPLADE query FLOPS weight")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N triples (smoke tests)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch

    cfg = DEFAULTS[args.model]
    base = args.base_model or cfg["base"]
    batch = args.batch_size or cfg["batch"]
    lr = args.lr or cfg["lr"]
    out = Path(args.out)
    cuda = torch.cuda.is_available()

    train_ds = load_triples(Path(args.triples) / "train.jsonl", args.limit)
    val_ds = load_triples(Path(args.triples) / "val.jsonl", args.limit // 5 if args.limit else 0)
    print(f"{len(train_ds)} train / {len(val_ds)} validation triples; base model {base}; device {'cuda' if cuda else 'cpu'}")

    try:
        from sentence_transformers.sentence_transformer.training_args import BatchSamplers
    except ImportError:  # sentence-transformers < 6
        from sentence_transformers.training_args import BatchSamplers
    common = dict(
        output_dir=str(out / "checkpoints"), num_train_epochs=args.epochs, per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch, learning_rate=lr, warmup_ratio=0.1, fp16=cuda, seed=args.seed,
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # identical texts in one batch would be false negatives
        eval_strategy="steps", eval_steps=args.log_steps, logging_steps=args.log_steps, logging_first_step=True,
        save_strategy="steps", save_steps=args.log_steps, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False, report_to="none",
    )

    if args.model == "dense":
        from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                           SentenceTransformerTrainingArguments, losses)

        model = SentenceTransformer(base, device="cuda" if cuda else "cpu")
        model.max_seq_length = args.max_seq_length
        loss = losses.MultipleNegativesRankingLoss(model)
        trainer_cls, targs = SentenceTransformerTrainer, SentenceTransformerTrainingArguments(**common)
    else:
        from sentence_transformers.sparse_encoder import (SparseEncoder, SparseEncoderTrainer,
                                                          SparseEncoderTrainingArguments, losses)

        model = SparseEncoder(base, device="cuda" if cuda else "cpu")
        model.max_seq_length = args.max_seq_length
        loss = losses.SpladeLoss(model, loss=losses.SparseMultipleNegativesRankingLoss(model),
                                 document_regularizer_weight=args.doc_reg, query_regularizer_weight=args.query_reg)
        trainer_cls, targs = SparseEncoderTrainer, SparseEncoderTrainingArguments(**common)

    trainer = trainer_cls(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds, loss=loss)
    before = trainer.evaluate()  # validation loss of the untuned model (step 0)
    trainer.train()
    model.save_pretrained(str(out / "final"))

    history = [{"step": 0, "eval_loss": before["eval_loss"]}] + [
        h for h in trainer.state.log_history if "loss" in h or "eval_loss" in h
    ]
    (out / "loss_history.json").write_text(json.dumps({
        "model": args.model, "base_model": base, "epochs": args.epochs, "batch_size": batch, "lr": lr,
        "max_seq_length": args.max_seq_length, "train_triples": len(train_ds), "val_triples": len(val_ds),
        "steps_per_epoch": trainer.state.max_steps / max(args.epochs, 1e-9),
        "best_checkpoint": trainer.state.best_model_checkpoint, "best_eval_loss": trainer.state.best_metric,
        "history": history,
    }, indent=1), encoding="utf-8")
    plot_loss(history, out / "loss.png", f"{args.model.capitalize()} model ({base.split('/')[-1]}) fine-tuning")
    print(f"Saved {out / 'final'}; best validation loss {trainer.state.best_metric}; plot {out / 'loss.png'}")


if __name__ == "__main__":
    main()
