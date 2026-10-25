"""Fine-tuning: contrastive bi-encoder and cross-encoder reranker.

Bi-encoder
    (input text, occupation document) pairs trained with
    MultipleNegativesRankingLoss. With hard negatives on, each pair also gets a
    negative document drawn from the gold occupation's nearest neighbours
    (cosine similarity of the *base* model's occupation embeddings), excluding
    every gold code of the anchor. ``NO_DUPLICATES`` batching keeps the same
    occupation document from appearing twice in a batch, which would otherwise
    turn a positive into an in-batch false negative.

Cross-encoder
    Reranks the bi-encoder's top-k. Trained listwise (cross-encoder
    MultipleNegativesRankingLoss) on (query, gold doc, 4 hard negatives) where
    the negatives come from the fine-tuned bi-encoder's own top-k.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

log = logging.getLogger("onet_analysis.matching.train")

N_NEIGHBORS = 10
CE_HARD_NEGATIVES = 4


def occupation_neighbors(doc_embeddings: np.ndarray, k: int = N_NEIGHBORS) -> np.ndarray:
    """(n_occ, k) indices of each occupation's most similar *other* occupations."""
    sim = doc_embeddings @ doc_embeddings.T
    np.fill_diagonal(sim, -np.inf)
    return np.argsort(-sim, axis=1)[:, :k]


def build_biencoder_examples(
    train_pairs: pd.DataFrame,
    socs: Sequence[str],
    docs: Sequence[str],
    *,
    neighbors: np.ndarray | None = None,
    seed: int = 13,
) -> pd.DataFrame:
    """Columns ``anchor, positive[, negative]`` from exploded (anchor, soc) pairs."""
    rng = np.random.default_rng(seed)
    pos = {s: i for i, s in enumerate(socs)}
    rows = []
    for r in train_pairs.itertuples(index=False):
        if r.soc not in pos:
            continue
        i = pos[r.soc]
        row = {"anchor": r.anchor, "positive": docs[i]}
        if neighbors is not None:
            gold = {pos[g] for g in str(r.gold).split(";") if g in pos}
            cands = [j for j in neighbors[i] if j not in gold]
            if not cands:
                continue
            row["negative"] = docs[int(rng.choice(cands))]
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _st_losses():
    try:
        from sentence_transformers.sentence_transformer import losses
    except ImportError:  # pragma: no cover - older sentence-transformers
        from sentence_transformers import losses
    return losses


def _batch_samplers():
    try:
        from sentence_transformers.sentence_transformer.training_args import BatchSamplers
    except ImportError:  # pragma: no cover - older sentence-transformers
        BatchSamplers = _batch_samplers()
    return BatchSamplers


def train_biencoder(
    base_model: str,
    examples: pd.DataFrame,
    out_dir: str | Path,
    *,
    epochs: float = 1.0,
    batch_size: int = 64,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    max_seq_length: int = 256,
    max_steps: int = -1,
    bf16: bool = False,
    seed: int = 13,
    model=None,
) -> dict:
    """Fine-tune a sentence-transformers bi-encoder; returns a timing/config record."""
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    BatchSamplers = _batch_samplers()

    out_dir = Path(out_dir)
    model = model or SentenceTransformer(base_model)
    model.max_seq_length = max_seq_length
    loss = _st_losses().MultipleNegativesRankingLoss(model)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_ratio,  # a float < 1 is a ratio in transformers >= 5
        bf16=bf16,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        seed=seed,
        dataloader_drop_last=True,
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, loss=loss,
                                         train_dataset=Dataset.from_pandas(examples, preserve_index=False))
    t0 = time.perf_counter()
    trainer.train()
    seconds = time.perf_counter() - t0
    model.save(str(out_dir))
    record = {
        "base_model": base_model, "examples": len(examples), "columns": list(examples.columns),
        "epochs": epochs, "max_steps": max_steps, "batch_size": batch_size, "learning_rate": learning_rate,
        "max_seq_length": max_seq_length, "bf16": bf16, "train_seconds": seconds,
        "steps": int(trainer.state.global_step),
        "loss_history": [h for h in trainer.state.log_history if "loss" in h],
    }
    (out_dir / "training.json").write_text(json.dumps(record, indent=2))
    log.info("Trained bi-encoder on %d examples in %.0fs -> %s", len(examples), seconds, out_dir)
    return record


def build_reranker_examples(
    queries: Sequence[str],
    golds: Sequence[str],
    candidate_idx: np.ndarray,
    socs: Sequence[str],
    docs: Sequence[str],
    *,
    n_negatives: int = CE_HARD_NEGATIVES,
    seed: int = 13,
) -> pd.DataFrame:
    """(anchor, positive, negative_1..n) rows; negatives = non-gold retriever candidates."""
    rng = np.random.default_rng(seed)
    pos = {s: i for i, s in enumerate(socs)}
    rows = []
    for q, gold, cands in zip(queries, golds, candidate_idx):
        gold_idx = [pos[g] for g in str(gold).split(";") if g in pos]
        if not gold_idx:
            continue
        negs = [j for j in cands if j not in set(gold_idx)][:n_negatives]
        if len(negs) < n_negatives:
            continue
        row = {"anchor": q, "positive": docs[int(rng.choice(gold_idx))]}
        row.update({f"negative_{k + 1}": docs[j] for k, j in enumerate(negs)})
        rows.append(row)
    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def train_reranker(
    base_model: str,
    examples: pd.DataFrame,
    out_dir: str | Path,
    *,
    epochs: float = 1.0,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    max_length: int = 320,
    in_batch_negatives: int = 2,
    max_steps: int = -1,
    bf16: bool = False,
    seed: int = 13,
    model=None,
) -> dict:
    """Fine-tune a cross-encoder listwise over hard negatives."""
    from datasets import Dataset
    from sentence_transformers import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder import losses as ce_losses
    BatchSamplers = _batch_samplers()

    out_dir = Path(out_dir)
    model = model or CrossEncoder(base_model, max_length=max_length, num_labels=1)
    loss = ce_losses.MultipleNegativesRankingLoss(model, num_negatives=in_batch_negatives)
    args = CrossEncoderTrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_ratio,  # a float < 1 is a ratio in transformers >= 5
        bf16=bf16,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        seed=seed,
        dataloader_drop_last=True,
    )
    trainer = CrossEncoderTrainer(model=model, args=args, loss=loss,
                                  train_dataset=Dataset.from_pandas(examples, preserve_index=False))
    t0 = time.perf_counter()
    trainer.train()
    seconds = time.perf_counter() - t0
    model.save(str(out_dir))
    record = {
        "base_model": base_model, "examples": len(examples), "epochs": epochs, "max_steps": max_steps,
        "batch_size": batch_size, "learning_rate": learning_rate, "max_length": max_length,
        "hard_negatives": sum(c.startswith("negative_") for c in examples.columns),
        "in_batch_negatives": in_batch_negatives, "bf16": bf16, "train_seconds": seconds,
        "steps": int(trainer.state.global_step),
        "loss_history": [h for h in trainer.state.log_history if "loss" in h],
    }
    (out_dir / "training.json").write_text(json.dumps(record, indent=2))
    log.info("Trained reranker on %d examples in %.0fs -> %s", len(examples), seconds, out_dir)
    return record
