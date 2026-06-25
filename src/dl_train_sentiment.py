"""
DistilBERT Training Loop for Per-Aspect Sentiment Classification.

Trains a separate SentimentClassificationModel for each of the 15 aspects.
Each model classifies feedback into positive(0)/negative(1)/neutral(2).
"""

import json
import logging
import os
import time
import tracemalloc
from collections import Counter
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from sklearn.metrics import f1_score, accuracy_score
from transformers import get_linear_schedule_with_warmup

from src.config import load_config, resolve_path
from src.dl_model import SentimentClassificationModel, count_parameters

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MIN_TRAINING_SAMPLES = 20


def collate_fn(batch):
    """Custom collate for sentiment datasets."""
    input_ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
    attention_mask = torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def compute_class_weights(dataset, num_classes: int = 3) -> torch.Tensor:
    """
    Compute class weights inversely proportional to class frequency.
    weight_j = n_samples / (n_classes * count_j)
    """
    labels = dataset["label"]
    counts = Counter(labels)
    n_samples = len(labels)
    weights = torch.zeros(num_classes, dtype=torch.float32)

    for cls_id in range(num_classes):
        count = counts.get(cls_id, 1)
        weights[cls_id] = n_samples / (num_classes * count)

    return weights


def evaluate_sentiment(model, dataloader, criterion, device) -> tuple:
    """Evaluate sentiment model. Returns (avg_loss, accuracy, macro_f1)."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(len(dataloader), 1)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, accuracy, macro_f1


def train_single_aspect(aspect: str, ds_dict, config: dict, device: torch.device) -> dict:
    """
    Train a SentimentClassificationModel for a single aspect.

    Returns:
        dict with model state, metrics, and training info
    """
    dl_config = config["dl_training"]
    seed = config["seed"]
    epochs = 3  # Fewer epochs for per-aspect models (less data)
    batch_size = dl_config["batch_size"]
    lr = dl_config["learning_rate"]
    weight_decay = dl_config["weight_decay"]
    warmup_ratio = dl_config["warmup_ratio"]
    patience = dl_config["early_stopping_patience"]
    model_name = dl_config["model_name"]

    # Check minimum samples
    n_train = len(ds_dict["train"])
    if n_train < MIN_TRAINING_SAMPLES:
        # Majority class fallback
        labels = ds_dict["train"]["label"]
        majority = Counter(labels).most_common(1)[0][0]
        sentiment_names = {0: "positive", 1: "negative", 2: "neutral"}
        logger.warning(f"  ⚠️  SKIPPED: {n_train} samples < {MIN_TRAINING_SAMPLES}. "
                       f"Fallback: majority class = '{sentiment_names[majority]}'")
        return {
            "status": "skipped",
            "fallback_class": majority,
            "n_train": n_train,
            "val_macro_f1": None,
            "model_state": None,
        }

    # DataLoaders
    train_loader = DataLoader(ds_dict["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(ds_dict["val"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn) if "val" in ds_dict else None

    # Class weights
    class_weights = compute_class_weights(ds_dict["train"]).to(device)

    # Model
    torch.manual_seed(seed)
    model = SentimentClassificationModel(num_classes=3, model_name=model_name).to(device)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Training loop
    best_val_macro_f1 = 0.0
    best_state = None
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        if val_loader:
            val_loss, val_acc, val_macro_f1 = evaluate_sentiment(model, val_loader, criterion, device)
            logger.info(f"    Epoch {epoch}/{epochs} — Train Loss: {avg_train_loss:.4f} | "
                        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro-F1: {val_macro_f1:.4f}")

            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                best_state = deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                logger.info(f"    Early stopping at epoch {epoch}")
                break
        else:
            logger.info(f"    Epoch {epoch}/{epochs} — Train Loss: {avg_train_loss:.4f} (no val set)")
            best_state = deepcopy(model.state_dict())

    train_time = time.time() - start_time

    return {
        "status": "trained",
        "model_state": best_state,
        "n_train": n_train,
        "val_macro_f1": round(best_val_macro_f1, 4),
        "training_time_s": round(train_time, 2),
    }


def plot_f1_per_aspect(results: dict, save_path: str):
    """Plot macro-F1 per aspect as horizontal bar chart."""
    aspects = []
    f1_scores = []

    for aspect, info in sorted(results.items(), key=lambda x: x[1].get("val_macro_f1", 0) or 0):
        aspects.append(aspect.replace("_", " "))
        f1_scores.append(info.get("val_macro_f1") or 0.0)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#e74c3c" if f < 0.5 else "#f39c12" if f < 0.7 else "#2ecc71" for f in f1_scores]
    bars = ax.barh(aspects, f1_scores, color=colors, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, f1_scores):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}" if val > 0 else "SKIP", va="center", fontsize=9)

    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Validation Macro-F1", fontsize=12)
    ax.set_title("DistilBERT Sentiment — Val Macro-F1 per Aspect", fontsize=14, fontweight="bold")
    ax.axvline(0.7, color="gray", linestyle="--", alpha=0.5, label="0.7 threshold")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info(f"F1 per-aspect plot saved: {save_path}")


def main():
    logger.info("=" * 70)
    logger.info("DISTILBERT PER-ASPECT SENTIMENT — TRAINING")
    logger.info("=" * 70)

    # ─── Config ───────────────────────────────────────────────────────────
    config = load_config()
    aspect_labels = config["labels"]["aspects"]
    data_dir = os.path.dirname(resolve_path(config["data"]["cleaned"]))
    output_dir = resolve_path(config["outputs"]["models"])
    eda_dir = resolve_path(config["outputs"]["eda"])
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(eda_dir, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Aspects: {len(aspect_labels)}")

    # Set seed
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # ─── Train per-aspect models ──────────────────────────────────────────
    all_results = {}
    all_model_states = {}

    tracemalloc.start()
    total_start = time.time()

    for i, aspect in enumerate(aspect_labels, 1):
        logger.info("")
        logger.info(f"{'─' * 70}")
        logger.info(f"[{i:02d}/15] ASPECT: {aspect}")
        logger.info(f"{'─' * 70}")

        # Load aspect-specific dataset
        ds_path = os.path.join(data_dir, "hf_datasets", f"sentiment_{aspect}")
        if not os.path.exists(ds_path):
            logger.warning(f"  Dataset not found: {ds_path} — skipping")
            all_results[aspect] = {"status": "missing", "val_macro_f1": None}
            continue

        ds_dict = load_from_disk(ds_path)
        logger.info(f"  Loaded: train={len(ds_dict.get('train', []))}, "
                    f"val={len(ds_dict.get('val', []))}, "
                    f"test={len(ds_dict.get('test', []))}")

        # Train
        result = train_single_aspect(aspect, ds_dict, config, device)
        all_results[aspect] = result

        if result["model_state"] is not None:
            all_model_states[aspect] = result["model_state"]
        elif result["status"] == "skipped":
            all_model_states[aspect] = {"type": "fallback", "prediction": result["fallback_class"]}

    total_time = time.time() - total_start
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # ─── Save All Models ──────────────────────────────────────────────────
    logger.info("")
    logger.info("─" * 70)
    logger.info("SAVING ARTIFACTS")
    logger.info("─" * 70)

    save_path = os.path.join(output_dir, "sentiment_classifiers_distilbert.pt")
    torch.save(all_model_states, save_path)
    logger.info(f"All models saved: {save_path}")

    # ─── Plot F1 per Aspect ───────────────────────────────────────────────
    plot_path = os.path.join(eda_dir, "distilbert_sentiment_f1_per_aspect.png")
    plot_f1_per_aspect(all_results, plot_path)

    # ─── Summary ──────────────────────────────────────────────────────────
    gpu_mem = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0

    print(f"\n{'═' * 70}")
    print("PER-ASPECT SENTIMENT TRAINING — SUMMARY")
    print(f"{'═' * 70}")

    print(f"\n  {'Aspect':<28} {'N_train':>8} {'Status':<9} {'Val F1':>8} {'Time(s)':>8}")
    print(f"  {'─' * 65}")

    trained_count = 0
    total_f1 = 0.0

    for aspect in aspect_labels:
        r = all_results.get(aspect, {})
        n = r.get("n_train", 0)
        status = r.get("status", "?")
        f1 = r.get("val_macro_f1")
        t = r.get("training_time_s", 0)

        f1_str = f"{f1:.4f}" if f1 is not None else "  N/A"
        t_str = f"{t:.1f}" if t else "  —"
        print(f"  {aspect:<28} {n:>8} {status:<9} {f1_str:>8} {t_str:>8}")

        if f1 is not None:
            trained_count += 1
            total_f1 += f1

    avg_f1 = total_f1 / max(trained_count, 1)
    skipped = sum(1 for r in all_results.values() if r.get("status") == "skipped")

    print(f"  {'─' * 65}")
    print(f"\n  Models trained:    {trained_count}/15")
    print(f"  Models skipped:    {skipped}")
    print(f"  Average val F1:    {avg_f1:.4f}")
    print(f"  Total time:        {total_time:.1f}s")
    print(f"  Peak RAM:          {peak_memory / (1024**2):.2f} MB")
    print(f"  GPU memory:        {gpu_mem:.2f} MB")
    print(f"  Models saved:      {save_path}")
    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    main()
