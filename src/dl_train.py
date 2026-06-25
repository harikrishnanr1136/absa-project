"""
DistilBERT Training Loop for Aspect Detection (Multi-Label).

Trains AspectDetectionModel on HuggingFace datasets with:
- BCEWithLogitsLoss with pos_weight for class imbalance
- AdamW optimizer with linear warmup scheduler
- Early stopping based on val macro-F1
- Training curves visualization
"""

import json
import logging
import os
import time
import tracemalloc
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from sklearn.metrics import f1_score
from transformers import get_linear_schedule_with_warmup

from src.config import load_config, resolve_path
from src.dl_model import AspectDetectionModel, count_parameters

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_pos_weight(dataset, num_labels: int) -> torch.Tensor:
    """
    Compute pos_weight per aspect for BCEWithLogitsLoss.

    pos_weight[j] = neg_count_j / pos_count_j
    This upweights the loss for positive examples of rare aspects.
    """
    labels = np.array(dataset["labels"])
    pos_count = labels.sum(axis=0)
    neg_count = labels.shape[0] - pos_count

    # Avoid division by zero
    pos_count = np.maximum(pos_count, 1)
    pos_weight = neg_count / pos_count

    logger.info(f"Pos weights (min={pos_weight.min():.2f}, max={pos_weight.max():.2f}, "
                f"mean={pos_weight.mean():.2f})")
    return torch.tensor(pos_weight, dtype=torch.float32)


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    input_ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
    attention_mask = torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long)
    labels = torch.tensor([b["labels"] for b in batch], dtype=torch.float32)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def evaluate(model, dataloader, criterion, device) -> tuple:
    """
    Evaluate model on a dataloader.

    Returns:
        (avg_loss, micro_f1, macro_f1)
    """
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

            preds = (logits >= 0.5).int().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, micro_f1, macro_f1


def plot_training_curves(history: dict, save_path: str):
    """Plot loss and macro-F1 curves over epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-o", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # F1
    ax2.plot(epochs, history["val_micro_f1"], "g-o", label="Val Micro-F1")
    ax2.plot(epochs, history["val_macro_f1"], "m-o", label="Val Macro-F1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("Validation F1 Scores")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("DistilBERT Aspect Detection — Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info(f"Training curves saved: {save_path}")


def main():
    logger.info("=" * 70)
    logger.info("DISTILBERT ASPECT DETECTION — TRAINING")
    logger.info("=" * 70)

    # ─── Config ───────────────────────────────────────────────────────────
    config = load_config()
    dl_config = config["dl_training"]
    seed = config["seed"]
    num_labels = len(config["labels"]["aspects"])

    epochs = dl_config["epochs"]
    batch_size = dl_config["batch_size"]
    lr = dl_config["learning_rate"]
    weight_decay = dl_config["weight_decay"]
    warmup_ratio = dl_config["warmup_ratio"]
    patience = dl_config["early_stopping_patience"]
    model_name = dl_config["model_name"]

    # Paths
    data_dir = os.path.dirname(resolve_path(config["data"]["cleaned"]))
    hf_path = os.path.join(data_dir, "hf_datasets", "aspect_detection")
    output_dir = resolve_path(config["outputs"]["models"])
    eda_dir = resolve_path(config["outputs"]["eda"])
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(eda_dir, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ─── Load Datasets ────────────────────────────────────────────────────
    logger.info("")
    logger.info("─" * 70)
    logger.info("Loading HuggingFace datasets...")
    logger.info("─" * 70)

    ds = load_from_disk(hf_path)
    logger.info(f"Train: {len(ds['train'])}, Val: {len(ds['val'])}, Test: {len(ds['test'])}")

    # DataLoaders
    train_loader = DataLoader(ds["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(ds["val"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(ds["test"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # ─── Compute pos_weight ───────────────────────────────────────────────
    logger.info("Computing class imbalance weights...")
    pos_weight = compute_pos_weight(ds["train"], num_labels).to(device)

    # ─── Model, Optimizer, Scheduler ──────────────────────────────────────
    logger.info("")
    logger.info("─" * 70)
    logger.info("Initializing model, optimizer, scheduler...")
    logger.info("─" * 70)

    model = AspectDetectionModel(num_labels=num_labels, model_name=model_name).to(device)
    count_parameters(model, "AspectDetectionModel")

    # BCEWithLogitsLoss = Sigmoid + BCELoss (numerically stable)
    # We don't use sigmoid in model forward when using this loss
    # Adjust model to return raw logits for BCEWithLogitsLoss
    # Remove sigmoid from model for training (use it only during inference)
    model.sigmoid = nn.Identity()  # Replace sigmoid with identity for training

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    logger.info(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")

    # ─── Training Loop ────────────────────────────────────────────────────
    logger.info("")
    logger.info("─" * 70)
    logger.info("TRAINING STARTED")
    logger.info("─" * 70)

    history = {"train_loss": [], "val_loss": [], "val_micro_f1": [], "val_macro_f1": []}
    best_val_macro_f1 = 0.0
    best_model_state = None
    epochs_no_improve = 0

    tracemalloc.start()
    total_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # ── Train ──
        model.train()
        train_loss = 0
        for step, batch in enumerate(train_loader):
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

        # ── Evaluate ──
        # Temporarily add sigmoid for evaluation threshold
        model.sigmoid = nn.Sigmoid()
        val_loss, val_micro_f1, val_macro_f1 = evaluate(model, val_loader, criterion, device)
        model.sigmoid = nn.Identity()  # Remove for next training epoch

        epoch_time = time.time() - epoch_start

        # Record history
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_micro_f1"].append(val_micro_f1)
        history["val_macro_f1"].append(val_macro_f1)

        logger.info(f"Epoch {epoch}/{epochs} — "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Micro-F1: {val_micro_f1:.4f} | "
                    f"Val Macro-F1: {val_macro_f1:.4f} | "
                    f"Time: {epoch_time:.1f}s")

        # ── Best model & early stopping ──
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_model_state = deepcopy(model.state_dict())
            epochs_no_improve = 0
            logger.info(f"  ★ New best model (macro-F1: {best_val_macro_f1:.4f})")
        else:
            epochs_no_improve += 1
            logger.info(f"  No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= patience:
            logger.info(f"  ⛔ Early stopping triggered (patience={patience})")
            break

    total_time = time.time() - total_start
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # ─── Load Best Model & Evaluate on Test ───────────────────────────────
    logger.info("")
    logger.info("─" * 70)
    logger.info("FINAL EVALUATION (best checkpoint)")
    logger.info("─" * 70)

    model.load_state_dict(best_model_state)
    model.sigmoid = nn.Sigmoid()

    test_loss, test_micro_f1, test_macro_f1 = evaluate(model, test_loader, criterion, device)
    logger.info(f"Test — Loss: {test_loss:.4f} | Micro-F1: {test_micro_f1:.4f} | Macro-F1: {test_macro_f1:.4f}")

    # ─── Save Model ──────────────────────────────────────────────────────
    model_save_path = os.path.join(output_dir, "aspect_detector_distilbert.pt")
    torch.save(best_model_state, model_save_path)
    logger.info(f"Best model saved: {model_save_path}")

    # ─── Plot Training Curves ─────────────────────────────────────────────
    plot_path = os.path.join(eda_dir, "distilbert_aspect_training_curves.png")
    plot_training_curves(history, plot_path)

    # ─── Summary ──────────────────────────────────────────────────────────
    gpu_mem = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0

    print(f"\n{'═' * 70}")
    print("TRAINING COMPLETE — SUMMARY")
    print(f"{'═' * 70}")
    print(f"  Model:             AspectDetectionModel (DistilBERT)")
    print(f"  Epochs completed:  {len(history['train_loss'])}/{epochs}")
    print(f"  Best val macro-F1: {best_val_macro_f1:.4f}")
    print(f"  Test micro-F1:     {test_micro_f1:.4f}")
    print(f"  Test macro-F1:     {test_macro_f1:.4f}")
    print(f"  Total time:        {total_time:.1f}s")
    print(f"  Peak RAM:          {peak_memory / (1024**2):.2f} MB")
    print(f"  GPU memory:        {gpu_mem:.2f} MB")
    print(f"  Model saved:       {model_save_path}")
    print(f"  Curves saved:      {plot_path}")
    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    main()
