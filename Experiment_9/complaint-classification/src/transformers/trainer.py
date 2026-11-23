"""
Configurable PyTorch training and evaluation loop for Hugging Face sequence classifiers.
Supports dynamic padding, AdamW + linear warmup schedule, gradient clipping,
optional class-weighted CrossEntropyLoss, mixed precision (when CUDA supports it),
early stopping, and best-checkpoint saving by validation Macro-F1.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup

from src.transformers.metrics import compute_classification_metrics


def predict_with_model(
    model: torch.nn.Module,
    dataset,
    tokenizer,
    batch_size: int = 32,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Run inference on a ComplaintTorchDataset.
    Returns (y_true, y_pred, probabilities, inference_time_sec).
    """
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    model.eval()
    model.to(device)
    all_preds: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels", None)
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)

            all_probs.append(probs)
            all_preds.append(preds)
            if labels is not None:
                all_labels.append(labels.cpu().numpy())

    elapsed = time.perf_counter() - t0
    y_pred = np.concatenate(all_preds, axis=0)
    probs_arr = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0) if all_labels else np.array([])
    return y_true, y_pred, probs_arr, float(elapsed)


def train_and_evaluate(
    model: torch.nn.Module,
    tokenizer,
    train_dataset,
    val_dataset,
    output_dir: Path,
    epochs: int = 3,
    batch_size: int = 16,
    eval_batch_size: int = 32,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    max_grad_norm: float = 1.0,
    class_weights: Optional[torch.Tensor] = None,
    early_stopping_patience: int = 2,
    use_amp: bool = False,
    device: str = "cpu",
    log_every_n_steps: int = 50,
) -> Dict[str, Any]:
    """
    Train a sequence classifier and save the best checkpoint by validation f1_macro.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator
    )

    model.to(device)
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    if class_weights is not None:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        loss_fn = nn.CrossEntropyLoss()

    amp_enabled = bool(use_amp and device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_f1 = -1.0
    best_epoch = 0
    best_val_metrics: Dict[str, float] = {}
    patience_counter = 0
    history: List[Dict[str, Any]] = []

    train_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        ep_start = time.perf_counter()
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels").to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(**inputs)
                loss = loss_fn(outputs.logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += float(loss.item())
            if log_every_n_steps > 0 and step % log_every_n_steps == 0:
                print(
                    f"  Epoch {epoch}/{epochs} | Step {step}/{len(train_loader)} | "
                    f"Loss: {running_loss / step:.4f}",
                    flush=True,
                )

        avg_train_loss = running_loss / max(1, len(train_loader))

        # Validation evaluation
        val_loader = DataLoader(
            val_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collator
        )
        model.eval()
        val_loss_sum = 0.0
        val_preds: List[np.ndarray] = []
        val_labels: List[np.ndarray] = []
        val_t0 = time.perf_counter()

        with torch.no_grad():
            for batch in val_loader:
                labels = batch.pop("labels").to(device)
                inputs = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**inputs)
                vloss = loss_fn(outputs.logits, labels)
                val_loss_sum += float(vloss.item())
                val_preds.append(torch.argmax(outputs.logits, dim=-1).cpu().numpy())
                val_labels.append(labels.cpu().numpy())

        val_infer_sec = time.perf_counter() - val_t0
        avg_val_loss = val_loss_sum / max(1, len(val_loader))
        y_val_true = np.concatenate(val_labels, axis=0)
        y_val_pred = np.concatenate(val_preds, axis=0)
        val_metrics = compute_classification_metrics(y_val_true, y_val_pred)

        ep_sec = time.perf_counter() - ep_start
        current_lr = float(scheduler.get_last_lr()[0])

        ep_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_f1_macro": round(val_metrics["f1_macro"], 4),
            "val_accuracy": round(val_metrics["accuracy"], 4),
            "val_f1_weighted": round(val_metrics["f1_weighted"], 4),
            "learning_rate": current_lr,
            "epoch_time_sec": round(ep_sec, 2),
            "val_infer_sec": round(val_infer_sec, 2),
        }
        history.append(ep_record)
        print(
            f"  [Epoch {epoch}/{epochs} Done in {ep_sec:.1f}s] "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Val Macro-F1: {val_metrics['f1_macro']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}",
            flush=True,
        )

        # Save best checkpoint by validation Macro-F1
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            best_val_metrics = {**val_metrics, "val_infer_sec": val_infer_sec}
            patience_counter = 0
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        else:
            patience_counter += 1
            if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
                print(f"  Early stopping triggered at epoch {epoch}.", flush=True)
                break

    total_train_sec = time.perf_counter() - train_start

    # Save training history inside checkpoint dir
    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return {
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "history": history,
        "total_train_time_sec": round(total_train_sec, 2),
    }
