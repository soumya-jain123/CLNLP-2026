"""
CLI Experiment Runner for Phase 5 Transformer Classification.

Usage:
    python -m src.transformers.experiment_runner --config configs/transformer_experiments.yaml
    python -m src.transformers.experiment_runner --config configs/transformer_experiments.yaml --experiments T01 T02 --resume
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.transformers.dataset import (
    ComplaintTorchDataset,
    EXPECTED_LABELS,
    ID2LABEL,
    LABEL2ID,
    create_stratified_split,
    encode_labels,
    load_processed_datasets,
)
from src.transformers.model_factory import (
    compute_balanced_class_weights,
    count_parameters,
    create_sequence_classifier,
)
from src.transformers.tokenizer_utils import load_tokenizer
from src.transformers.trainer import train_and_evaluate
from src.utils.hardware_detection import detect_hardware
from src.utils.reproducibility import set_global_seed


def _dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total_bytes / (1024**2), 2)


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_experiments(
    config_path: Path,
    selected_ids: Optional[List[str]] = None,
    resume: bool = False,
    pilot: bool = False,
    project_root: Path = _PROJECT_ROOT,
) -> pd.DataFrame:
    """
    Execute configured transformer experiments sequentially on train/validation splits.
    Never touches the held-out test set during model selection or tuning.
    """
    cfg = load_config(config_path)
    seed = int(cfg.get("seed", 42))
    val_size = float(cfg.get("val_size", 0.2))
    hw = detect_hardware(project_root)
    device = hw["device"]

    processed_dir = project_root / cfg.get("processed_dir", "data/processed")
    reports_dir = project_root / cfg.get("reports_dir", "reports/transformers")
    models_base_dir = project_root / cfg.get("models_dir", "models/transformers")
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_base_dir.mkdir(parents=True, exist_ok=True)

    results_csv = reports_dir / "experiment_results.csv"
    history_csv = reports_dir / "training_history.csv"
    comparison_csv = reports_dir / "model_comparison.csv"

    # Load existing results if resuming
    existing_results: List[Dict[str, Any]] = []
    completed_ids = set()
    if resume and results_csv.exists():
        prev_df = pd.read_csv(results_csv)
        existing_results = prev_df.to_dict(orient="records")
        completed_ids = set(
            prev_df[prev_df["status"] == "completed"]["experiment_id"].astype(str).tolist()
        )
        print(f"Resuming: skipping already completed experiments {sorted(completed_ids)}")

    existing_history: List[Dict[str, Any]] = []
    if resume and history_csv.exists():
        existing_history = pd.read_csv(history_csv).to_dict(orient="records")

    # Load and split training data (test set isolated)
    df_train, _, _ = load_processed_datasets(processed_dir, text_col="text_minimal")
    train_sub, val_sub = create_stratified_split(
        df_train, val_size=val_size, random_state=seed
    )

    if pilot:
        train_sub = train_sub.groupby("label", group_keys=False).head(12).reset_index(drop=True)
        val_sub = val_sub.groupby("label", group_keys=False).head(5).reset_index(drop=True)
        print(f"[PILOT MODE] Using {len(train_sub)} train / {len(val_sub)} val samples.")

    y_train_ids = encode_labels(train_sub["label"].tolist(), LABEL2ID)
    y_val_ids = encode_labels(val_sub["label"].tolist(), LABEL2ID)

    experiments = cfg.get("experiments", [])
    if selected_ids:
        experiments = [e for e in experiments if e["id"] in selected_ids]

    for exp in experiments:
        exp_id = str(exp["id"])
        if resume and exp_id in completed_ids:
            continue

        model_name = exp["model_name"]
        checkpoint = exp["checkpoint"]
        text_col = exp.get("input_col", "text_minimal")
        max_length = int(exp.get("max_length", 128))
        lr = float(exp.get("learning_rate", 2e-5))
        batch_size = int(exp.get("batch_size", hw["recommended_batch_size"]))
        epochs = 1 if pilot else int(exp.get("epochs", 3))
        weight_decay = float(exp.get("weight_decay", 0.01))
        warmup_ratio = float(exp.get("warmup_ratio", 0.1))
        loss_type = str(exp.get("loss", "standard"))
        patience = int(exp.get("early_stopping_patience", 2))
        use_amp = bool(exp.get("use_amp", hw["fp16_supported"]))

        exp_out_dir = models_base_dir / exp_id
        print(
            f"\n=== Running {exp_id}: {model_name} ({checkpoint}) | "
            f"L={max_length} | Loss={loss_type} | LR={lr} | Epochs={epochs} | Device={device} ===",
            flush=True,
        )

        set_global_seed(seed)
        try:
            tokenizer = load_tokenizer(checkpoint)
            train_ds = ComplaintTorchDataset(
                train_sub[text_col].tolist(), y_train_ids, tokenizer, max_length=max_length
            )
            val_ds = ComplaintTorchDataset(
                val_sub[text_col].tolist(), y_val_ids, tokenizer, max_length=max_length
            )

            model = create_sequence_classifier(
                checkpoint, num_labels=len(EXPECTED_LABELS), label2id=LABEL2ID, id2label=ID2LABEL
            )
            total_params, trainable_params = count_parameters(model)

            cls_weights = (
                compute_balanced_class_weights(y_train_ids, num_labels=len(EXPECTED_LABELS))
                if loss_type == "weighted"
                else None
            )

            train_res = train_and_evaluate(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_ds,
                val_dataset=val_ds,
                output_dir=exp_out_dir,
                epochs=epochs,
                batch_size=batch_size,
                eval_batch_size=max(32, batch_size * 2),
                learning_rate=lr,
                weight_decay=weight_decay,
                warmup_ratio=warmup_ratio,
                class_weights=cls_weights,
                early_stopping_patience=patience,
                use_amp=use_amp,
                device=device,
            )

            best_val = train_res["best_val_metrics"]
            model_size_mb = _dir_size_mb(exp_out_dir)

            record = {
                "experiment_id": exp_id,
                "model_name": model_name,
                "checkpoint": checkpoint,
                "tokenizer": checkpoint,
                "input_col": text_col,
                "max_length": max_length,
                "learning_rate": lr,
                "batch_size": batch_size,
                "epochs_configured": epochs,
                "best_epoch": train_res["best_epoch"],
                "weight_decay": weight_decay,
                "loss": loss_type,
                "seed": seed,
                "val_f1_macro": round(float(best_val["f1_macro"]), 4),
                "val_accuracy": round(float(best_val["accuracy"]), 4),
                "val_f1_weighted": round(float(best_val["f1_weighted"]), 4),
                "val_precision_macro": round(float(best_val["precision_macro"]), 4),
                "val_recall_macro": round(float(best_val["recall_macro"]), 4),
                "training_time_sec": train_res["total_train_time_sec"],
                "val_inference_time_sec": round(float(best_val["val_infer_sec"]), 2),
                "total_params": total_params,
                "trainable_params": trainable_params,
                "model_size_mb": model_size_mb,
                "device_info": f"{device} ({hw['cpu_cores_logical']} cores)",
                "status": "completed",
                "error_details": "",
            }

            with open(exp_out_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)

            for ep_row in train_res["history"]:
                existing_history.append(
                    {"experiment_id": exp_id, "model_name": model_name, **ep_row}
                )

            existing_results = [r for r in existing_results if r["experiment_id"] != exp_id]
            existing_results.append(record)

        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            print(f"[FAILED] {exp_id}: {err_msg}", flush=True)
            traceback.print_exc()
            existing_results = [r for r in existing_results if r["experiment_id"] != exp_id]
            existing_results.append(
                {
                    "experiment_id": exp_id,
                    "model_name": model_name,
                    "checkpoint": checkpoint,
                    "tokenizer": checkpoint,
                    "input_col": text_col,
                    "max_length": max_length,
                    "learning_rate": lr,
                    "batch_size": batch_size,
                    "epochs_configured": epochs,
                    "best_epoch": 0,
                    "weight_decay": weight_decay,
                    "loss": loss_type,
                    "seed": seed,
                    "val_f1_macro": None,
                    "val_accuracy": None,
                    "val_f1_weighted": None,
                    "val_precision_macro": None,
                    "val_recall_macro": None,
                    "training_time_sec": 0.0,
                    "val_inference_time_sec": 0.0,
                    "total_params": 0,
                    "trainable_params": 0,
                    "model_size_mb": 0.0,
                    "device_info": device,
                    "status": "failed",
                    "error_details": err_msg,
                }
            )

        # Persist incrementally after each experiment for crash recovery
        res_df = pd.DataFrame(existing_results)
        res_df.to_csv(results_csv, index=False)
        if existing_history:
            pd.DataFrame(existing_history).to_csv(history_csv, index=False)

    final_df = pd.DataFrame(existing_results)
    if not final_df.empty and "val_f1_macro" in final_df.columns:
        comp_df = final_df[final_df["status"] == "completed"].sort_values(
            "val_f1_macro", ascending=False
        )
        comp_df.to_csv(comparison_csv, index=False)
    return final_df


def main():
    parser = argparse.ArgumentParser(description="Run Phase 5 Transformer Experiments.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/transformer_experiments.yaml",
        help="Path to YAML experiment config.",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Optional list of experiment IDs to run (e.g. T01 T02).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already completed experiments in experiment_results.csv.",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run fast 1-epoch pilot on small stratified subset.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = _PROJECT_ROOT / cfg_path

    df = run_experiments(
        config_path=cfg_path,
        selected_ids=args.experiments,
        resume=args.resume,
        pilot=args.pilot,
        project_root=_PROJECT_ROOT,
    )
    print("\n=== Experiment Runner Summary ===")
    print(df[["experiment_id", "model_name", "max_length", "loss", "val_f1_macro", "val_accuracy", "training_time_sec", "status"]])


if __name__ == "__main__":
    main()
