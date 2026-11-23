"""
Confidence Calibration for Phase 4 LinearSVC Pipeline.

Fits CalibratedClassifierCV (Platt scaling / sigmoid, cv=5) strictly on
training data (never on test data), evaluates multiclass Brier score &
Expected Calibration Error (ECE) on a held-out 20% validation split,
saves a reliability diagram, and exports models/calibrated_linear_svc.joblib.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score
from sklearn.model_selection import train_test_split

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def compute_ece(y_true_int: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Compute top-label Expected Calibration Error (ECE)."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == y_true_int).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true_int)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi) if lo > 0 else (confidences >= lo) & (confidences <= hi)
        if np.any(mask):
            ece += (np.sum(mask) / n) * abs(float(np.mean(accuracies[mask])) - float(np.mean(confidences[mask])))
    return float(ece)


def fit_and_evaluate_calibration(project_root: Path = _PROJECT_ROOT) -> Dict[str, Any]:
    """
    Validate probability calibration on 80/20 stratified train/val split,
    save reliability diagram, then fit full-train calibrated pipeline to disk.
    """
    base_model_path = project_root / "models" / "best_traditional_ml_model.joblib"
    train_csv = project_root / "data" / "processed" / "train_processed.csv"
    out_model_path = project_root / "models" / "calibrated_linear_svc.joblib"
    fig_path = project_root / "reports" / "figures" / "prioritization" / "calibration_reliability_diagram.png"
    metrics_path = project_root / "reports" / "prioritization" / "calibration_metrics.json"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    base_pipe = joblib.load(base_model_path)
    df_train = pd.read_csv(train_csv)

    X = df_train["text_stopword"].astype(str).values
    y = df_train["label"].values
    classes = sorted(df_train["label"].unique().tolist())
    label2id = {c: i for i, c in enumerate(classes)}

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    val_calibrator = CalibratedClassifierCV(
        estimator=clone(base_pipe), method="sigmoid", cv=5
    )
    val_calibrator.fit(X_tr, y_tr)
    val_probs = val_calibrator.predict_proba(X_val)
    val_preds = val_calibrator.classes_[val_probs.argmax(axis=1)]

    y_val_int = np.array([label2id[lbl] for lbl in y_val])
    # Multiclass Brier score (mean squared error over one-hot probabilities)
    y_val_onehot = np.eye(len(classes))[y_val_int]
    multiclass_brier = float(np.mean(np.sum((val_probs - y_val_onehot) ** 2, axis=1)))
    ece = compute_ece(y_val_int, val_probs, n_bins=10)

    # Reliability diagram (top-label confidence vs accuracy)
    top_conf = val_probs.max(axis=1)
    top_correct = (val_preds == y_val).astype(int)
    prob_true, prob_pred = calibration_curve(top_correct, top_conf, n_bins=10, strategy="uniform")

    plt.figure(figsize=(6.8, 5.2))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.plot(prob_pred, prob_true, "o-", color="#4C72B0", label=f"Calibrated LinearSVC (ECE={ece:.4f})")
    plt.title("LinearSVC Confidence Calibration Reliability Diagram (Val Split)")
    plt.xlabel("Mean Predicted Confidence")
    plt.ylabel("Fraction of Correct Predictions")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Fit production calibrator on full cleaned training data (never test!)
    prod_calibrator = CalibratedClassifierCV(
        estimator=clone(base_pipe), method="sigmoid", cv=5
    )
    prod_calibrator.fit(X, y)
    joblib.dump(prod_calibrator, out_model_path)

    metrics = {
        "calibration_method": "CalibratedClassifierCV(method='sigmoid', cv=5)",
        "validation_samples": int(len(X_val)),
        "val_accuracy": round(float(accuracy_score(y_val, val_preds)), 4),
        "val_f1_macro": round(float(f1_score(y_val, val_preds, average="macro")), 4),
        "multiclass_brier_score": round(multiclass_brier, 4),
        "expected_calibration_error_ece": round(ece, 4),
        "calibrated_model_path": str(out_model_path.relative_to(project_root)).replace("\\", "/"),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    res = fit_and_evaluate_calibration()
    print("Calibration complete:", res)
