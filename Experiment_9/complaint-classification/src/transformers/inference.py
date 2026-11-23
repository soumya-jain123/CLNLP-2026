"""
Standalone inference utility for loading a saved Hugging Face transformer
complaint classifier and generating predictions in a separate process.
"""

import json
from pathlib import Path
from typing import Dict, List, Sequence, Union

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class TransformerComplaintClassifier:
    """Wrapper for loading a saved transformer model and running batch inference."""

    def __init__(self, model_dir: Union[str, Path], device: str = None):
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        self.model.to(self.device)
        self.model.eval()

        meta_path = self.model_dir / "experiment_metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            self.max_length = int(self.metadata.get("max_length", 256))
        else:
            self.metadata = {}
            self.max_length = 256

    def predict(
        self, texts: Sequence[str], batch_size: int = 32
    ) -> List[Dict[str, object]]:
        """Predict class label and probability distribution for a list of complaint texts."""
        if isinstance(texts, str):
            texts = [texts]

        results: List[Dict[str, object]] = []
        id2label = self.model.config.id2label

        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = [str(t) if t is not None else "" for t in texts[start : start + batch_size]]
                enc = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                logits = self.model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                preds = np.argmax(probs, axis=-1)

                for p_idx, prob_vec in zip(preds, probs):
                    lbl = id2label[int(p_idx)] if int(p_idx) in id2label else id2label[str(int(p_idx))]
                    results.append(
                        {
                            "predicted_label": lbl,
                            "predicted_id": int(p_idx),
                            "confidence": round(float(prob_vec[p_idx]), 4),
                        }
                    )
        return results
