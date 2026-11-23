"""
Token-level interpretability for transformer sequence classifiers using
Gradient x Input embedding attribution (with optional attention extraction).

Limitations:
- Gradient x Input measures local first-order sensitivity of the target logit to
  subword embeddings; it does not prove global causal necessity.
- Attention weights alone are not treated as definitive explanations.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch


def compute_token_attributions(
    model: torch.nn.Module,
    tokenizer,
    text: str,
    target_class_idx: int,
    max_length: int = 256,
    device: str = "cpu",
    top_k: int = 12,
) -> Dict[str, object]:
    """
    Compute Gradient x Input attribution scores for each subword token.
    Returns tokens, normalized importance scores, and top-k contributing tokens.
    """
    model.eval()
    model.to(device)

    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    embedding_layer = model.get_input_embeddings()
    inputs_embeds = embedding_layer(input_ids).detach().clone()
    inputs_embeds.requires_grad_(True)

    outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))

    model.zero_grad()
    target_logit = logits[0, target_class_idx]
    target_logit.backward()

    grads = inputs_embeds.grad.detach()[0]  # (seq_len, hidden_dim)
    embeds = inputs_embeds.detach()[0]
    # L1 norm of gradient * input embedding per token
    raw_scores = torch.sum(torch.abs(grads * embeds), dim=-1).cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
    special_tokens = set(tokenizer.all_special_tokens)

    filtered: List[Tuple[str, float]] = []
    for tok, sc in zip(tokens, raw_scores):
        if tok in special_tokens:
            continue
        filtered.append((tok, float(sc)))

    total_sc = sum(s for _, s in filtered) + 1e-12
    norm_tokens = [(tok, round(sc / total_sc, 4)) for tok, sc in filtered]
    top_tokens = sorted(norm_tokens, key=lambda x: x[1], reverse=True)[:top_k]

    return {
        "predicted_idx": pred_idx,
        "predicted_prob": round(float(probs[pred_idx]), 4),
        "target_class_idx": int(target_class_idx),
        "target_prob": round(float(probs[target_class_idx]), 4),
        "num_tokens": len(tokens),
        "token_scores": norm_tokens,
        "top_tokens": top_tokens,
    }
