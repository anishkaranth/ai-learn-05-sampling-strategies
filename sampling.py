"""Temperature, top-k, and top-p (nucleus) sampling from scratch."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    t = max(float(temperature), 1e-8)
    x = logits.astype(np.float64) / t
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def sample_temperature(logits: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    p = softmax(logits, temperature)
    return int(rng.choice(len(p), p=p))


def sample_top_k(logits: np.ndarray, k: int, temperature: float, rng: np.random.Generator) -> int:
    k = max(1, min(k, len(logits)))
    top_idx = np.argpartition(logits, -k)[-k:]
    masked = np.full_like(logits, -np.inf, dtype=np.float64)
    masked[top_idx] = logits[top_idx]
    p = softmax(masked, temperature)
    return int(rng.choice(len(p), p=p))


def sample_top_p(logits: np.ndarray, p: float, temperature: float, rng: np.random.Generator) -> int:
    """Nucleus sampling: keep smallest set of tokens with cumulative prob >= p."""
    probs = softmax(logits, temperature)
    order = np.argsort(-probs)
    sorted_p = probs[order]
    cum = np.cumsum(sorted_p)
    cutoff = int(np.searchsorted(cum, p, side="left"))
    keep = order[: cutoff + 1]
    masked = np.zeros_like(probs)
    masked[keep] = probs[keep]
    masked /= masked.sum()
    return int(rng.choice(len(masked), p=masked))


def entropy(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    return float(-(p * np.log(p)).sum())


def diversity_metrics(samples: np.ndarray, vocab_size: int) -> Dict[str, float]:
    """samples: (N,) token ids"""
    counts = np.bincount(samples, minlength=vocab_size).astype(np.float64)
    probs = counts / max(counts.sum(), 1.0)
    unique = float(np.count_nonzero(counts))
    return {
        "unique_tokens": unique,
        "unique_ratio": unique / vocab_size,
        "entropy": entropy(probs),
        "top1_share": float(counts.max() / max(counts.sum(), 1.0)),
    }
