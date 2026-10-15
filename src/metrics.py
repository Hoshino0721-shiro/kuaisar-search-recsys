"""Small dependency-free ranking and probabilistic metrics."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Sequence


def _validate(labels, scores, probability=False):
    if not labels or len(labels) != len(scores):
        raise ValueError('Non-empty labels and matching scores are required')
    if any(y not in (0, 1) for y in labels):
        raise ValueError('Binary labels required; binarize click_cnt explicitly')
    if any(not math.isfinite(p) or (probability and not 0 <= p <= 1) for p in scores):
        raise ValueError('Invalid score/probability')


def _validate_ranking(ranked, k):
    if not isinstance(k, int) or k < 0:
        raise ValueError('k must be a non-negative integer')
    if len(set(ranked)) != len(ranked):
        raise ValueError('Candidate list must be deduplicated')


def logloss(labels: Sequence[float], probs: Sequence[float]) -> float:
    _validate(labels, probs, True)
    eps = 1e-15
    return -sum(y * math.log(max(eps, min(1 - eps, p))) + (1 - y) * math.log(max(eps, min(1 - eps, 1 - p))) for y, p in zip(labels, probs)) / max(1, len(labels))


def brier(labels: Sequence[float], probs: Sequence[float]) -> float:
    _validate(labels, probs, True)
    return sum((y - p) ** 2 for y, p in zip(labels, probs)) / max(1, len(labels))


def auc(labels: Sequence[float], scores: Sequence[float]) -> float | None:
    _validate(labels, scores)
    pairs = sorted(zip(scores, labels))
    positives = sum(y > 0 for _, y in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        avg_rank = (index + 1 + end) / 2
        rank_sum += avg_rank * sum(y > 0 for _, y in pairs[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def gauc(labels: Sequence[float], scores: Sequence[float], groups: Sequence[str]) -> float | None:
    _validate(labels, scores)
    if len(groups) != len(labels):
        raise ValueError('Group length mismatch')
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for y, score, group in zip(labels, scores, groups):
        grouped[str(group)].append((y, score))
    weighted = [(len(items), auc([y for y, _ in items], [score for _, score in items])) for items in grouped.values()]
    weighted = [(n, value) for n, value in weighted if value is not None]
    return sum(n * value for n, value in weighted) / sum(n for n, _ in weighted) if weighted else None


def recall_at_k(relevant: Iterable[str], ranked: Sequence[str], k: int) -> float:
    _validate_ranking(ranked, k)
    relevant_set = set(relevant)
    return len(relevant_set.intersection(ranked[:k])) / max(1, len(relevant_set))


def ndcg_at_k(relevant: Iterable[str], ranked: Sequence[str], k: int) -> float:
    _validate_ranking(ranked, k)
    relevant_set = set(relevant)
    dcg = sum(1 / math.log2(index + 2) for index, item in enumerate(ranked[:k]) if item in relevant_set)
    ideal = sum(1 / math.log2(index + 2) for index in range(min(k, len(relevant_set))))
    return dcg / ideal if ideal else 0.0


def mrr_at_k(relevant: Iterable[str], ranked: Sequence[str], k: int) -> float:
    _validate_ranking(ranked, k)
    relevant_set = set(relevant)
    for index, item in enumerate(ranked[:k], start=1):
        if item in relevant_set:
            return 1.0 / index
    return 0.0
