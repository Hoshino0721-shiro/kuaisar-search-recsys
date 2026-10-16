"""Four-stage orchestration and observed-positive diagnostics.

This module has no model framework dependency. Providers own retrieval and
feature availability; scorers receive explicit, ordered candidate ID lists.
"""
from dataclasses import asdict, dataclass
from collections import defaultdict
from math import isfinite, log1p, log2
from time import perf_counter
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True)
class Candidate:
    item_id: int
    source: str
    rank: int
    available_at_ms: int


@dataclass(frozen=True)
class Budget:
    recall: int = 2000
    coarse: int = 400
    fine: int = 80
    output: int = 20
    category_penalty: float = 0.0
    max_per_author: int | None = None
    coarse_enabled: bool = True

    def __post_init__(self):
        sizes = (self.recall, self.coarse, self.fine, self.output)
        if any(isinstance(n, bool) or not isinstance(n, int) for n in sizes):
            raise ValueError('Candidate budgets must be integers')
        if not self.recall >= self.coarse >= self.fine >= self.output > 0:
            raise ValueError('Require recall >= coarse >= fine >= output > 0')
        if not isfinite(self.category_penalty) or self.category_penalty < 0:
            raise ValueError('Category penalty must be finite and nonnegative')
        if self.max_per_author is not None and self.max_per_author <= 0:
            raise ValueError('Author limit must be positive')
        if not isinstance(self.coarse_enabled, bool):
            raise ValueError('coarse_enabled must be boolean')


def ordered_scores(items: Sequence[int], scores: Sequence[float]):
    if len(items) != len(scores):
        raise ValueError('Scorer returned wrong number of scores')
    pairs = [(int(i), float(s)) for i, s in zip(items, scores)]
    if not all(isfinite(s) for _, s in pairs):
        raise ValueError('Scorer returned nonfinite scores')
    return sorted(pairs, key=lambda row: (-row[1], row[0]))


def observed_positive_metrics(items, positives: Mapping[int, int]):
    """Logged-positive coverage, never inferred negative labels or online CTR."""
    if any(n < 0 for n in positives.values()):
        raise ValueError('Positive event counts must be nonnegative')
    chosen = set(items)
    total = sum(positives.values())
    hits = sum(n for i, n in positives.items() if i in chosen)
    relevant = {i for i, n in positives.items() if n > 0}
    ideal = sum(1 / log2(j + 2) for j in range(min(20, len(relevant))))
    dcg = sum(1 / log2(j + 2) for j, i in enumerate(items[:20]) if i in relevant)
    return {'positive_events': total, 'hit_events': hits,
            'observed_event_recall': hits / total if total else None,
            'user_period_ndcg_at_20': dcg / ideal if ideal else None}


class Cascade:
    def __init__(self, coarse: Callable, fine: Callable, budget=Budget(),
                 categories=None, authors=None):
        self.coarse = coarse
        self.fine = fine
        self.budget = budget
        self.categories = categories or {}
        self.authors = authors or {}

    def run(self, user_id: int, timestamp_ms: int, routes: Sequence[Candidate],
            *, exclude_items=(), route_weights=None, positives=None):
        start = perf_counter()
        weights = route_weights or {}
        excluded = set(exclude_items)
        ranks = {}
        rejected = defaultdict(int)
        for c in routes:
            if c.rank < 1:
                raise ValueError('Route ranks start at 1')
            if c.available_at_ms > timestamp_ms:
                rejected['not_yet_available'] += 1
                continue
            if c.item_id in excluded:
                rejected['excluded_item'] += 1
                continue
            # One item contributes once per route, even if a provider duplicates it.
            key = (c.item_id, c.source)
            ranks[key] = min(ranks.get(key, c.rank), c.rank)
        fused = defaultdict(float)
        sources = defaultdict(set)
        for (iid, source), rank in ranks.items():
            weight = float(weights.get(source, 1.0))
            if not isfinite(weight) or weight < 0:
                raise ValueError('Route weights must be finite and nonnegative')
            if weight:
                fused[iid] += weight / (60 + rank)
                sources[iid].add(source)
        recalled = [i for i, _ in ordered_scores(list(fused), list(fused.values()))[:self.budget.recall]]
        recall_ms = (perf_counter() - start) * 1000
        steps = []

        def record(name, ids, elapsed, before):
            entry = {'stage': name, 'input_count': before, 'output_count': len(ids),
                     'elapsed_ms': elapsed, 'items': list(ids)}
            if positives is not None:
                entry.update(observed_positive_metrics(ids, positives))
                previous = steps[-1]['hit_events'] if steps else None
                entry['positive_retention_vs_previous'] = (
                    entry['hit_events'] / previous if previous else None)
            steps.append(entry)

        record('recall', recalled, recall_ms, len(routes))
        ids = recalled
        for name, scorer, limit in [('coarse', self.coarse, self.budget.coarse),
                                    ('fine', self.fine, self.budget.fine)]:
            tick = perf_counter()
            if name == 'coarse' and not self.budget.coarse_enabled:
                record(name, ids, (perf_counter() - tick) * 1000, len(ids))
                continue
            values = scorer(user_id, ids, timestamp_ms) if ids else []
            ranked = ordered_scores(ids, values)
            kept = [i for i, _ in ranked[:limit]]
            record(name, kept, (perf_counter() - tick) * 1000, len(ids))
            ids = kept

        # Fine scores are the utility for reranking. Do not call either model again.
        remaining = dict(ranked[:self.budget.fine])
        counts = defaultdict(int)
        author_counts = defaultdict(int)
        selected = []
        tick = perf_counter()
        if self.budget.category_penalty == 0 and self.budget.max_per_author is None:
            selected = list(remaining)[:self.budget.output]
            remaining.clear()
        while remaining and len(selected) < self.budget.output:
            eligible = []
            for iid, score in remaining.items():
                author = self.authors.get(iid)
                if (self.budget.max_per_author is not None and author is not None
                        and author_counts[author] >= self.budget.max_per_author):
                    continue
                category = self.categories.get(iid)
                adjustment = (self.budget.category_penalty * log1p(counts[category])
                              if category not in (None, 0, '', 'UNKNOWN') else 0)
                eligible.append((iid, score - adjustment))
            if not eligible:
                break
            iid, _ = min(eligible, key=lambda row: (-row[1], row[0]))
            selected.append(iid)
            remaining.pop(iid)
            category = self.categories.get(iid)
            counts[category] += 1
            if self.authors.get(iid) is not None:
                author_counts[self.authors[iid]] += 1
        record('rerank', selected, (perf_counter() - tick) * 1000, len(ids))
        return {'user_id': user_id, 'timestamp_ms': timestamp_ms,
                'budget': asdict(self.budget), 'items': selected,
                'candidate_sources': {str(i): sorted(sources[i]) for i in recalled},
                'rejected': dict(rejected), 'shortfall': self.budget.output - len(selected),
                'stages': steps, 'total_elapsed_ms': (perf_counter() - start) * 1000,
                'timing_scope': 'cascade only; route retrieval is measured by its caller',
                'label_scope': 'observed positives only; unexposed candidates are unlabeled'}
