"""Direct RRF output using shared fusion, without learned scorer invocations."""
from collections import defaultdict
from dataclasses import asdict
from time import perf_counter
from src.v2.cascade import observed_positive_metrics
from src.v2.recall_fusion import select_recall


def recommend_rrf(user_id, timestamp_ms, routes, budget, weights=None, positives=None):
    if budget.category_penalty or budget.max_per_author is not None:
        raise ValueError('Direct RRF requires no category or author reranking')
    tick = perf_counter()
    ranks = {}
    sources = defaultdict(set)
    rejected = defaultdict(int)
    for candidate in routes:
        if candidate.rank < 1:
            raise ValueError('Route ranks start at 1')
        if candidate.available_at_ms > timestamp_ms:
            rejected['not_yet_available'] += 1
            continue
        key = (candidate.item_id, candidate.source)
        ranks[key] = min(ranks.get(key, candidate.rank), candidate.rank)
        sources[candidate.item_id].add(candidate.source)
    recalled = select_recall(ranks, budget.recall, weights, {'method': 'rrf'})
    selected = recalled[:budget.output]
    stages = []
    for name, ids, before in [('recall', recalled, len(routes)), ('output', selected, len(recalled))]:
        stage = {'stage': name, 'input_count': before, 'output_count': len(ids), 'items': ids}
        if positives is not None:
            stage.update(observed_positive_metrics(ids, positives))
        stages.append(stage)
    return {'user_id': user_id, 'timestamp_ms': timestamp_ms, 'budget': asdict(budget),
            'items': selected, 'stages': stages, 'candidate_sources': {str(i): sorted(sources[i]) for i in recalled},
            'rejected': dict(rejected), 'shortfall': budget.output-len(selected),
            'total_elapsed_ms': (perf_counter()-tick)*1000,
            'timing_scope': 'fusion and output only; retrieval measured by caller',
            'label_scope': 'future logged-positive coverage; unexposed candidates are unlabeled'}
