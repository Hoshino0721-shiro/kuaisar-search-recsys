"""Deterministic recall allocation shared by evaluation and the cascade."""
from collections import defaultdict
from math import isfinite


def route_limits(config):
    default = config.get('route_k', 1000)
    if isinstance(default, bool) or not isinstance(default, int) or default < 1:
        raise ValueError('route_k must be a positive integer')
    overrides = config.get('route_limits', {})
    if not isinstance(overrides, dict) or set(overrides) - {'popular', 'recent', 'itemcf', 'tower'}:
        raise ValueError('Unknown route limit')
    limits = {name: overrides.get(name, default) for name in ('popular', 'recent', 'itemcf')}
    if 'tower' in overrides or config.get('tower_index'):
        limits['tower'] = overrides.get('tower',0)
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in limits.values()):
        raise ValueError('Route limits must be nonnegative integers')
    return limits


def validate_policy(policy, limit):
    policy = {} if policy is None else policy
    if not isinstance(policy, dict) or policy.get('method', 'rrf') not in ('rrf', 'quota'):
        raise ValueError('Unknown recall fusion method')
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError('Recall budget must be a positive integer')
    if policy.get('method', 'rrf') == 'rrf':
        if set(policy) - {'method'}:
            raise ValueError('Unexpected RRF policy fields')
        return
    if set(policy) - {'method', 'allocations', 'fill_order'}:
        raise ValueError('Unexpected quota policy fields')
    allocations = policy.get('allocations', [])
    fills = policy.get('fill_order', [])
    if not isinstance(allocations, list) or not isinstance(fills, list):
        raise ValueError('Quota allocations and fill_order must be lists')
    names, total = [], 0
    for entry in allocations:
        if (not isinstance(entry, list) or len(entry) != 2 or not isinstance(entry[0], str)
                or not entry[0] or isinstance(entry[1], bool)
                or not isinstance(entry[1], int) or entry[1] < 0):
            raise ValueError('Expected [source, nonnegative quota]')
        names.append(entry[0])
        total += entry[1]
    if len(set(names)) != len(names) or total > limit:
        raise ValueError('Duplicate source or quotas exceed recall budget')
    if (any(not isinstance(n, str) or not n for n in fills)
            or len(set(fills)) != len(fills)):
        raise ValueError('Invalid quota fill_order')


def select_recall(ranks, limit, weights=None, policy=None):
    """Select unique IDs from {(item, source): best one-based route rank}.

    Quotas count *new unique* candidates in allocation order. Empty/overlapping
    routes yield their unused capacity to fill_order, never to duplicated IDs.
    RRF remains the default, including its original constant 60 and ID ties.
    """
    validate_policy(policy, limit)
    if any(isinstance(r, bool) or not isinstance(r, int) or r < 1 for r in ranks.values()):
        raise ValueError('Route ranks must be positive integers')
    policy = policy or {}
    weights = {} if weights is None else weights
    if any(not isfinite(float(w)) or float(w) < 0 for w in weights.values()):
        raise ValueError('Route weights must be finite and nonnegative')
    if policy.get('method', 'rrf') == 'rrf':
        scores = defaultdict(float)
        for (iid, source), rank in ranks.items():
            weight = float(weights.get(source, 1.))
            if weight:
                scores[iid] += weight / (60 + rank)
        return sorted(scores, key=lambda i: (-scores[i], i))[:limit]
    lists = defaultdict(list)
    for (iid, source), rank in ranks.items():
        lists[source].append((rank, iid))
    lists = {name: [iid for _, iid in sorted(rows)] for name, rows in lists.items()}
    seen, selected = set(), []

    def take(source, count):
        added = 0
        for iid in lists.get(source, []):
            if added >= count or len(selected) >= limit:
                break
            if iid not in seen:
                selected.append(iid)
                seen.add(iid)
                added += 1

    for source, quota in policy.get('allocations', []):
        take(source, quota)
    for source in policy.get('fill_order', []):
        take(source, limit - len(selected))
    return selected
