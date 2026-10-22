"""Pure prior-day temporal features and actual bounded recall-route features."""
import numpy as np

TEMPORAL = ['log_item_exposures_1d', 'log_item_clicks_1d', 'item_ctr_1d',
            'log_item_exposures_3d', 'log_item_clicks_3d', 'item_ctr_3d',
            'click_share_1d_in_history', 'log_click_growth_1d_vs_prior2d']
ROUTE = ['recent_route_hit', 'itemcf_route_hit', 'recent_rrf60', 'itemcf_rrf60',
         'log_itemcf_score', 'combined_rrf60']
WIDTH = 20 + len(TEMPORAL) + len(ROUTE)


def temporal_values(items, n1, p1, n3, p3, cumulative_clicks, prior):
    items = np.asarray(items, dtype=np.int64)
    if not np.isfinite(prior) or not 0 <= prior <= 1:
        raise ValueError('Invalid prior')
    a, b, c, d, total = [np.asarray(x)[items] for x in (n1, p1, n3, p3, cumulative_clicks)]
    if np.any(b > a) or np.any(d > c) or np.any(b > d) or np.any(b > total):
        raise ValueError('Inconsistent historical counts')
    x = np.column_stack([np.log1p(a), np.log1p(b), (b + 20*prior)/(a + 20),
                         np.log1p(c), np.log1p(d), (d + 20*prior)/(c + 20),
                         b/(total + 1), np.log1p(b) - np.log1p((d-b)/2)]).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite temporal values')
    return x


def recent_order(clicks, k):
    ids = np.flatnonzero(np.asarray(clicks) > 0)
    return ids[np.lexsort((ids, -np.asarray(clicks)[ids]))[:k]].tolist()


def route_values(items, recent, cf_items, cf_scores):
    if len(cf_items) != len(cf_scores) or len(set(cf_items)) != len(cf_items) or len(set(recent)) != len(recent):
        raise ValueError('Invalid distinct route results')
    rr = {int(i): r for r, i in enumerate(recent, 1)}
    cr = {int(i): r for r, i in enumerate(cf_items, 1)}
    cs = dict(zip(cf_items, cf_scores))
    r = np.fromiter((rr.get(int(i), 0) for i in items), dtype=np.int32, count=len(items))
    c = np.fromiter((cr.get(int(i), 0) for i in items), dtype=np.int32, count=len(items))
    s = np.fromiter((cs.get(int(i), 0.) for i in items), dtype=np.float64, count=len(items))
    if np.any(s < 0) or not np.isfinite(s).all():
        raise ValueError('Invalid CF scores')
    rv = np.where(r > 0, 60./(60+r), 0.)
    cv = np.where(c > 0, 60./(60+c), 0.)
    return np.column_stack([r > 0, c > 0, rv, cv, np.log1p(s), rv+cv]).astype(np.float32)


def active_mask(temporal, route):
    mask = np.ones(WIDTH, dtype=np.float32)
    if not temporal:
        mask[20:20+len(TEMPORAL)] = 0
    if not route:
        mask[20+len(TEMPORAL):] = 0
    return mask


def standardized(x, mean, std, temporal, route):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != WIDTH or not np.isfinite(x).all():
        raise ValueError('Expected finite shared-width features')
    return ((x - mean)/std) * active_mask(temporal, route)
