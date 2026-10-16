"""Cheap feature path matching the selected eight v1 cache columns exactly."""
import numpy as np


def coarse_features(state, users, items):
    users = np.asarray(users, dtype=np.int64)
    items = np.asarray(items, dtype=np.int64)
    if users.shape != items.shape or users.ndim != 1:
        raise ValueError('Expected aligned one-dimensional user/item arrays')
    if (np.any(users < 0) or np.any(users >= len(state.un))
            or np.any(items < 0) or np.any(items >= len(state.categories))):
        raise ValueError('Feature ID outside snapshot bounds')
    category = state.categories[items]
    user_category = users * state.nc + category
    prior = float(state.up.sum() / max(1, state.un.sum())) if state.un.sum() else .5
    def smooth(positive, count, strength):
        return (positive + strength * prior) / (count + strength)
    values = [smooth(state.up[users], state.un[users], 20),
              smooth(state.ip[items], state.inum[items], 20),
              smooth(state.cp[category], state.cn[category], 100),
              smooth(state.ucp[user_category], state.ucn[user_category], 20),
              np.log1p(state.un[users]), np.log1p(state.inum[items]),
              (state.ucp[user_category] + 1) / (state.up[users] + state.nc),
              state.inum[items] > 0]
    result = np.column_stack(values).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite coarse features')
    return result
