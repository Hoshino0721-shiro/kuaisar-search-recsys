from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from src.itemcf_index import IndexConfig, ItemCFIndex, build, make_snapshot, graph_matrices, export_prefix


def fixture(config, root):
    # User 1 and 2 share item10; user3 has many items and should be downweighted.
    events = [(1, 10, 90), (1, 20, 80), (1, 20, 70),
              (2, 10, 90), (2, 30, 80),
              (3, 10, 90), (3, 20, 80), (3, 30, 70), (3, 40, 60),
              (4, 40, 100)]  # cutoff event must never enter the index
    s = make_snapshot(events, [10, 20, 30, 40], 100, config, 'fixture-manifest')
    with redirect_stdout(io.StringIO()):
        build(s, config, root)
    return s


class ItemCFIndexTests(unittest.TestCase):
    def test_reload_matches_dense_cosine_reference_and_repeated_events_are_binary(self):
        cfg = IndexConfig(neighbors=4, max_user_items=5, seed_items=5,
                          user_iuf=False, query_half_life_days=None, block_items=2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'index'
            s = fixture(cfg, root)
            b = np.zeros((len(s.users), len(s.items)))
            b[s.rows, s.columns] = 1
            norm = b / np.sqrt(b.sum(axis=0))
            sim = norm.T @ norm
            np.fill_diagonal(sim, 0)
            index = ItemCFIndex(root, expected_manifest_sha256='fixture-manifest')
            try:
                result = index.query(1, 4, explain=True)
                expected = sim[[0, 1]].sum(axis=0)
                actual = dict(zip(result['items'], result['scores']))
                for j, iid in enumerate(s.items):
                    self.assertAlmostEqual(actual.get(int(iid), 0), expected[j], places=6)
                self.assertEqual(result['seed_count'], 2)
                self.assertEqual(result['items'], [30, 40, 10, 20])
                self.assertIn('top_seed_contributions', result)
                self.assertEqual(index.query(1, 4, exclude_history=True)['items'], [30, 40])
                self.assertEqual(index.query(1, 4, exclude_items={30})['items'], [40, 10, 20])
                self.assertFalse(index.has_user(4))
                self.assertEqual(index.query(4)['reason'], 'no_usable_history')
                for row in range(len(s.items)):
                    self.assertNotIn(row, index.neighbors[row])
            finally:
                index.close()

    def test_iuf_and_support_shrinkage_match_independent_dense_formula(self):
        cfg = IndexConfig(neighbors=4, max_user_items=5, seed_items=5,
                          user_iuf=True, shrinkage=2, min_cooccurrence=2, block_items=2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'index'
            s = fixture(cfg, root)
            b = np.zeros((len(s.users), len(s.items)))
            b[s.rows, s.columns] = 1
            degree = b.sum(axis=0)
            pair = b.T @ b
            expected = b.T @ np.diag(1 / np.log(2 + s.user_degree)) @ b
            expected /= np.sqrt(degree[:, None] * degree[None, :])
            expected *= pair / (pair + 2)
            expected[pair < 2] = 0
            np.fill_diagonal(expected, 0)
            index = ItemCFIndex(root)
            try:
                for row in range(len(s.items)):
                    actual = np.zeros(len(s.items))
                    valid = index.neighbors[row] >= 0
                    actual[index.neighbors[row, valid]] = index.similarities[row, valid]
                    np.testing.assert_allclose(actual, expected[row], atol=1e-7)
            finally:
                index.close()

    def test_topk_ties_are_deterministic_and_query_work_is_bounded(self):
        cfg = IndexConfig(neighbors=1, max_user_items=5, seed_items=2,
                          user_iuf=False, query_half_life_days=None, block_items=1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'index'
            fixture(cfg, root)
            index = ItemCFIndex(root)
            try:
                # item40 is linked equally to item20 and item30; raw item ID breaks tie.
                self.assertEqual(int(index.items[index.neighbors[3, 0]]), 20)
                result = index.query(3, 1000)
                self.assertLessEqual(result['edges_visited'], 2)
                self.assertEqual(len(result['items']), len(set(result['items'])))
                self.assertEqual(result['shortfall'], 1000 - len(result['items']))
                self.assertEqual(len(list(index.query_many([1, 99], 2))), 2)
            finally:
                index.close()

    def test_fresh_histories_filter_future_unknown_and_duplicates_and_decay_scores(self):
        cfg = IndexConfig(neighbors=4, max_user_items=5, seed_items=5,
                          user_iuf=False, query_half_life_days=1.)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'index'
            fixture(cfg, root)
            index = ItemCFIndex(root)
            try:
                now = 86_400_100
                history = [(10, 100), (10, 90), (999, 50), (20, now), (30, now + 1)]
                decayed = index.query(99, 4, now, history=history)
                raw = index.query(99, 4, now, history=history, half_life_days=None)
                self.assertEqual(decayed['ignored_history'], {'not_prior': 2, 'unknown_item': 1, 'duplicate': 1})
                np.testing.assert_allclose(decayed['scores'], np.array(raw['scores']) / 2)
                self.assertEqual(decayed['seed_count'], 1)
                with self.assertRaises(ValueError):
                    index.query(1, timestamp_ms=99)
            finally:
                index.close()

    def test_publication_snapshot_and_checksum_guards(self):
        cfg = IndexConfig(neighbors=3, max_user_items=5, seed_items=3)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'index'
            s = fixture(cfg, root)
            with self.assertRaises(FileExistsError):
                build(s, cfg, root)
            with self.assertRaises(ValueError):
                build(s, replace(cfg, max_user_items=6), Path(temp) / 'bad')
            with self.assertRaises(ValueError):
                ItemCFIndex(root, expected_manifest_sha256='wrong')
            with (root / 'neighbors.npy').open('ab') as f:
                f.write(b'corruption')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                ItemCFIndex(root)

    def test_cap_window_and_configuration_validation(self):
        cfg = IndexConfig(neighbors=3, max_user_items=2, seed_items=2, window_days=1)
        s = make_snapshot([(1, 10, 90), (1, 20, 80), (1, 30, 70), (1, 40, -100_000_000)],
                          [10, 20, 30, 40], 100, cfg, 'fixture')
        self.assertEqual(s.columns.tolist(), [0, 1])
        self.assertEqual(s.user_degree.tolist(), [3])
        for args in ({'neighbors': 0}, {'query_half_life_days': 0}, {'shrinkage': -1},
                     {'seed_items': 201}, {'user_iuf': 'yes'}):
            with self.assertRaises(ValueError):
                IndexConfig(**args)

    def test_popularity_normalization_has_explicit_dense_reference(self):
        cfg = IndexConfig(neighbors=4, max_user_items=5, seed_items=5,
                          user_iuf=False, popularity_alpha=.25)
        s = make_snapshot([(1, 10, 90), (1, 20, 80), (2, 10, 90), (2, 30, 80)],
                          [10, 20, 30], 100, cfg, 'fixture')
        normalized, binary = graph_matrices(s, cfg)
        b = binary.toarray()
        cooccurrence = b.T @ b
        degree = b.sum(axis=0)
        expected = cooccurrence / (degree[:, None] * degree[None, :]) ** .25
        np.testing.assert_allclose((normalized.T @ normalized).toarray(), expected)

    def test_graph_decay_matches_time_weighted_pair_reference(self):
        cfg = IndexConfig(neighbors=3, max_user_items=3, seed_items=3,
                          user_iuf=False, graph_half_life_days=1.)
        day = 86_400_000
        cutoff = 4 * day
        s = make_snapshot([(1, 10, cutoff - day // 10), (1, 20, cutoff - 2 * day),
                           (2, 10, cutoff - day // 2), (2, 30, cutoff - day)],
                          [10, 20, 30], cutoff, cfg, 'fixture')
        normalized, _ = graph_matrices(s, cfg)
        # Relative recency of the two users' item10 events changes the pair strengths.
        first, second = 2 ** (-.1), 2 ** (-.5)
        expected = np.array([[1., np.sqrt(first / (first + second)), np.sqrt(second / (first + second))],
                             [np.sqrt(first / (first + second)), 1., 0.],
                             [np.sqrt(second / (first + second)), 0., 1.]])
        np.testing.assert_allclose((normalized.T @ normalized).toarray(), expected)

    def test_excluding_supplied_history_covers_items_beyond_seed_budget(self):
        cfg = IndexConfig(neighbors=4, max_user_items=5, seed_items=2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'index'
            fixture(cfg, root)
            index = ItemCFIndex(root)
            try:
                result = index.query(99, 4, history=[(10, 95), (20, 90), (30, 80)], exclude_history=True)
                self.assertEqual(result['seed_count'], 2)
                self.assertEqual(result['items'], [40])
            finally:
                index.close()

    def test_export_is_exact_prefix_and_records_lineage(self):
        cfg = IndexConfig(neighbors=4, max_user_items=5, seed_items=3)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'index'
            fixture(cfg, root)
            source = ItemCFIndex(root)
            try:
                metadata = export_prefix(source, Path(temp) / 'small', 2)
                smaller = ItemCFIndex(Path(temp) / 'small')
                try:
                    with redirect_stdout(io.StringIO()):
                        direct_config = replace(cfg, neighbors=2)
                        events = [(1, 10, 90), (1, 20, 80), (1, 20, 70),
                                  (2, 10, 90), (2, 30, 80),
                                  (3, 10, 90), (3, 20, 80), (3, 30, 70), (3, 40, 60)]
                        snapshot = make_snapshot(events, [10, 20, 30, 40], 100, direct_config, 'fixture-manifest')
                        build(snapshot, direct_config, Path(temp) / 'direct')
                    np.testing.assert_array_equal(smaller.neighbors, np.load(Path(temp) / 'direct/neighbors.npy'))
                    np.testing.assert_array_equal(smaller.similarities, np.load(Path(temp) / 'direct/similarities.npy'))
                    for uid in (1, 2, 3, 99):
                        a = source.query(uid, 4, neighbor_limit=2)
                        b = smaller.query(uid, 4)
                        self.assertEqual(a['items'], b['items'])
                        self.assertEqual(a['scores'], b['scores'])
                    self.assertEqual(metadata['derived_from_version'], source.metadata['version'])
                    self.assertLess(metadata['array_bytes'], source.metadata['array_bytes'])
                finally:
                    smaller.close()
            finally:
                source.close()


if __name__ == '__main__':
    unittest.main()
