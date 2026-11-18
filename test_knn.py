"""Run with: python -m unittest discover -v"""
import unittest
import numpy as np
from knn_advanced import KNNClassifier, pairwise_distances, find_best_k
from knn_regression import KNNRegressor, find_best_k as best_regression_k
from knn_kdtree import KDTree, brute_force_knn
from KNN import knn_predict


class KNNTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0., 0.], [1., 0.], [0., 1.], [5., 5.], [6., 5.], [5., 6.]])
        self.y = np.array(['A'] * 3 + ['B'] * 3)

    def test_basic(self):
        label, neighbors = knn_predict(self.X, self.y, [0, 0], 3)
        self.assertEqual(label, 'A')
        self.assertEqual(len(neighbors), 3)

    def test_classifier_chunk_matches_full(self):
        query = np.array([[0., 0.], [5.5, 5.5], [1., 1.]])
        for metric in ('euclidean', 'manhattan', 'chebyshev', 'minkowski', 'cosine'):
            small = KNNClassifier(k=3, metric=metric, batch_size=1).fit(self.X, self.y)
            large = KNNClassifier(k=3, metric=metric, batch_size=100).fit(self.X, self.y)
            np.testing.assert_array_equal(small.predict(query), large.predict(query))
            np.testing.assert_allclose(small.predict_proba(query), large.predict_proba(query))
            np.testing.assert_allclose(small.predict_proba(query).sum(axis=1), 1.)

    def test_exact_match_ignores_nonzero_distance(self):
        x = np.array([[0.], [0.], [1.]])
        c = KNNClassifier(k=3, weights='distance').fit(x, ['A', 'B', 'B'])
        np.testing.assert_allclose(c.predict_proba([[0.]]), [[0.5, 0.5]])
        r = KNNRegressor(k=3, weights='distance').fit(x, [2., 4., 100.])
        self.assertAlmostEqual(r.predict([[0.]])[0], 3.)

    def test_validation(self):
        with self.assertRaises(ValueError): KNNClassifier(k=0)
        with self.assertRaises(ValueError): KNNRegressor(k=0)
        with self.assertRaises(ValueError): KNNClassifier().fit(self.X, self.y[:-1])
        with self.assertRaises(ValueError): pairwise_distances(self.X, np.ones((2, 3)))
        with self.assertRaises(ValueError): find_best_k(self.X, self.y, [99], n_splits=3)
        with self.assertRaises(ValueError): best_regression_k(self.X, np.arange(6.), [99], n_splits=3)

    def test_kdtree_matches_bruteforce(self):
        rng = np.random.default_rng(17)
        X = rng.random((100, 3))
        Q = rng.random((12, 3))
        for metric in ('euclidean', 'manhattan', 'chebyshev', 'minkowski'):
            tree = KDTree(X, leaf_size=8, metric=metric)
            a, _ = tree.query(Q, 5)
            b, _ = brute_force_knn(X, Q, 5, metric=metric)
            np.testing.assert_allclose(a, b)

    def test_regressor_consistency(self):
        target = np.array([0., 1., 1.5, 10., 11., 11.5])
        query = np.array([[0.3, 0.2], [5.1, 5.2]])
        brute = KNNRegressor(k=3, algorithm='brute').fit(self.X, target)
        tree = KNNRegressor(k=3, algorithm='kd_tree').fit(self.X, target)
        np.testing.assert_allclose(brute.predict(query), tree.predict(query))


if __name__ == '__main__':
    unittest.main()
