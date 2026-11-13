"""The instability index recomputation and the early-termination bound."""

from __future__ import annotations

import unittest

import numpy as np

from analysis.gait120_index_validation import (
    IndexDefinition,
    _roc_auc,
    index_trace,
)


class IndexTraceTests(unittest.TestCase):
    def test_margin_inside_the_polygon_scores_zero(self) -> None:
        trace = index_trace(np.full(20, 0.1), 0.03, IndexDefinition())
        self.assertTrue(np.all(trace == 0.0))

    def test_margin_far_outside_scores_the_weights_of_the_level_terms(self) -> None:
        trace = index_trace(np.full(20, -2.0), 0.03, IndexDefinition())
        self.assertAlmostEqual(float(trace[-1]), 0.80, places=6)

    def test_no_finite_margin_scores_the_flight_value(self) -> None:
        trace = index_trace(np.full(5, np.nan), 0.03, IndexDefinition())
        self.assertTrue(np.allclose(trace, 0.25))

    def test_higher_ramps_do_not_raise_the_index(self) -> None:
        margin = np.linspace(0.0, -1.0, 34)
        base = index_trace(margin, 0.03, IndexDefinition())
        higher = index_trace(
            margin, 0.03, IndexDefinition(minimum=(0.525, 0.5), end=(0.375, 0.4375), slope=(0.1875, 0.5625))
        )
        self.assertTrue(np.all(higher <= base + 1e-12))


class RocTests(unittest.TestCase):
    def test_separated_groups(self) -> None:
        self.assertEqual(_roc_auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])), 1.0)

    def test_identical_groups(self) -> None:
        self.assertEqual(_roc_auc(np.array([1.0, 1.0]), np.array([1.0, 1.0])), 0.5)


if __name__ == "__main__":
    unittest.main()
