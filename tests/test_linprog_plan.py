import unittest

import numpy as np

from trpl_track import linprog_plan as LP


class TestLinprogPlan(unittest.TestCase):
    def test_chord_length(self):
        path = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])
        np.testing.assert_allclose(LP.chord_length(path), [0.0, 3.0, 7.0])

    def test_nearest_point(self):
        path = np.array([[0.0, 0.0], [10.0, 0.0]])
        self.assertEqual(LP.nearest_point(path, [9.0, 0.0]), 1)
        self.assertEqual(LP.nearest_point(path, [1.0, 0.0]), 0)

    def test_trj2path_distance_on_path_is_zero(self):
        path = np.array([[0.0, float(i)] for i in range(11)])
        trj = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
        self.assertAlmostEqual(LP.trj2path_distance(trj, path), 0.0, places=6)

    def test_guided_gap_interp_shape_and_bounds(self):
        path = np.array([[0.0, float(i)] for i in range(11)])
        gap = LP.guided_gap_interp(np.array([0.0, 0.0]), np.array([0.0, 10.0]), path, 4)
        self.assertEqual(gap.shape, (4, 2))
        self.assertTrue(np.isfinite(gap).all())
        # interior stays within the endpoints' y-range
        self.assertTrue((gap[:, 1] >= -1).all() and (gap[:, 1] <= 11).all())

    def test_enumerate_rects_refine(self):
        rect = [0.0, 0.0, 10.0, 10.0]
        dx = np.array([-4.0, 0.0, 4.0])
        dy = np.array([-4.0, 0.0, 4.0])
        cr = LP._enumerate_rects_refine(rect, dx, dy)
        self.assertEqual(cr.shape, (9, 4))
        # the zero-offset candidate equals the original rect
        self.assertTrue(any(np.allclose(row, rect) for row in cr))


if __name__ == "__main__":
    unittest.main()
