import unittest

import numpy as np

from trpl_track import planning as P
from trpl_track.tracklet import ObjectTraj


class TestPlanning(unittest.TestCase):
    def test_fix_poly_ground_drops_closing_vertex_and_clamps(self):
        pg = np.array([[0, 5, 5, -1, 0],
                       [0, 0, 3, 3, 0]], float)   # last col == first col (closed)
        out = P.fix_poly_ground(pg)
        self.assertEqual(out.shape[1], 4)          # closing vertex dropped
        self.assertGreaterEqual(out.min(), 0.0)    # negatives clamped

    def test_carboxes2carobs(self):
        cars = np.array([[10.0, 20.0, 30.0, 40.0, 0.5]])
        polys = P.carboxes2carobs(cars, np.eye(3))
        self.assertEqual(len(polys), 1)
        self.assertEqual(polys[0].shape, (4, 2))
        # non-degenerate quad
        self.assertGreater(np.ptp(polys[0][:, 1]), 0)

    def test_prepare_ped_obs(self):
        t = ObjectTraj(T=10, startt=2, endt=4)
        t.trj_3d[3] = [100.0, 50.0]
        ped = P.prepare_ped_obs([t], 10)
        sq = ped[0][3]
        self.assertEqual(sq.shape, (4, 2))
        thick = 0.25 * 1.2 * 100 / 5.0
        self.assertAlmostEqual(sq[:, 0].min(), 100.0 - thick)
        self.assertAlmostEqual(sq[:, 0].max(), 100.0 + thick)
        self.assertIsNone(ped[0][0])               # outside [startt,endt]

    def test_build_state_graph(self):
        obs = np.zeros((3, 3), int)
        ig2yx, yx2ig, src, dst, nbr = P.build_state_graph(obs)
        self.assertEqual(len(ig2yx), 9)
        self.assertTrue((yx2ig >= 0).all())
        self.assertGreater(len(src), 0)
        self.assertTrue(src.max() < 9 and dst.max() < 9)

    def test_is_looped(self):
        ig2yx = np.array([[0, i] for i in range(5)])
        self.assertTrue(P.is_looped([0, 1, 0], ig2yx))      # repeated node
        self.assertFalse(P.is_looped([0, 1, 2, 3, 4], ig2yx))

    def test_shortest_path_line(self):
        obs = np.zeros((1, 5), int)
        ig2yx, yx2ig, src, dst, nbr = P.build_state_graph(obs)
        feat = [np.ones((1, 5)) for _ in range(3)]
        ew = P.edge_weights(ig2yx, src, dst, nbr, feat, [1.0, 0.0, 0.0])
        dist, path = P.shortest_path(ig2yx, src, dst, ew, 0, 4)
        self.assertEqual(path, [0, 1, 2, 3, 4])
        self.assertAlmostEqual(dist, 4.0, places=6)

    def test_get_legal_index(self):
        obs = np.zeros((3, 3), int)
        ig2yx, yx2ig, src, dst, nbr = P.build_state_graph(obs)
        self.assertEqual(P.get_legal_index(yx2ig, ig2yx, 1, 1), yx2ig[1, 1])
        # far point snaps to nearest free cell (the far corner)
        self.assertEqual(P.get_legal_index(yx2ig, ig2yx, 100, 100), yx2ig[2, 2])


if __name__ == "__main__":
    unittest.main()
