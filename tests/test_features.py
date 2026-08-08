import unittest

import numpy as np

from trpl_track import features as F
from trpl_track.config import NBINS


class TestFeatures(unittest.TestCase):
    def test_sat_scalar_and_array(self):
        self.assertEqual(F.sat(5, 3), 3)
        self.assertEqual(F.sat(2, 3), 2)
        self.assertEqual(F.sat2(9, 2, 8), 8)
        self.assertEqual(F.sat2(1, 2, 8), 2)
        np.testing.assert_array_equal(F.sat2(np.array([1, 5, 9]), 2, 8), [2, 5, 8])

    def test_rgb_bin_image(self):
        img = np.array([[[32, 0, 0], [0, 64, 0], [0, 0, 96]]], dtype=np.uint8)
        b = F.rgb_bin_image(img)
        np.testing.assert_array_equal(b, [[1, 16, 192]])

    def test_compute_part_rects(self):
        model = [(0.0, 0.0, 1.0, 1.0), (0.25, 0.0, 0.75, 0.2)]
        r = F.compute_part_rects(10, 20, 30, 40, model)
        np.testing.assert_allclose(r[0], [10, 20, 40, 60])
        np.testing.assert_allclose(r[1], [10 + 0.25 * 30, 20, 10 + 0.75 * 30, 20 + 0.2 * 40])

    def test_collect_hist_normalized(self):
        bin_img = np.zeros((20, 20), dtype=np.int64)   # every pixel -> bin 0
        rects = np.array([[0.0, 0.0, 10.0, 10.0]])
        hp, hq = F.collect_hist(bin_img, rects)
        self.assertEqual(hp.shape, (1, NBINS))
        self.assertAlmostEqual(hp[0].sum(), 1.0, places=6)
        self.assertAlmostEqual(hp[0, 0], 1.0, places=6)   # all mass in bin 0

    def test_kldivergence_rows(self):
        hp = np.zeros((2, NBINS)); hq = np.zeros((2, NBINS))
        hp[0, 0] = 1.0; hq[0, 0] = 1.0          # identical -> 0
        hp[1, 0] = 1.0; hq[1, 1] = 1.0          # disjoint  -> > 0
        v = F.kldivergence_rows(hp, hq)
        self.assertAlmostEqual(v[0], 0.0, places=6)
        self.assertGreater(v[1], 0.0)

    def test_candidate_array_subpixel(self):
        ca = F.CandidateArray()
        ca.fill_fxfy(feetx=0, feety=0, xr=[0, 10], yr=[0, 10], ns=1)
        ijs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]])
        ca.fill_score(np.array([1.0, 2.0, 3.0, 4.0]), ijs)
        self.assertAlmostEqual(ca.get_subpixel_score(0, 0)[0], 1.0)
        self.assertAlmostEqual(ca.get_subpixel_score(5, 5)[0], 2.5)  # bilinear mean
        self.assertAlmostEqual(ca.get_subpixel_score(100, 100)[0], -5.0)  # out of range


if __name__ == "__main__":
    unittest.main()
