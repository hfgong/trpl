import tempfile
import unittest
from pathlib import Path

import numpy as np

from trpl_track.tracklet import ObjectTraj, save_tracklets, load_tracklets


class TestTracklet(unittest.TestCase):
    def test_defaults(self):
        t = ObjectTraj(T=10, startt=2, endt=5)
        self.assertEqual(len(t.trj), 2)
        self.assertEqual(t.trj[0].shape, (10, 4))
        self.assertEqual(t.trj_3d.shape, (10, 2))
        self.assertEqual(t.length, 4)
        self.assertFalse(t.is_empty())
        self.assertTrue(ObjectTraj(T=10).is_empty())

    def test_save_load_roundtrip(self):
        a = ObjectTraj(T=8, startt=1, endt=4, state=2)
        a.trj[0][1] = [1, 2, 3, 4]
        a.trj_3d[2] = [7.5, 8.5]
        a.scores[0, 3] = 9.0
        a.hist_p[1][0, 5] = 0.25
        b = ObjectTraj(T=8, startt=0, endt=7)
        p = Path(tempfile.mkdtemp()) / "trlets.npz"
        save_tracklets(p, [a, b])
        loaded = load_tracklets(p)
        self.assertEqual(len(loaded), 2)
        la = loaded[0]
        self.assertEqual((la.startt, la.endt, la.state), (1, 4, 2))
        np.testing.assert_allclose(la.trj[0][1], [1, 2, 3, 4])
        np.testing.assert_allclose(la.trj_3d[2], [7.5, 8.5])
        self.assertAlmostEqual(la.scores[0, 3], 9.0)
        self.assertAlmostEqual(la.hist_p[1][0, 5], 0.25)


if __name__ == "__main__":
    unittest.main()
