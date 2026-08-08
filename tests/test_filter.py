import unittest

import numpy as np

from trpl_track import filter as filt
from trpl_track.tracklet import ObjectTraj

T = 60


def _trlet(startt, endt, gx_start=0.0, gx_end=0.0, box=(0, 0, 9, 9)):
    t = ObjectTraj(T=T, startt=startt, endt=endt)
    t.trj_3d[startt, 0] = gx_start
    t.trj_3d[endt, 0] = gx_end
    for tt in range(startt, endt + 1):
        for cam in range(2):
            t.trj[cam][tt] = box
    return t


class TestFilter(unittest.TestCase):
    def test_gating_valid_link(self):
        a = _trlet(0, 2, gx_end=100.0)
        b = _trlet(5, 10, gx_start=110.0)
        Tff = filt.prepare_valid_linkset([a, b])
        self.assertEqual(Tff[0, 1], 1)   # disjoint, gap 3<=36, speed 10/50<=3
        self.assertEqual(Tff[1, 0], 0)   # b ends after a starts -> not a predecessor
        self.assertEqual(Tff[0, 0], 0)   # no self link

    def test_gating_speed_too_high(self):
        a = _trlet(0, 2, gx_end=100.0)
        b = _trlet(5, 10, gx_start=1000.0)     # |900|/50 = 18 > gap 3
        Tff = filt.prepare_valid_linkset([a, b])
        self.assertEqual(Tff[0, 1], 0)

    def test_gating_gap_too_large(self):
        a = _trlet(0, 2, gx_end=100.0)
        b = _trlet(45, 50, gx_start=101.0)     # 2+36=38 < 45
        Tff = filt.prepare_valid_linkset([a, b])
        self.assertEqual(Tff[0, 1], 0)

    def test_filter_tracklets_length(self):
        good, idx = filt.filter_tracklets([_trlet(0, 2), _trlet(0, 1)])
        self.assertEqual(len(good), 1)     # len>=3 kept, len 2 dropped
        self.assertEqual(idx, [0])

    def test_appmodel_match_symmetric(self):
        a = _trlet(0, 2); b = _trlet(0, 2)
        for cam in range(2):
            a.hist_p[cam][0, 0] = 1.0; a.hist_q[cam][0, 1] = 1.0
            b.hist_p[cam][0, 0] = 0.5; b.hist_p[cam][0, 2] = 0.5
            b.hist_q[cam][0, 3] = 1.0
        self.assertAlmostEqual(filt.appmodel_match(a, b), filt.appmodel_match(b, a), places=6)

    def test_prepare_app_affinity_offset(self):
        a = _trlet(0, 2, gx_end=100.0)
        b = _trlet(5, 10, gx_start=110.0)
        for cam in range(2):
            a.hist_p[cam][0, 0] = 1.0; a.hist_q[cam][0, 1] = 1.0
            b.hist_p[cam][0, 0] = 1.0; b.hist_q[cam][0, 1] = 1.0
        Tff = filt.prepare_valid_linkset([a, b])
        Aff = filt.prepare_app_affinity(Tff, [a, b])
        expected = filt.appmodel_match(a, b) - filt.APP_MATCH_THR
        self.assertAlmostEqual(Aff[0, 1], expected, places=6)
        self.assertEqual(Aff[1, 0], 0.0)   # not gated


if __name__ == "__main__":
    unittest.main()
