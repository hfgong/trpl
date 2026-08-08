import unittest

import numpy as np

from trpl_track.finalize import finalize_trajectory_plan
from trpl_track.tracklet import ObjectTraj

T = 20


def _trlet(startt, endt):
    return ObjectTraj(T=T, startt=startt, endt=endt)


class TestFinalize(unittest.TestCase):
    def test_isolated_when_no_links(self):
        good = [_trlet(0, 2), _trlet(5, 8)]
        gap_rind = -np.ones((2, 2), int)
        gap = [[[] for _ in range(2)] for _ in range(2)]
        final, index, state = finalize_trajectory_plan(2, T, np.zeros((0, 2), int),
                                                       good, gap_rind, gap)
        self.assertEqual(len(final), 2)
        self.assertEqual(sorted(index), [[0], [1]])   # each chain is a singleton

    def test_chain_no_gap(self):
        good = [_trlet(0, 2), _trlet(3, 5)]     # adjacent -> no gap frames
        gap_rind = -np.ones((2, 2), int)
        gap = [[[] for _ in range(2)] for _ in range(2)]
        links = np.array([[0, 1]])
        final, index, state = finalize_trajectory_plan(2, T, links, good, gap_rind, gap)
        self.assertEqual(len(final), 1)
        f = final[0]
        self.assertEqual((f.startt, f.endt), (0, 5))
        self.assertTrue((state[0, 0:6] == 1).all())     # all observed
        self.assertTrue((state[0, 6:] == -1).all())

    def test_chain_with_gap(self):
        good = [_trlet(0, 2), _trlet(4, 6)]     # gap at frame 3
        gap_rind = -np.ones((2, 2), int)
        gap_rind[0, 1] = 0
        gt = ObjectTraj(T=T, startt=3, endt=3)
        gt.trj_3d[3] = [1.0, 2.0]
        gap = [[[] for _ in range(2)] for _ in range(2)]
        gap[0][1] = [gt]
        links = np.array([[0, 1]])
        final, index, state = finalize_trajectory_plan(2, T, links, good, gap_rind, gap)
        self.assertEqual(len(final), 1)
        self.assertEqual(index[0], [0, 1])
        self.assertEqual(state[0, 3], 0)          # gap-filled
        self.assertEqual(state[0, 2], 1)          # observed
        np.testing.assert_allclose(final[0].trj_3d[3], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
