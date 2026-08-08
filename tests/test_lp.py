import unittest

import numpy as np

from trpl_track.lp import solve_linprog


class TestLP(unittest.TestCase):
    def test_empty(self):
        Tff = np.zeros((3, 3), int)
        Aff = np.zeros((3, 3))
        LMat, links = solve_linprog(Tff, Aff)
        self.assertEqual(links.shape, (0, 2))
        self.assertEqual(int(LMat.sum()), 0)

    def test_single_successor_constraint(self):
        # node 0 may link to 1 or 2, but only one successor allowed
        Tff = np.zeros((3, 3), int)
        Tff[0, 1] = Tff[0, 2] = 1
        Aff = np.zeros((3, 3))
        Aff[0, 1] = 5.0
        Aff[0, 2] = 3.0
        LMat, links = solve_linprog(Tff, Aff)
        self.assertEqual([tuple(r) for r in links], [(0, 1)])
        self.assertEqual(int(LMat.sum()), 1)

    def test_bipartite_matching(self):
        ng = 4
        Tff = np.zeros((ng, ng), int)
        for i in (0, 1):
            for j in (2, 3):
                Tff[i, j] = 1
        Aff = np.zeros((ng, ng))
        Aff[0, 2] = 10.0; Aff[0, 3] = 1.0
        Aff[1, 2] = 1.0;  Aff[1, 3] = 10.0
        LMat, links = solve_linprog(Tff, Aff)
        got = {tuple(r) for r in links}
        self.assertEqual(got, {(0, 2), (1, 3)})
        # each row and column used at most once
        self.assertTrue((LMat.sum(axis=1) <= 1).all())
        self.assertTrue((LMat.sum(axis=0) <= 1).all())

    def test_negative_weights_not_linked(self):
        Tff = np.zeros((2, 2), int)
        Tff[0, 1] = 1
        Aff = np.zeros((2, 2))
        Aff[0, 1] = -3.0        # negative -> not worth linking
        LMat, links = solve_linprog(Tff, Aff)
        self.assertEqual(links.shape, (0, 2))


if __name__ == "__main__":
    unittest.main()
