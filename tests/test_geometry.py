import unittest

import numpy as np

from trpl_track import geometry as geo


class TestGeometry(unittest.TestCase):
    def test_estimate_homography_identity(self):
        x = np.array([0, 1, 0, 1], float)
        y = np.array([0, 0, 1, 1], float)
        A = geo.estimate_homography(x, y, x, y)
        np.testing.assert_allclose(A, np.eye(3), atol=1e-9)

    def test_estimate_and_apply_translation(self):
        x = np.array([0, 1, 0, 1, 2], float)
        y = np.array([0, 0, 1, 1, 3], float)
        nx, ny = x + 10, y - 5
        A = geo.estimate_homography(x, y, nx, ny)
        ax, ay = geo.apply_homography(A, x, y)
        np.testing.assert_allclose(ax, nx, atol=1e-6)
        np.testing.assert_allclose(ay, ny, atol=1e-6)

    def test_apply_homography_roundtrip(self):
        A = geo.estimate_homography([0, 1, 0, 1], [0, 0, 1, 1],
                                    [2, 5, 1, 4], [3, 3, 7, 8])
        Ainv = np.linalg.inv(A)
        x = np.array([0.3, 0.7, 0.9])
        y = np.array([0.2, 0.5, 0.1])
        gx, gy = geo.apply_homography(A, x, y)
        bx, by = geo.apply_homography(Ainv, gx, gy)
        np.testing.assert_allclose(bx, x, atol=1e-9)
        np.testing.assert_allclose(by, y, atol=1e-9)

    def test_point_in_polygon(self):
        xs = np.array([0, 1, 1, 0], float)
        ys = np.array([0, 0, 1, 1], float)
        self.assertTrue(geo.point_in_polygon(0.5, 0.5, xs, ys))
        self.assertFalse(geo.point_in_polygon(2.0, 2.0, xs, ys))
        self.assertFalse(geo.point_in_polygon(-0.1, 0.5, xs, ys))

    def test_mask_from_polygon(self):
        xs = np.array([1, 4, 4, 1], float)
        ys = np.array([1, 1, 3, 3], float)
        m = geo.mask_from_polygon(5, 6, xs, ys)
        self.assertEqual(m.shape, (5, 6))
        self.assertEqual(m.dtype, np.uint8)
        self.assertGreater(int(m.sum()), 0)

    def test_get_plane_intersection(self):
        KK = np.eye(3)
        plane = np.array([0.0, 0.0, 1.0, -5.0])   # z = 5
        p2d = np.array([[2.0, 4.0], [3.0, 6.0]])
        p3d = geo.get_plane_intersection(KK, plane, p2d)
        np.testing.assert_allclose(p3d[2], [5.0, 5.0], atol=1e-9)
        np.testing.assert_allclose(p3d[0], [10.0, 20.0], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
