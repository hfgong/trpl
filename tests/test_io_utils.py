import tempfile
import unittest
from pathlib import Path

import numpy as np

from trpl_track import io_utils


def _write(text):
    f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    f.write(text)
    f.close()
    return Path(f.name)


class TestIoUtils(unittest.TestCase):
    def test_read_text_array2d_delimiters(self):
        p = _write("1, 2, 3\n4 5 6\n")
        a = io_utils.read_text_array2d(p)
        np.testing.assert_array_equal(a, [[1, 2, 3], [4, 5, 6]])

    def test_read_text_array2d_ragged_padded(self):
        p = _write("1 2 3\n4 5\n")
        a = io_utils.read_text_array2d(p)
        np.testing.assert_array_equal(a, [[1, 2, 3], [4, 5, 0]])

    def test_read_keyvalue_pairs(self):
        p = _write("horiz_mean = 456.58\nhoriz_sig=7.2302\n")
        d = io_utils.read_keyvalue_pairs(p)
        self.assertAlmostEqual(d["horiz_mean"], 456.58)
        self.assertAlmostEqual(d["horiz_sig"], 7.2302)

    def test_ublas_matrix_roundtrip(self):
        m = np.array([[1.5, -2.0], [3.0, 4.25]])
        p = _write("")
        io_utils.write_ublas_matrix(p, m)
        back = io_utils.read_ublas_matrix(p)
        np.testing.assert_allclose(back, m)

    def test_image_basename(self):
        self.assertEqual(io_utils.image_basename("a/b/cam1-x.jpg"), "cam1-x")

    def test_read_detection_refine_missing(self):
        arr = io_utils.read_detection_refine(None)
        self.assertEqual(arr.shape, (0, 4))

    def test_read_homography_field(self):
        text = ("[1,2](([3,3]((1,0,0),(0,1,0),(0,0,1)),"
                "[3,3]((2,0,0),(0,2,0),(0,0,1))))")
        p = _write(text)
        T, ncam, H = io_utils.read_homography_field(p)
        self.assertEqual((T, ncam), (1, 2))
        np.testing.assert_allclose(H[0][0], np.eye(3))
        np.testing.assert_allclose(H[0][1], np.diag([2.0, 2.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
