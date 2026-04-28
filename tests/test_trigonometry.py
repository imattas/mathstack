"""Unit tests for trigonometry package."""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mathcore.trigonometry import (
    sin,
    cos,
    tan,
    radians,
    degrees,
    unit_circle_point,
    law_of_cosines_side,
    triangle_area_heron,
    verify_pythagorean_identity,
)


class TestTrigonometry(unittest.TestCase):
    def test_basic_trig_values(self):
        self.assertAlmostEqual(sin(math.pi / 2), 1.0, places=10)
        self.assertAlmostEqual(cos(0.0), 1.0, places=10)
        self.assertAlmostEqual(tan(0.0), 0.0, places=10)

    def test_angle_conversion(self):
        self.assertAlmostEqual(radians(180.0), math.pi, places=10)
        self.assertAlmostEqual(degrees(math.pi / 2), 90.0, places=10)

    def test_unit_circle_point(self):
        x, y = unit_circle_point(math.pi / 2)
        self.assertAlmostEqual(x, 0.0, places=10)
        self.assertAlmostEqual(y, 1.0, places=10)

    def test_law_of_cosines_side(self):
        side = law_of_cosines_side(3.0, 4.0, math.pi / 2)
        self.assertAlmostEqual(side, 5.0, places=10)

    def test_heron_area(self):
        area = triangle_area_heron(3.0, 4.0, 5.0)
        self.assertAlmostEqual(area, 6.0, places=10)

    def test_identity(self):
        self.assertTrue(verify_pythagorean_identity(1.2345))


if __name__ == "__main__":
    unittest.main()
