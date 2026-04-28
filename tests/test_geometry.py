"""Unit tests for geometry module."""

import pytest
import math
from mathcore.core.geometry import (
    Point, Line, Circle, Triangle, Vector,
    distance, midpoint, slope, find_line_intersection,
    line_collision_detection, circle_line_intersection,
    point_to_line_distance
)


class TestPoint:
    """Tests for Point class."""
    
    def test_point_creation(self):
        p = Point(3, 4)
        assert p.x == 3
        assert p.y == 4
    
    def test_point_distance(self):
        p1 = Point(0, 0)
        p2 = Point(3, 4)
        assert p1.distance_to(p2) == 5.0
    
    def test_point_translate(self):
        p = Point(1, 2)
        p_translated = p.translate(2, 3)
        assert p_translated.x == 3
        assert p_translated.y == 5
    
    def test_point_rotate(self):
        p = Point(1, 0)
        p_rotated = p.rotate(math.pi / 2)  # 90 degrees
        assert abs(p_rotated.x) < 1e-10
        assert abs(p_rotated.y - 1) < 1e-10


class TestLine:
    """Tests for Line class."""
    
    def test_line_from_points(self):
        p1 = Point(0, 0)
        p2 = Point(1, 1)
        line = Line(p1, p2)
        assert line.slope == 1.0
    
    def test_line_evaluate(self):
        line = Line(slope=2, intercept=1)  # y = 2x + 1
        assert line.evaluate(0) == 1
        assert line.evaluate(1) == 3
    
    def test_line_contains_point(self):
        line = Line(Point(0, 0), Point(1, 1))
        assert line.contains_point(Point(2, 2)) == True
        assert line.contains_point(Point(1, 0)) == False
    
    def test_line_intersection(self):
        line1 = Line(Point(0, 0), Point(1, 1))  # y = x
        line2 = Line(Point(0, 1), Point(1, 0))  # y = -x + 1
        intersection = find_line_intersection(line1, line2)
        assert intersection is not None
        assert abs(intersection.x - 0.5) < 1e-10
        assert abs(intersection.y - 0.5) < 1e-10
    
    def test_line_parallel(self):
        line1 = Line(slope=2, intercept=0)
        line2 = Line(slope=2, intercept=1)
        intersection = find_line_intersection(line1, line2)
        assert intersection is None


class TestCircle:
    """Tests for Circle class."""
    
    def test_circle_creation(self):
        circle = Circle(Point(0, 0), 5)
        assert circle.radius == 5
    
    def test_circle_area(self):
        circle = Circle(Point(0, 0), 1)
        assert abs(circle.area() - math.pi) < 1e-10
    
    def test_circle_circumference(self):
        circle = Circle(Point(0, 0), 1)
        assert abs(circle.circumference() - 2 * math.pi) < 1e-10
    
    def test_circle_contains_point(self):
        circle = Circle(Point(0, 0), 5)
        assert circle.contains_point(Point(3, 4)) == True
        assert circle.contains_point(Point(10, 0)) == False


class TestTriangle:
    """Tests for Triangle class."""
    
    def test_triangle_creation(self):
        t = Triangle(Point(0, 0), Point(3, 0), Point(0, 4))
        assert len(t.vertices) == 3
    
    def test_triangle_perimeter(self):
        t = Triangle(Point(0, 0), Point(3, 0), Point(0, 4))
        assert abs(t.perimeter() - 12) < 1e-10  # 3 + 4 + 5
    
    def test_triangle_area(self):
        t = Triangle(Point(0, 0), Point(4, 0), Point(0, 3))
        assert abs(t.area() - 6) < 1e-10
    
    def test_triangle_centroid(self):
        t = Triangle(Point(0, 0), Point(3, 0), Point(0, 3))
        centroid = t.centroid()
        assert abs(centroid.x - 1) < 1e-10
        assert abs(centroid.y - 1) < 1e-10


class TestVector:
    """Tests for Vector class."""
    
    def test_vector_magnitude(self):
        v = Vector(3, 4)
        assert v.magnitude() == 5.0
    
    def test_vector_normalize(self):
        v = Vector(3, 4)
        v_norm = v.normalize()
        assert abs(v_norm.magnitude() - 1) < 1e-10
    
    def test_vector_dot_product(self):
        v1 = Vector(1, 0)
        v2 = Vector(0, 1)
        assert v1.dot_product(v2) == 0
    
    def test_vector_add(self):
        v1 = Vector(1, 2)
        v2 = Vector(3, 4)
        v3 = v1 + v2
        assert v3.x == 4
        assert v3.y == 6


class TestGeometryFunctions:
    """Tests for geometry utility functions."""
    
    def test_distance(self):
        p1 = Point(0, 0)
        p2 = Point(3, 4)
        assert distance(p1, p2) == 5.0
    
    def test_midpoint(self):
        p1 = Point(0, 0)
        p2 = Point(4, 4)
        mid = midpoint(p1, p2)
        assert mid.x == 2
        assert mid.y == 2
    
    def test_slope(self):
        p1 = Point(0, 0)
        p2 = Point(1, 2)
        assert slope(p1, p2) == 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
