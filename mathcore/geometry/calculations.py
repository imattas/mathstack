"""Geometric calculations: distances, angles, areas, slopes."""

import math
from .shapes import Point, Vector, Line, Triangle


def distance(p1: Point, p2: Point) -> float:
    """Calculate distance between two points."""
    return p1.distance_to(p2)


def midpoint(p1: Point, p2: Point) -> Point:
    """Calculate midpoint between two points."""
    return Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)


def angle_between(v1: Vector, v2: Vector) -> float:
    """Calculate angle between two vectors (in radians)."""
    dot = v1.dot_product(v2)
    mag1 = v1.magnitude()
    mag2 = v2.magnitude()
    if mag1 == 0 or mag2 == 0:
        return 0
    cos_angle = dot / (mag1 * mag2)
    # Clamp to [-1, 1] to avoid numerical errors
    cos_angle = max(-1, min(1, cos_angle))
    return math.acos(cos_angle)


def angle_at_point(p1: Point, vertex: Point, p2: Point) -> float:
    """Calculate angle at vertex formed by three points (in radians)."""
    v1 = Vector(p1.x - vertex.x, p1.y - vertex.y)
    v2 = Vector(p2.x - vertex.x, p2.y - vertex.y)
    return angle_between(v1, v2)


def area_triangle(p1: Point, p2: Point, p3: Point) -> float:
    """Calculate area of triangle from three points."""
    triangle = Triangle(p1, p2, p3)
    return triangle.area()


def line_slope(p1: Point, p2: Point) -> float:
    """Calculate slope between two points."""
    if p1.x == p2.x:
        return float('inf')
    return (p2.y - p1.y) / (p2.x - p1.x)


def line_intercept(point: Point, slope: float) -> float:
    """Calculate y-intercept given point and slope."""
    return point.y - slope * point.x


def point_to_line_distance(point: Point, line: Line) -> float:
    """Calculate perpendicular distance from point to line."""
    if line.slope == float('inf'):  # Vertical line
        return abs(point.x - line.point1.x)
    
    # Distance = |ax + by + c| / sqrt(a² + b²)
    # For y = mx + b: mx - y + b = 0
    a = line.slope
    b = -1
    c = line.intercept
    
    numerator = abs(a * point.x + b * point.y + c)
    denominator = math.sqrt(a**2 + b**2)
    
    if denominator == 0:
        return 0
    return numerator / denominator


def degrees_to_radians(degrees: float) -> float:
    """Convert degrees to radians."""
    return degrees * math.pi / 180


def radians_to_degrees(radians: float) -> float:
    """Convert radians to degrees."""
    return radians * 180 / math.pi
