"""Geometric transformations: rotation, translation, reflection, scaling."""

import math
from .shapes import Point, Vector, Line, Circle, Triangle, Polygon, Rectangle


def rotate_point(point: Point, angle: float, center: Point = None) -> Point:
    """Rotate point by angle (radians) around center."""
    return point.rotate(angle, center)


def translate_point(point: Point, dx: float, dy: float) -> Point:
    """Translate point by (dx, dy)."""
    return point.translate(dx, dy)


def reflect_point(point: Point, line: Line) -> Point:
    """Reflect point across a line."""
    if line.slope == float('inf'):  # Vertical line
        x_line = line.point1.x
        reflected_x = 2 * x_line - point.x
        return Point(reflected_x, point.y)
    
    # Perpendicular slope
    perp_slope = -1 / line.slope if line.slope != 0 else float('inf')
    
    # Line through point perpendicular to reflection line
    if perp_slope == float('inf'):
        intersection_x = point.x
        intersection_y = line.evaluate(intersection_x)
    else:
        perp_intercept = point.y - perp_slope * point.x
        intersection_x = (perp_intercept - line.intercept) / (line.slope - perp_slope)
        intersection_y = line.evaluate(intersection_x)
    
    intersection = Point(intersection_x, intersection_y)
    
    # Reflect point
    reflected_x = 2 * intersection.x - point.x
    reflected_y = 2 * intersection.y - point.y
    return Point(reflected_x, reflected_y)


def scale_point(point: Point, factor: float, center: Point = None) -> Point:
    """Scale point relative to center."""
    if center is None:
        center = Point(0, 0)
    dx = point.x - center.x
    dy = point.y - center.y
    return Point(center.x + dx * factor, center.y + dy * factor)


def rotate_shape(vertices: list, angle: float, center: Point = None) -> list:
    """Rotate all vertices by angle around center."""
    return [rotate_point(v, angle, center) for v in vertices]


def translate_shape(vertices: list, dx: float, dy: float) -> list:
    """Translate all vertices by (dx, dy)."""
    return [translate_point(v, dx, dy) for v in vertices]


def reflect_shape(vertices: list, line: Line) -> list:
    """Reflect all vertices across line."""
    return [reflect_point(v, line) for v in vertices]


def scale_shape(vertices: list, factor: float, center: Point = None) -> list:
    """Scale all vertices relative to center."""
    return [scale_point(v, factor, center) for v in vertices]
