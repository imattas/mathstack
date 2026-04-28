"""
Comprehensive Geometry Package

All geometry functionality including:
- Basic shapes (Point, Vector, Line, Circle, Triangle, Polygon)
- Shape operations and properties
- Line intersections and distances
- Transformations (rotation, translation, reflection, scaling)
- Area and perimeter calculations
"""

from .shapes import Point, Vector, Line, Circle, Triangle, Polygon, Rectangle
from .transformations import rotate_point, translate_point, reflect_point, scale_point
from .transformations import rotate_shape, translate_shape, reflect_shape, scale_shape
from .calculations import distance, midpoint, angle_between, area_triangle
from .calculations import point_to_line_distance, line_slope, line_intercept
from .intersections import (
    find_line_intersection,
    line_collision_detection,
    circle_line_intersection,
    circle_circle_intersection,
)

__all__ = [
    'Point', 'Vector', 'Line', 'Circle', 'Triangle', 'Polygon', 'Rectangle',
    'rotate_point', 'translate_point', 'reflect_point', 'scale_point',
    'rotate_shape', 'translate_shape', 'reflect_shape', 'scale_shape',
    'distance', 'midpoint', 'angle_between', 'area_triangle',
    'point_to_line_distance', 'line_slope', 'line_intercept',
    'find_line_intersection', 'line_collision_detection',
    'circle_line_intersection', 'circle_circle_intersection',
]
