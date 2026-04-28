"""Line and circle intersection calculations."""

import math
from typing import List, Tuple, Optional
from .shapes import Point, Line, Circle


def find_line_intersection(line1: Line, line2: Line) -> Optional[Point]:
    """Find intersection point of two lines."""
    # Handle vertical lines
    if line1.slope == float('inf') and line2.slope == float('inf'):
        return None  # Parallel vertical lines
    
    if line1.slope == float('inf'):
        # line1 is vertical: x = line1.point1.x
        x = line1.point1.x
        y = line2.evaluate(x)
        return Point(x, y)
    
    if line2.slope == float('inf'):
        # line2 is vertical: x = line2.point1.x
        x = line2.point1.x
        y = line1.evaluate(x)
        return Point(x, y)
    
    # Check if parallel (same slope)
    if abs(line1.slope - line2.slope) < 1e-10:
        return None  # Parallel lines
    
    # Solve: m1*x + b1 = m2*x + b2
    x = (line2.intercept - line1.intercept) / (line1.slope - line2.slope)
    y = line1.evaluate(x)
    return Point(x, y)


def line_collision_detection(line1: Line, line2: Line) -> Tuple[bool, Optional[Point]]:
    """Check if lines collide and return intersection."""
    intersection = find_line_intersection(line1, line2)
    
    if intersection is None:
        return (False, None)
    
    # Check if intersection is within segment bounds
    def in_segment(p: Point, l: Line) -> bool:
        min_x = min(l.point1.x, l.point2.x)
        max_x = max(l.point1.x, l.point2.x)
        min_y = min(l.point1.y, l.point2.y)
        max_y = max(l.point1.y, l.point2.y)
        return min_x <= p.x <= max_x and min_y <= p.y <= max_y
    
    if in_segment(intersection, line1) and in_segment(intersection, line2):
        return (True, intersection)
    
    return (False, intersection)


def circle_line_intersection(circle: Circle, line: Line) -> List[Point]:
    """Find intersection points between circle and line."""
    # Distance from center to line
    if line.slope == float('inf'):  # Vertical line
        x = line.point1.x
        d = abs(x - circle.center.x)
    else:
        # Distance = |ax + by + c| / sqrt(a² + b²)
        a = line.slope
        b = -1
        c = line.intercept
        numerator = abs(a * circle.center.x + b * circle.center.y + c)
        denominator = math.sqrt(a**2 + b**2)
        d = numerator / denominator
    
    # No intersection if distance > radius
    if d > circle.radius + 1e-10:
        return []
    
    # Find closest point on line to center
    if line.slope == float('inf'):
        closest = Point(line.point1.x, circle.center.y)
    else:
        perp_slope = -1 / line.slope if line.slope != 0 else float('inf')
        if perp_slope == float('inf'):
            closest_x = circle.center.x
            closest_y = line.evaluate(closest_x)
        else:
            perp_intercept = circle.center.y - perp_slope * circle.center.x
            closest_x = (perp_intercept - line.intercept) / (line.slope - perp_slope)
            closest_y = line.evaluate(closest_x)
        closest = Point(closest_x, closest_y)
    
    # Distance from closest point to center
    dist_to_center = circle.center.distance_to(closest)
    
    if abs(dist_to_center - circle.radius) < 1e-10:
        # One intersection (tangent)
        return [closest]
    
    # Two intersections
    if line.slope == float('inf'):
        # Vertical line
        chord_half = math.sqrt(circle.radius**2 - d**2)
        return [
            Point(line.point1.x, circle.center.y - chord_half),
            Point(line.point1.x, circle.center.y + chord_half)
        ]
    else:
        # General line
        direction = Vector(1, line.slope) if line.slope != 0 else Vector(1, 0)
        direction = direction.normalize()
        
        chord_half = math.sqrt(circle.radius**2 - d**2)
        
        p1 = Point(
            closest.x - direction.x * chord_half,
            closest.y - direction.y * chord_half
        )
        p2 = Point(
            closest.x + direction.x * chord_half,
            closest.y + direction.y * chord_half
        )
        return [p1, p2]


def circle_circle_intersection(circle1: Circle, circle2: Circle) -> List[Point]:
    """Find intersection points between two circles."""
    d = circle1.center.distance_to(circle2.center)
    r1 = circle1.radius
    r2 = circle2.radius
    
    # No intersection
    if d > r1 + r2 + 1e-10 or d < abs(r1 - r2) - 1e-10:
        return []
    
    # One intersection (tangent)
    if abs(d - (r1 + r2)) < 1e-10 or abs(d - abs(r1 - r2)) < 1e-10:
        # Find tangent point
        if d == 0:
            return []  # Concentric circles
        t = r1 / d
        point = Point(
            circle1.center.x + t * (circle2.center.x - circle1.center.x),
            circle1.center.y + t * (circle2.center.y - circle1.center.y)
        )
        return [point]
    
    # Two intersections
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h = math.sqrt(r1**2 - a**2)
    
    px = circle1.center.x + a * (circle2.center.x - circle1.center.x) / d
    py = circle1.center.y + a * (circle2.center.y - circle1.center.y) / d
    
    intersection_base_x = px + h * (circle2.center.y - circle1.center.y) / d
    intersection_base_y = py - h * (circle2.center.x - circle1.center.x) / d
    
    intersection_2_x = px - h * (circle2.center.y - circle1.center.y) / d
    intersection_2_y = py + h * (circle2.center.x - circle1.center.x) / d
    
    return [
        Point(intersection_base_x, intersection_base_y),
        Point(intersection_2_x, intersection_2_y)
    ]


# Helper class for vector operations
class Vector:
    """Simple vector for calculations."""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2)
    
    def normalize(self) -> 'Vector':
        mag = self.magnitude()
        if mag == 0:
            return Vector(0, 0)
        return Vector(self.x / mag, self.y / mag)
