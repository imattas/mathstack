"""
Advanced geometry module with support for coordinate plane operations,
geometric shapes, intersections, and distance calculations.
"""

import math
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass


@dataclass
class Point:
    """Represents a point in 2D coordinate plane."""
    x: float
    y: float
    
    def __repr__(self):
        return f"Point({self.x}, {self.y})"
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def distance_to(self, other: 'Point') -> float:
        """Calculate distance to another point."""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    def translate(self, dx: float, dy: float) -> 'Point':
        """Translate point by (dx, dy)."""
        return Point(self.x + dx, self.y + dy)
    
    def rotate(self, angle: float, center: 'Point' = None) -> 'Point':
        """Rotate point by angle (in radians) around center point."""
        if center is None:
            center = Point(0, 0)
        
        # Translate to origin
        x = self.x - center.x
        y = self.y - center.y
        
        # Rotate
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        new_x = x * cos_a - y * sin_a
        new_y = x * sin_a + y * cos_a
        
        # Translate back
        return Point(new_x + center.x, new_y + center.y)


class Line:
    """Represents a line in 2D coordinate plane.
    
    A line can be defined by:
    1. Two points
    2. Slope and y-intercept (y = mx + b)
    3. General form (ax + by + c = 0)
    """
    
    def __init__(self, point1: Point = None, point2: Point = None, 
                 slope: float = None, intercept: float = None,
                 a: float = None, b: float = None, c: float = None):
        """Initialize a line.
        
        Args:
            point1, point2: Two points on the line
            slope, intercept: m and b from y = mx + b
            a, b, c: Coefficients from ax + by + c = 0
        """
        if point1 and point2:
            self.point1 = point1
            self.point2 = point2
            self.slope = self._calculate_slope(point1, point2)
            self.intercept = point1.y - self.slope * point1.x if self.slope != float('inf') else None
        elif slope is not None and intercept is not None:
            self.slope = slope
            self.intercept = intercept
            self.point1 = Point(0, intercept)
            self.point2 = Point(1, slope + intercept)
        elif a is not None and b is not None:
            if b != 0:
                self.slope = -a / b
                self.intercept = -c / b
                self.point1 = Point(0, self.intercept)
                self.point2 = Point(1, self.slope + self.intercept)
            else:
                # Vertical line: x = -c/a
                self.slope = float('inf')
                self.intercept = None
                x_val = -c / a
                self.point1 = Point(x_val, 0)
                self.point2 = Point(x_val, 1)
    
    @staticmethod
    def _calculate_slope(p1: Point, p2: Point) -> float:
        """Calculate slope between two points."""
        if p1.x == p2.x:
            return float('inf')  # Vertical line
        return (p2.y - p1.y) / (p2.x - p1.x)
    
    def __repr__(self):
        if self.slope == float('inf'):
            return f"Line(x = {self.point1.x})"
        return f"Line(y = {self.slope}x + {self.intercept})"
    
    def evaluate(self, x: float) -> float:
        """Get y value for given x."""
        if self.slope == float('inf'):
            raise ValueError("Vertical line - cannot evaluate y for given x")
        return self.slope * x + self.intercept
    
    def contains_point(self, point: Point) -> bool:
        """Check if point lies on the line."""
        if self.slope == float('inf'):
            return point.x == self.point1.x
        return abs(point.y - self.evaluate(point.x)) < 1e-10


class Circle:
    """Represents a circle in 2D coordinate plane."""
    
    def __init__(self, center: Point, radius: float):
        """Initialize circle with center and radius."""
        if radius < 0:
            raise ValueError("Radius must be non-negative")
        self.center = center
        self.radius = radius
    
    def __repr__(self):
        return f"Circle(center={self.center}, radius={self.radius})"
    
    def area(self) -> float:
        """Calculate area of circle."""
        return math.pi * self.radius ** 2
    
    def circumference(self) -> float:
        """Calculate circumference of circle."""
        return 2 * math.pi * self.radius
    
    def contains_point(self, point: Point) -> bool:
        """Check if point is inside or on the circle."""
        return self.center.distance_to(point) <= self.radius
    
    def point_on_circle(self, angle: float) -> Point:
        """Get point on circle at given angle (in radians)."""
        x = self.center.x + self.radius * math.cos(angle)
        y = self.center.y + self.radius * math.sin(angle)
        return Point(x, y)


class Triangle:
    """Represents a triangle in 2D coordinate plane."""
    
    def __init__(self, p1: Point, p2: Point, p3: Point):
        """Initialize triangle with three points."""
        self.vertices = [p1, p2, p3]
    
    def __repr__(self):
        return f"Triangle({self.vertices[0]}, {self.vertices[1]}, {self.vertices[2]})"
    
    def side_lengths(self) -> Tuple[float, float, float]:
        """Calculate lengths of all three sides."""
        a = self.vertices[1].distance_to(self.vertices[2])
        b = self.vertices[0].distance_to(self.vertices[2])
        c = self.vertices[0].distance_to(self.vertices[1])
        return (a, b, c)
    
    def perimeter(self) -> float:
        """Calculate perimeter of triangle."""
        return sum(self.side_lengths())
    
    def area(self) -> float:
        """Calculate area using Heron's formula."""
        a, b, c = self.side_lengths()
        s = (a + b + c) / 2
        area_squared = s * (s - a) * (s - b) * (s - c)
        if area_squared < 0:
            return 0
        return math.sqrt(area_squared)
    
    def centroid(self) -> Point:
        """Calculate centroid (center of mass)."""
        x = (self.vertices[0].x + self.vertices[1].x + self.vertices[2].x) / 3
        y = (self.vertices[0].y + self.vertices[1].y + self.vertices[2].y) / 3
        return Point(x, y)
    
    def is_valid(self) -> bool:
        """Check if triangle inequality holds."""
        a, b, c = self.side_lengths()
        return a + b > c and b + c > a and c + a > b


class Vector:
    """Represents a 2D vector."""
    
    def __init__(self, x: float, y: float):
        """Initialize vector with components."""
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"
    
    def magnitude(self) -> float:
        """Calculate magnitude (length) of vector."""
        return math.sqrt(self.x ** 2 + self.y ** 2)
    
    def normalize(self) -> 'Vector':
        """Return normalized (unit) vector."""
        mag = self.magnitude()
        if mag == 0:
            return Vector(0, 0)
        return Vector(self.x / mag, self.y / mag)
    
    def dot_product(self, other: 'Vector') -> float:
        """Calculate dot product with another vector."""
        return self.x * other.x + self.y * other.y
    
    def cross_product_magnitude(self, other: 'Vector') -> float:
        """Calculate magnitude of cross product (2D)."""
        return self.x * other.y - self.y * other.x
    
    def angle_between(self, other: 'Vector') -> float:
        """Calculate angle between two vectors (in radians)."""
        dot = self.dot_product(other)
        cross = self.cross_product_magnitude(other)
        return math.atan2(cross, dot)
    
    def __add__(self, other: 'Vector') -> 'Vector':
        """Add two vectors."""
        return Vector(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: 'Vector') -> 'Vector':
        """Subtract two vectors."""
        return Vector(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: float) -> 'Vector':
        """Multiply vector by scalar."""
        return Vector(self.x * scalar, self.y * scalar)


# Geometry utility functions

def distance(p1: Point, p2: Point) -> float:
    """Calculate Euclidean distance between two points."""
    return p1.distance_to(p2)


def midpoint(p1: Point, p2: Point) -> Point:
    """Find midpoint between two points."""
    return Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)


def slope(p1: Point, p2: Point) -> float:
    """Calculate slope between two points."""
    if p1.x == p2.x:
        return float('inf')
    return (p2.y - p1.y) / (p2.x - p1.x)


def find_line_intersection(line1: Line, line2: Line) -> Optional[Point]:
    """Find intersection point of two lines.
    
    Args:
        line1, line2: Two Line objects
        
    Returns:
        Point of intersection, or None if lines are parallel
    """
    # Handle vertical lines
    if line1.slope == float('inf') and line2.slope == float('inf'):
        return None  # Parallel vertical lines
    
    if line1.slope == float('inf'):
        x = line1.point1.x
        y = line2.evaluate(x)
        return Point(x, y)
    
    if line2.slope == float('inf'):
        x = line2.point1.x
        y = line1.evaluate(x)
        return Point(x, y)
    
    # Check if parallel (same slope)
    if abs(line1.slope - line2.slope) < 1e-10:
        return None  # Parallel non-vertical lines
    
    # Solve: m1*x + b1 = m2*x + b2
    x = (line2.intercept - line1.intercept) / (line1.slope - line2.slope)
    y = line1.evaluate(x)
    
    return Point(x, y)


def line_collision_detection(line1: Line, line2: Line) -> Dict:
    """Advanced collision detection between two lines.
    
    Returns:
        Dict with intersection info and collision details
    """
    intersection = find_line_intersection(line1, line2)
    
    result = {
        'intersects': intersection is not None,
        'parallel': line1.slope == line2.slope,
        'coincident': line1.slope == line2.slope and line1.intercept == line2.intercept,
        'intersection_point': intersection
    }
    
    return result


def circle_line_intersection(circle: Circle, line: Line) -> List[Point]:
    """Find intersection points between a circle and a line."""
    intersections = []
    
    # Handle vertical line
    if line.slope == float('inf'):
        x = line.point1.x
        # Solve (x - cx)^2 + (y - cy)^2 = r^2
        dx = x - circle.center.x
        if abs(dx) <= circle.radius:
            dy_sq = circle.radius ** 2 - dx ** 2
            dy = math.sqrt(dy_sq)
            intersections.append(Point(x, circle.center.y + dy))
            if dy != 0:
                intersections.append(Point(x, circle.center.y - dy))
    else:
        # y = mx + b, solve (x - cx)^2 + (mx + b - cy)^2 = r^2
        m = line.slope
        b = line.intercept
        cx, cy, r = circle.center.x, circle.center.y, circle.radius
        
        # Expand: (1 + m^2)x^2 + 2(mb - mcy - cx)x + (b - cy)^2 + cx^2 - r^2 = 0
        a = 1 + m ** 2
        b_coeff = 2 * (m * b - m * cy - cx)
        c_coeff = (b - cy) ** 2 + cx ** 2 - r ** 2
        
        discriminant = b_coeff ** 2 - 4 * a * c_coeff
        
        if discriminant >= 0:
            sqrt_disc = math.sqrt(discriminant)
            x1 = (-b_coeff + sqrt_disc) / (2 * a)
            x2 = (-b_coeff - sqrt_disc) / (2 * a)
            
            intersections.append(Point(x1, line.evaluate(x1)))
            if discriminant > 0:
                intersections.append(Point(x2, line.evaluate(x2)))
    
    return intersections


def point_to_line_distance(point: Point, line: Line) -> float:
    """Calculate perpendicular distance from point to line."""
    if line.slope == float('inf'):
        return abs(point.x - line.point1.x)
    
    # Distance = |ax + by + c| / sqrt(a^2 + b^2)
    # Convert from y = mx + b to mx - y + b = 0
    # a = m, b = -1, c = b
    m = line.slope
    b = line.intercept
    numerator = abs(m * point.x - point.y + b)
    denominator = math.sqrt(m ** 2 + 1)
    return numerator / denominator
