"""Geometric shapes: Point, Vector, Line, Circle, Triangle, Polygon, Rectangle."""

import math
from typing import List, Tuple, Optional
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
        """Rotate point by angle (radians) around center."""
        if center is None:
            center = Point(0, 0)
        x = self.x - center.x
        y = self.y - center.y
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        new_x = x * cos_a - y * sin_a
        new_y = x * sin_a + y * cos_a
        return Point(new_x + center.x, new_y + center.y)


@dataclass
class Vector:
    """Represents a 2D vector."""
    x: float
    y: float
    
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"
    
    def magnitude(self) -> float:
        """Calculate magnitude (length) of vector."""
        return math.sqrt(self.x ** 2 + self.y ** 2)
    
    def dot_product(self, other: 'Vector') -> float:
        """Calculate dot product with another vector."""
        return self.x * other.x + self.y * other.y
    
    def cross_product(self, other: 'Vector') -> float:
        """Calculate 2D cross product (scalar result)."""
        return self.x * other.y - self.y * other.x
    
    def normalize(self) -> 'Vector':
        """Return unit vector."""
        mag = self.magnitude()
        if mag == 0:
            return Vector(0, 0)
        return Vector(self.x / mag, self.y / mag)
    
    def add(self, other: 'Vector') -> 'Vector':
        """Add two vectors."""
        return Vector(self.x + other.x, self.y + other.y)
    
    def subtract(self, other: 'Vector') -> 'Vector':
        """Subtract two vectors."""
        return Vector(self.x - other.x, self.y - other.y)
    
    def scale(self, scalar: float) -> 'Vector':
        """Scale vector by scalar."""
        return Vector(self.x * scalar, self.y * scalar)


class Line:
    """Represents a line in 2D coordinate plane."""
    
    def __init__(self, point1: Point, point2: Point):
        """Initialize line from two points."""
        self.point1 = point1
        self.point2 = point2
        self.slope = self._calculate_slope()
        self.intercept = self._calculate_intercept()
    
    def _calculate_slope(self) -> float:
        """Calculate slope between two points."""
        if self.point1.x == self.point2.x:
            return float('inf')  # Vertical line
        return (self.point2.y - self.point1.y) / (self.point2.x - self.point1.x)
    
    def _calculate_intercept(self) -> Optional[float]:
        """Calculate y-intercept."""
        if self.slope == float('inf'):
            return None
        return self.point1.y - self.slope * self.point1.x
    
    def evaluate(self, x: float) -> float:
        """Evaluate y value at given x."""
        if self.slope == float('inf'):
            raise ValueError("Cannot evaluate vertical line")
        return self.slope * x + self.intercept
    
    def contains_point(self, point: Point) -> bool:
        """Check if point lies on line."""
        if self.slope == float('inf'):
            return abs(point.x - self.point1.x) < 1e-10
        return abs(point.y - self.evaluate(point.x)) < 1e-10
    
    def __repr__(self):
        if self.slope == float('inf'):
            return f"Line(x = {self.point1.x})"
        return f"Line(y = {self.slope}x + {self.intercept})"


class Circle:
    """Represents a circle in 2D coordinate plane."""
    
    def __init__(self, center: Point, radius: float):
        """Initialize circle with center and radius."""
        self.center = center
        self.radius = radius
    
    def area(self) -> float:
        """Calculate area of circle."""
        return math.pi * self.radius ** 2
    
    def circumference(self) -> float:
        """Calculate circumference of circle."""
        return 2 * math.pi * self.radius
    
    def contains_point(self, point: Point) -> bool:
        """Check if point is inside circle."""
        return self.center.distance_to(point) <= self.radius
    
    def point_on_circle(self, point: Point) -> bool:
        """Check if point is on circle boundary."""
        dist = self.center.distance_to(point)
        return abs(dist - self.radius) < 1e-10
    
    def __repr__(self):
        return f"Circle(center={self.center}, radius={self.radius})"


class Triangle:
    """Represents a triangle in 2D coordinate plane."""
    
    def __init__(self, p1: Point, p2: Point, p3: Point):
        """Initialize triangle from three points."""
        self.vertices = [p1, p2, p3]
        self.p1, self.p2, self.p3 = p1, p2, p3
    
    def side_lengths(self) -> Tuple[float, float, float]:
        """Calculate lengths of all sides."""
        a = self.p1.distance_to(self.p2)
        b = self.p2.distance_to(self.p3)
        c = self.p3.distance_to(self.p1)
        return (a, b, c)
    
    def perimeter(self) -> float:
        """Calculate perimeter of triangle."""
        a, b, c = self.side_lengths()
        return a + b + c
    
    def area(self) -> float:
        """Calculate area using cross product."""
        v1 = Vector(self.p2.x - self.p1.x, self.p2.y - self.p1.y)
        v2 = Vector(self.p3.x - self.p1.x, self.p3.y - self.p1.y)
        return abs(v1.cross_product(v2)) / 2
    
    def centroid(self) -> Point:
        """Calculate centroid of triangle."""
        x = (self.p1.x + self.p2.x + self.p3.x) / 3
        y = (self.p1.y + self.p2.y + self.p3.y) / 3
        return Point(x, y)
    
    def is_valid(self) -> bool:
        """Check if triangle is valid (non-zero area)."""
        return self.area() > 1e-10
    
    def __repr__(self):
        return f"Triangle({self.p1}, {self.p2}, {self.p3})"


class Polygon:
    """Represents a polygon in 2D coordinate plane."""
    
    def __init__(self, vertices: List[Point]):
        """Initialize polygon from list of vertices."""
        if len(vertices) < 3:
            raise ValueError("Polygon must have at least 3 vertices")
        self.vertices = vertices
    
    def side_lengths(self) -> List[float]:
        """Calculate lengths of all sides."""
        lengths = []
        for i in range(len(self.vertices)):
            p1 = self.vertices[i]
            p2 = self.vertices[(i + 1) % len(self.vertices)]
            lengths.append(p1.distance_to(p2))
        return lengths
    
    def perimeter(self) -> float:
        """Calculate perimeter of polygon."""
        return sum(self.side_lengths())
    
    def area(self) -> float:
        """Calculate area using shoelace formula."""
        n = len(self.vertices)
        area = 0
        for i in range(n):
            x1, y1 = self.vertices[i].x, self.vertices[i].y
            x2, y2 = self.vertices[(i + 1) % n].x, self.vertices[(i + 1) % n].y
            area += x1 * y2 - x2 * y1
        return abs(area) / 2
    
    def centroid(self) -> Point:
        """Calculate centroid of polygon."""
        x = sum(v.x for v in self.vertices) / len(self.vertices)
        y = sum(v.y for v in self.vertices) / len(self.vertices)
        return Point(x, y)
    
    def __repr__(self):
        return f"Polygon({len(self.vertices)} vertices)"


class Rectangle:
    """Represents a rectangle in 2D coordinate plane."""
    
    def __init__(self, x: float, y: float, width: float, height: float):
        """Initialize rectangle at (x,y) with given width and height."""
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    def area(self) -> float:
        """Calculate area of rectangle."""
        return self.width * self.height
    
    def perimeter(self) -> float:
        """Calculate perimeter of rectangle."""
        return 2 * (self.width + self.height)
    
    def vertices(self) -> List[Point]:
        """Get all four vertices."""
        return [
            Point(self.x, self.y),
            Point(self.x + self.width, self.y),
            Point(self.x + self.width, self.y + self.height),
            Point(self.x, self.y + self.height),
        ]
    
    def center(self) -> Point:
        """Get center point."""
        return Point(self.x + self.width / 2, self.y + self.height / 2)
    
    def __repr__(self):
        return f"Rectangle(x={self.x}, y={self.y}, w={self.width}, h={self.height})"
