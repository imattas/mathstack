#!/usr/bin/env python
"""Demonstrate mathstack geometry functionality."""

from mathcore.core import geometry

print("=" * 70)
print("📐 MATHSTACK GEOMETRY DEMO")
print("=" * 70)

# Example 1: Points and distances
print("\n📍 Example 1: Points and Distance")
print("-" * 70)
p1 = geometry.Point(0, 0)
p2 = geometry.Point(3, 4)
print(f"Point 1: {p1}")
print(f"Point 2: {p2}")
distance = geometry.distance(p1, p2)
print(f"Distance: {distance}")
print(f"Expected: 5 (3-4-5 right triangle)")

# Example 2: Vectors
print("\n➡️  Example 2: Vector Operations")
print("-" * 70)
v1 = geometry.Vector(1, 2)
v2 = geometry.Vector(3, 4)
print(f"Vector 1: {v1}")
print(f"Vector 2: {v2}")
# Vectors have magnitude/length
mag1 = (v1.x**2 + v1.y**2)**0.5
mag2 = (v2.x**2 + v2.y**2)**0.5
print(f"Magnitude of V1: {mag1:.4f}")
print(f"Magnitude of V2: {mag2:.4f}")

# Example 3: Lines
print("\n📏 Example 3: Line Operations")
print("-" * 70)
line1 = geometry.Line(0, 0, 2, 4)  # from (0,0) to (2,4)
print(f"Line: {line1}")
print(f"Slope: {line1.slope}")
print(f"y-intercept: {line1.intercept}")
# Calculate length using distance between endpoints
length = geometry.distance(line1.point1, line1.point2)
print(f"Length: {length:.4f}")

# Example 4: Circles
print("\n⭕ Example 4: Circle Properties")
print("-" * 70)
center = geometry.Point(0, 0)
circle = geometry.Circle(center, 5)  # center Point(0,0), radius 5
print(f"Circle: {circle}")
print(f"Radius: {circle.radius}")
print(f"Area: {circle.area():.4f}")
print(f"Circumference: {circle.circumference():.4f}")
print(f"Expected area: {5**2 * 3.14159:.4f}")

# Example 5: Triangles
print("\n🔺 Example 5: Triangle Properties")
print("-" * 70)
p1 = geometry.Point(0, 0)
p2 = geometry.Point(4, 0)
p3 = geometry.Point(2, 3)
triangle = geometry.Triangle(p1, p2, p3)
print(f"Triangle: {triangle}")
print(f"Perimeter: {triangle.perimeter():.4f}")
print(f"Area: {triangle.area():.4f}")

# Example 6: Line intersection
print("\n✖️  Example 6: Line Intersection")
print("-" * 70)
line1 = geometry.Line(0, 0, 2, 2)  # y = x
line2 = geometry.Line(0, 2, 2, 0)  # y = -x + 2
intersection = geometry.find_line_intersection(line1, line2)
print(f"Line 1: from (0,0) to (2,2)")
print(f"Line 2: from (0,2) to (2,0)")
if intersection:
    print(f"Intersection point: {intersection}")
    print(f"Expected: (1, 1)")
else:
    print("Lines are parallel")

# Example 7: Midpoint
print("\n🎯 Example 7: Midpoint Calculation")
print("-" * 70)
p1 = geometry.Point(0, 0)
p2 = geometry.Point(10, 4)
midpoint = geometry.midpoint(p1, p2)
print(f"Point 1: {p1}")
print(f"Point 2: {p2}")
print(f"Midpoint: {midpoint}")
print(f"Expected: (5, 2)")

# Example 8: Right angle check
print("\n∟ Example 8: Perpendicular Lines")
print("-" * 70)
v1 = geometry.Vector(3, 4)
v2 = geometry.Vector(-4, 3)
dot = v1.x * v2.x + v1.y * v2.y  # manual dot product
print(f"Vector 1: {v1}")
print(f"Vector 2: {v2}")
print(f"Dot product: {dot}")
if dot == 0:
    print("✓ Vectors are perpendicular")
else:
    print(f"Dot product ≠ 0, so not perpendicular")

# Example 9: Distance point to line
print("\n📌 Example 9: Point to Line Distance")
print("-" * 70)
point = geometry.Point(0, 0)
line = geometry.Line(1, 0, 1, 1)  # vertical line at x=1
dist = geometry.point_to_line_distance(point, line)
print(f"Point: {point}")
print(f"Line: x=1")
print(f"Distance: {dist:.4f}")
print(f"Expected: 1.0")

# Example 10: Coordinate geometry batch
print("\n📊 Example 10: Batch Geometry Calculations")
print("-" * 70)
coordinates = [
    ((0, 0), (3, 4), "3-4-5 triangle"),
    ((1, 1), (4, 5), "Arbitrary points"),
    ((0, 0), (5, 12), "5-12-13 triangle"),
]

for (x1, y1), (x2, y2), description in coordinates:
    p1 = geometry.Point(x1, y1)
    p2 = geometry.Point(x2, y2)
    dist = geometry.distance(p1, p2)
    mid = geometry.midpoint(p1, p2)
    print(f"{description:20} | Distance: {dist:6.2f} | Midpoint: {mid}")

print("\n" + "=" * 70)
print("✅ Geometry demo complete!")
print("=" * 70)
