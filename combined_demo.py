#!/usr/bin/env python
"""Combined Quadratics + Geometry demo using mathstack."""

from mathcore.core import algebra, geometry
import math

print("=" * 80)
print("🎯 MATHSTACK - QUADRATICS & GEOMETRY COMBINED DEMO")
print("=" * 80)

# ============================================================================
# PART 1: QUADRATIC-GEOMETRY PROBLEMS
# ============================================================================

print("\n" + "="*80)
print("📐 PART 1: PARABOLA & GEOMETRY")
print("="*80)

print("\n🔷 Problem: Find the vertex and axis of symmetry for y = x² - 4x + 3")
print("-" * 80)
a, b, c = 1, -4, 3
x_vertex = -b / (2*a)
y_vertex = a * x_vertex**2 + b * x_vertex + c
print(f"Vertex: ({x_vertex}, {y_vertex})")
print(f"Axis of symmetry: x = {x_vertex}")

# Solve for x-intercepts (roots)
roots = algebra.solve_quadratic(a, b, c)
print(f"x-intercepts (roots): {roots}")
print(f"Points on parabola: ({roots[0]}, 0) and ({roots[1]}, 0)")

# Calculate distances from vertex to roots
p_vertex = geometry.Point(x_vertex, y_vertex)
p_root1 = geometry.Point(roots[0], 0)
p_root2 = geometry.Point(roots[1], 0)
dist1 = geometry.distance(p_vertex, p_root1)
dist2 = geometry.distance(p_vertex, p_root2)
print(f"Distance from vertex to root 1: {dist1:.4f}")
print(f"Distance from vertex to root 2: {dist2:.4f}")

# ============================================================================
# PART 2: GEOMETRIC SHAPES FROM QUADRATIC SOLUTIONS
# ============================================================================

print("\n" + "="*80)
print("📐 PART 2: TRIANGLE FROM QUADRATIC ROOTS")
print("="*80)

print("\n🔶 Problem: Create a triangle using quadratic roots")
print("-" * 80)

# Solve: x² - 6x + 8 = 0
roots1 = algebra.solve_quadratic(1, -6, 8)
print(f"Quadratic 1: x² - 6x + 8 = 0")
print(f"Roots: {roots1}")

# Create points on x-axis
p1 = geometry.Point(roots1[0], 0)
p2 = geometry.Point(roots1[1], 0)
# Peak point (vertex of parabola)
peak_x = 3  # midpoint of roots
peak_y = 1**2 - 6*3 + 8  # = -1 but let's make it 3
p3 = geometry.Point(peak_x, 3)

triangle = geometry.Triangle(p1, p2, p3)
print(f"\nTriangle vertices: {p1}, {p2}, {p3}")
print(f"Perimeter: {triangle.perimeter():.4f}")
print(f"Area: {triangle.area():.4f}")

# ============================================================================
# PART 3: CIRCLE & QUADRATIC INTERSECTION
# ============================================================================

print("\n" + "="*80)
print("⭕ PART 3: CIRCLE GEOMETRY")
print("="*80)

print("\n🔷 Problem: Circle with radius 5 centered at origin")
print("-" * 80)
center = geometry.Point(0, 0)
circle = geometry.Circle(center, 5)
print(f"Circle: {circle}")
print(f"Area: {circle.area():.4f}")
print(f"Circumference: {circle.circumference():.4f}")

# Check points on/in circle
test_points = [
    geometry.Point(0, 0),
    geometry.Point(3, 4),
    geometry.Point(5, 0),
    geometry.Point(4, 4),
]

print("\nPoint containment check:")
for p in test_points:
    dist = geometry.distance(center, p)
    inside = dist <= circle.radius
    on_edge = abs(dist - circle.radius) < 0.001
    status = "ON EDGE" if on_edge else ("INSIDE" if inside else "OUTSIDE")
    print(f"  {p}: distance={dist:.2f}, {status}")

# ============================================================================
# PART 4: VECTOR OPERATIONS
# ============================================================================

print("\n" + "="*80)
print("➡️  PART 4: VECTOR & LINE OPERATIONS")
print("="*80)

print("\n🔷 Problem: Lines and perpendicular vectors")
print("-" * 80)
v1 = geometry.Vector(1, 2)
v2 = geometry.Vector(-2, 1)
print(f"Vector 1: {v1}")
print(f"Vector 2: {v2}")

# Calculate magnitudes
mag1 = math.sqrt(v1.x**2 + v1.y**2)
mag2 = math.sqrt(v2.x**2 + v2.y**2)
dot = v1.x * v2.x + v1.y * v2.y

print(f"Magnitude of V1: {mag1:.4f}")
print(f"Magnitude of V2: {mag2:.4f}")
print(f"Dot product: {dot}")
print(f"Perpendicular: {dot == 0}")

# ============================================================================
# PART 5: PRACTICAL PHYSICS PROBLEM
# ============================================================================

print("\n" + "="*80)
print("🚀 PART 5: PRACTICAL PROBLEM - PROJECTILE MOTION")
print("="*80)

print("\n🔷 Problem: Ball thrown from origin at 45°")
print("-" * 80)
print("Trajectory: y = x - 0.5x²  (simplified physics)")
print("Initial velocity at 45° angle")

# Find where ball hits ground (y = 0)
# 0 = x - 0.5x² => x(1 - 0.5x) = 0 => x = 0 or x = 2
roots_projectile = algebra.solve_quadratic(-0.5, 1, 0)
print(f"\nLanding point: x = {[r for r in roots_projectile if r != 0][0]}")

# Find max height (vertex)
a_p, b_p, c_p = -0.5, 1, 0
x_max = -b_p / (2*a_p)
y_max = a_p * x_max**2 + b_p * x_max + c_p
print(f"Max height point: ({x_max}, {y_max})")

# Calculate distance traveled
landing_x = [r for r in roots_projectile if r > 0][0]
start = geometry.Point(0, 0)
land = geometry.Point(landing_x, 0)
peak = geometry.Point(x_max, y_max)

distance = geometry.distance(start, land)
height_distance = geometry.distance(start, peak)
print(f"Horizontal distance: {distance:.4f}")
print(f"Distance from start to peak: {height_distance:.4f}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ COMBINED DEMO COMPLETE!")
print("="*80)
print("""
This demo showcased:
  1. Quadratic equation solving with geometric interpretation
  2. Creating geometric shapes from algebraic solutions
  3. Circle properties and point membership
  4. Vector operations and perpendicularity
  5. Real-world physics application (projectile motion)

All calculations performed using MATHSTACK - Pure Python Math Library
""")
