#!/usr/bin/env python
"""Demonstrate mathstack quadratic functionality."""

import mathcore
from mathcore.core import algebra, arithmetic

print("=" * 60)
print("🎯 MATHSTACK QUADRATICS DEMO")
print("=" * 60)

# Example 1: Solve quadratic equation ax² + bx + c = 0
print("\n📐 Example 1: Solve x² - 5x + 6 = 0")
print("-" * 60)
solutions = algebra.solve_quadratic(1, -5, 6)
print(f"Solutions: {solutions}")
print(f"Expected: (2, 3)")

# Example 2: Quadratic with different coefficients
print("\n📐 Example 2: Solve 2x² + 3x - 2 = 0")
print("-" * 60)
solutions = algebra.solve_quadratic(2, 3, -2)
print(f"Solutions: {solutions}")

# Example 3: Quadratic with complex roots
print("\n📐 Example 3: Solve x² + 1 = 0 (complex roots)")
print("-" * 60)
solutions = algebra.solve_quadratic(1, 0, 1)
print(f"Solutions: {solutions}")

# Example 4: Quadratic formula showcase
print("\n📐 Example 4: Quadratic Formula: x² + 2x - 3 = 0")
print("-" * 60)
a, b, c = 1, 2, -3
discriminant = b**2 - 4*a*c
print(f"a = {a}, b = {b}, c = {c}")
print(f"Discriminant (b² - 4ac) = {discriminant}")
solutions = algebra.solve_quadratic(a, b, c)
print(f"Solutions: {solutions}")

# Example 5: Complete the square visualization
print("\n📐 Example 5: Factoring Quadratic x² + 5x + 6")
print("-" * 60)
poly = algebra.Polynomial({0: 6, 1: 5, 2: 1})  # {power: coefficient}
print(f"Polynomial: {poly}")
roots = algebra.solve_quadratic(1, 5, 6)
print(f"Roots: {roots}")
print(f"Factored form: (x - {roots[0]})(x - {roots[1]})")

# Example 6: Vertex form calculation
print("\n📐 Example 6: Vertex of Parabola y = x² - 4x + 3")
print("-" * 60)
a, b, c = 1, -4, 3
x_vertex = -b / (2 * a)
y_vertex = a * x_vertex**2 + b * x_vertex + c
print(f"Vertex coordinates: ({x_vertex}, {y_vertex})")
print(f"Axis of symmetry: x = {x_vertex}")

# Example 7: Multiple quadratics
print("\n📐 Example 7: Batch Quadratic Solutions")
print("-" * 60)
test_cases = [
    (1, -6, 8, "x² - 6x + 8 = 0"),
    (1, 0, -4, "x² - 4 = 0"),
    (2, -4, 2, "2x² - 4x + 2 = 0"),
]

for a, b, c, equation in test_cases:
    solutions = algebra.solve_quadratic(a, b, c)
    print(f"{equation:20} → x = {solutions}")

print("\n" + "=" * 60)
print("✅ Quadratics demo complete!")
print("=" * 60)
