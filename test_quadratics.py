#!/usr/bin/env python
"""Test the interactive quadratic solver with automated inputs."""

import sys
from mathcore.core import algebra

print("\n" + "="*70)
print("🧮 AUTOMATED QUADRATIC SOLVER TEST")
print("="*70)

test_cases = [
    (1, -5, 6, "x² - 5x + 6 = 0"),
    (1, 0, -9, "x² - 9 = 0"),
    (2, -8, 6, "2x² - 8x + 6 = 0"),
    (1, 2, 5, "x² + 2x + 5 = 0 (complex)"),
    (3, -12, 12, "3x² - 12x + 12 = 0"),
    (1, -7, 12, "x² - 7x + 12 = 0"),
]

for a, b, c, description in test_cases:
    print(f"\n📊 {description}")
    
    # Calculate discriminant
    discriminant = b**2 - 4*a*c
    
    # Solve
    solutions = algebra.solve_quadratic(a, b, c)
    
    print(f"   Discriminant: {discriminant}")
    
    if isinstance(solutions[0], complex):
        print(f"   Complex roots: {solutions[0]}, {solutions[1]}")
    else:
        print(f"   Real roots: x₁ = {solutions[0]}, x₂ = {solutions[1]}")
        # Verify
        check1 = a * solutions[0]**2 + b * solutions[0] + c
        check2 = a * solutions[1]**2 + b * solutions[1] + c
        print(f"   Verification: f(x₁) = {abs(check1):.2e}, f(x₂) = {abs(check2):.2e}")

print("\n" + "="*70)
print("✅ All tests complete!")
print("="*70 + "\n")
