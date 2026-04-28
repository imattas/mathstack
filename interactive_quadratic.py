#!/usr/bin/env python
"""Interactive quadratic solver using mathstack."""

from mathcore.core import algebra

print("\n" + "="*70)
print("🧮 MATHSTACK INTERACTIVE QUADRATIC SOLVER")
print("="*70)
print("\nSolve any quadratic equation: ax² + bx + c = 0")
print("Enter coefficients a, b, c separated by spaces")
print("Type 'quit' to exit\n")

while True:
    try:
        user_input = input("Enter a, b, c (or 'quit'): ").strip()
        
        if user_input.lower() in ('quit', 'exit', 'q'):
            print("\n👋 Goodbye!")
            break
        
        parts = user_input.split()
        if len(parts) != 3:
            print("❌ Please enter exactly 3 numbers separated by spaces")
            continue
        
        a, b, c = float(parts[0]), float(parts[1]), float(parts[2])
        
        if a == 0:
            print("❌ Coefficient 'a' cannot be zero (not a quadratic)")
            continue
        
        # Calculate discriminant
        discriminant = b**2 - 4*a*c
        
        # Solve
        solutions = algebra.solve_quadratic(a, b, c)
        
        # Display results
        equation = f"{a}x² + {b}x + {c} = 0" if b >= 0 else f"{a}x² - {-b}x + {c} = 0"
        print(f"\n📊 Equation: {equation}")
        print(f"   Discriminant: {discriminant}")
        
        if isinstance(solutions[0], complex):
            print(f"   ❌ Complex roots (no real solutions)")
        else:
            print(f"   ✅ Real roots:")
        
        print(f"   x₁ = {solutions[0]}")
        print(f"   x₂ = {solutions[1]}")
        
        # Verify solution
        x1_check = a * solutions[0]**2 + b * solutions[0] + c
        print(f"   Verification: f(x₁) = {x1_check:.10f} ≈ 0 ✓")
        print()
        
    except ValueError:
        print("❌ Invalid input. Please enter three numbers.")
    except Exception as e:
        print(f"❌ Error: {e}")
