"""Unit tests for algebra module."""

import pytest
from mathcore.core.algebra import (
    Polynomial, solve_quadratic, solve_cubic,
    simplify, factor_expression, expand_expression
)


class TestPolynomial:
    """Tests for Polynomial class."""
    
    def test_polynomial_creation(self):
        p = Polynomial({2: 1, 1: -5, 0: 6})
        assert p.coefficients == {2: 1, 1: -5, 0: 6}
    
    def test_polynomial_kwargs(self):
        p = Polynomial(x2=1, x1=-5, x0=6)
        assert p.degree() == 2
    
    def test_polynomial_addition(self):
        p1 = Polynomial({2: 1, 1: 2})
        p2 = Polynomial({2: 1, 1: 3})
        p3 = p1 + p2
        assert p3.coefficients == {2: 2, 1: 5}
    
    def test_polynomial_multiplication(self):
        p1 = Polynomial({1: 1})  # x
        p2 = Polynomial({1: 1})  # x
        p3 = p1 * p2
        assert p3.coefficients == {2: 1}  # x^2
    
    def test_polynomial_evaluate(self):
        p = Polynomial({2: 1, 0: 1})  # x^2 + 1
        assert p.evaluate(2) == 5
        assert p.evaluate(0) == 1
    
    def test_polynomial_derivative(self):
        p = Polynomial({3: 1, 2: 2, 1: 3})  # x^3 + 2x^2 + 3x
        deriv = p.derivative()
        # Derivative: 3x^2 + 4x + 3
        assert deriv.coefficients == {2: 3, 1: 4, 0: 3}


class TestEquationSolving:
    """Tests for equation solving functions."""
    
    def test_solve_quadratic(self):
        # x^2 - 5x + 6 = 0 -> x = 2, 3
        x1, x2 = solve_quadratic(1, -5, 6)
        assert abs(x1 - 3) < 1e-10 or abs(x1 - 2) < 1e-10
        assert abs(x2 - 3) < 1e-10 or abs(x2 - 2) < 1e-10
    
    def test_solve_cubic(self):
        # x^3 - 8 = 0 -> x = 2
        roots = solve_cubic(1, 0, 0, -8)
        # At least one root should be 2
        assert any(abs(r - 2) < 0.1 for r in roots)


class TestSimplification:
    """Tests for simplification functions."""
    
    def test_simplify_arithmetic(self):
        assert simplify("2+3") == "5"
        assert simplify("10/2") == "5.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
