"""Unit tests for calculus module."""

import pytest
import math
from mathcore.core.calculus import (
    derivative, second_derivative, integral,
    find_critical_points, second_derivative_test,
    find_root_bisection, find_root_newton,
    optimize_minimize
)


class TestDerivatives:
    """Tests for derivative functions."""
    
    def test_derivative_polynomial(self):
        # f(x) = x^2, f'(x) = 2x, f'(2) = 4
        f = lambda x: x**2
        deriv = derivative(f, 2)
        assert abs(deriv - 4) < 0.01
    
    def test_second_derivative(self):
        # f(x) = x^3, f''(x) = 6x, f''(2) = 12
        f = lambda x: x**3
        second_deriv = second_derivative(f, 2)
        assert abs(second_deriv - 12) < 0.1


class TestIntegration:
    """Tests for integration functions."""
    
    def test_integral_simpons(self):
        # ∫x² from 0 to 1 = 1/3
        f = lambda x: x**2
        result = integral(f, 0, 1, method='simpson', n=1000)
        assert abs(result - 1/3) < 0.001
    
    def test_integral_trapezoid(self):
        # ∫x from 0 to 2 = 2
        f = lambda x: x
        result = integral(f, 0, 2, method='trapezoid', n=1000)
        assert abs(result - 2) < 0.001


class TestRootFinding:
    """Tests for root finding functions."""
    
    def test_bisection(self):
        # Find root of x² - 4 near 0 and 3 (root at 2)
        f = lambda x: x**2 - 4
        root = find_root_bisection(f, 0, 3)
        assert abs(root - 2) < 1e-6
    
    def test_newton_raphson(self):
        # Find root of x² - 2 (root at sqrt(2) ≈ 1.414)
        f = lambda x: x**2 - 2
        root = find_root_newton(f, 1.5)
        assert abs(root - math.sqrt(2)) < 1e-6


class TestOptimization:
    """Tests for optimization functions."""
    
    def test_minimize_simple(self):
        # f(x) = (x-2)² has minimum at x=2
        f = lambda x: (x - 2)**2
        x_min, f_min = optimize_minimize(f, 0, 5)
        assert abs(x_min - 2) < 0.1
        assert abs(f_min) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
