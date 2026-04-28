"""
Advanced calculus operations including derivatives, integrals,
limits, series expansions, and optimization.
"""

import math
from typing import Callable, List, Tuple, Optional
from mathcore.core.algebra import Polynomial


def derivative(f: Callable[[float], float], x: float, h: float = 1e-7) -> float:
    """Calculate numerical derivative of function f at point x.
    Uses central difference method for accuracy.
    
    Args:
        f: Function to differentiate
        x: Point at which to calculate derivative
        h: Step size (smaller is more accurate but less stable)
        
    Returns:
        Derivative at x
    """
    # Central difference: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
    return (f(x + h) - f(x - h)) / (2 * h)


def second_derivative(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """Calculate numerical second derivative.
    
    Args:
        f: Function to differentiate
        x: Point at which to calculate
        h: Step size
        
    Returns:
        Second derivative at x
    """
    # f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h^2
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)


def integral_riemann(f: Callable[[float], float], a: float, b: float, 
                     n: int = 1000) -> float:
    """Approximate definite integral using Riemann sum (midpoint rule).
    
    Args:
        f: Function to integrate
        a, b: Integration limits
        n: Number of subintervals
        
    Returns:
        Approximate integral value
    """
    width = (b - a) / n
    total = 0
    for i in range(n):
        x = a + (i + 0.5) * width
        total += f(x)
    return total * width


def integral_trapezoid(f: Callable[[float], float], a: float, b: float, 
                      n: int = 1000) -> float:
    """Approximate definite integral using trapezoidal rule.
    More accurate than Riemann sum for smooth functions.
    
    Args:
        f: Function to integrate
        a, b: Integration limits
        n: Number of subintervals
        
    Returns:
        Approximate integral value
    """
    width = (b - a) / n
    total = (f(a) + f(b)) / 2
    for i in range(1, n):
        x = a + i * width
        total += f(x)
    return total * width


def integral_simpson(f: Callable[[float], float], a: float, b: float, 
                    n: int = 1000) -> float:
    """Approximate definite integral using Simpson's rule.
    Very accurate for smooth functions. n should be even.
    
    Args:
        f: Function to integrate
        a, b: Integration limits
        n: Number of subintervals (must be even)
        
    Returns:
        Approximate integral value
    """
    if n % 2 != 0:
        n += 1
    
    width = (b - a) / n
    total = f(a) + f(b)
    
    for i in range(1, n, 2):
        x = a + i * width
        total += 4 * f(x)
    
    for i in range(2, n - 1, 2):
        x = a + i * width
        total += 2 * f(x)
    
    return total * width / 3


def integral(f: Callable[[float], float], a: float, b: float, 
            method: str = 'simpson', n: int = 1000) -> float:
    """Calculate definite integral using specified method.
    
    Args:
        f: Function to integrate
        a, b: Integration limits
        method: 'simpson' (default, most accurate), 'trapezoid', or 'riemann'
        n: Number of subintervals
        
    Returns:
        Approximate integral value
    """
    if method == 'simpson':
        return integral_simpson(f, a, b, n)
    elif method == 'trapezoid':
        return integral_trapezoid(f, a, b, n)
    else:
        return integral_riemann(f, a, b, n)


def limit(f: Callable[[float], float], x: float, direction: str = 'both', 
         epsilon: float = 1e-10) -> Optional[float]:
    """Compute limit of function as x approaches a point.
    
    Args:
        f: Function
        x: Point to approach
        direction: 'left', 'right', or 'both'
        epsilon: Step size for approaching the limit
        
    Returns:
        Approximate limit value
    """
    try:
        if direction == 'left':
            return f(x - epsilon)
        elif direction == 'right':
            return f(x + epsilon)
        else:  # both
            left = f(x - epsilon)
            right = f(x + epsilon)
            if abs(left - right) < 1e-6:
                return (left + right) / 2
            return None
    except:
        return None


def find_critical_points(f: Callable[[float], float], a: float, b: float, 
                        step: float = 0.1) -> List[float]:
    """Find critical points (where derivative = 0) in interval [a, b].
    
    Args:
        f: Function to analyze
        a, b: Interval bounds
        step: Search step size
        
    Returns:
        List of approximate critical point locations
    """
    critical_points = []
    x = a
    prev_deriv = derivative(f, x)
    
    while x <= b:
        x += step
        curr_deriv = derivative(f, x)
        
        # Check for sign change (critical point)
        if prev_deriv * curr_deriv < 0:
            # Refine using binary search
            left, right = x - step, x
            for _ in range(10):  # 10 iterations for refinement
                mid = (left + right) / 2
                if derivative(f, mid) * prev_deriv < 0:
                    right = mid
                else:
                    left = mid
            critical_points.append((left + right) / 2)
        
        prev_deriv = curr_deriv
    
    return critical_points


def second_derivative_test(f: Callable[[float], float], x: float) -> str:
    """Classify critical point using second derivative test.
    
    Args:
        f: Function
        x: Critical point to test
        
    Returns:
        'local_maximum', 'local_minimum', or 'inflection_point'
    """
    f_double_prime = second_derivative(f, x)
    
    if abs(f_double_prime) < 1e-10:
        return 'inflection_point'
    elif f_double_prime > 0:
        return 'local_minimum'
    else:
        return 'local_maximum'


def taylor_series(f: Callable[[float], float], x0: float, n: int = 5) -> Polynomial:
    """Compute Taylor series expansion around x0.
    
    Args:
        f: Function to expand
        x0: Point around which to expand
        n: Degree of polynomial
        
    Returns:
        Polynomial approximation
    """
    coefficients = {}
    h = 1e-5
    
    for i in range(n + 1):
        # Approximate nth derivative using finite differences
        if i == 0:
            coefficients[0] = f(x0)
        else:
            # Approximate derivative
            deriv_f = f
            for _ in range(i):
                deriv_f = lambda t, df=deriv_f, h=h: derivative(df, t, h)
            
            coefficients[i] = deriv_f(x0) / math.factorial(i)
    
    return Polynomial(coefficients)


def series_expansion(f: Callable[[float], float], x0: float = 0, n: int = 5) -> str:
    """Get string representation of series expansion.
    
    Args:
        f: Function to expand
        x0: Expansion point
        n: Degree
        
    Returns:
        String representation
    """
    poly = taylor_series(f, x0, n)
    return str(poly)


def find_root_bisection(f: Callable[[float], float], a: float, b: float, 
                       tolerance: float = 1e-10) -> float:
    """Find root of function using bisection method.
    Requires f(a) and f(b) to have opposite signs.
    
    Args:
        f: Function
        a, b: Interval endpoints
        tolerance: Convergence tolerance
        
    Returns:
        Approximate root
    """
    if f(a) * f(b) > 0:
        raise ValueError("Function must have opposite signs at endpoints")
    
    while abs(b - a) > tolerance:
        c = (a + b) / 2
        if f(c) == 0:
            return c
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
    
    return (a + b) / 2


def find_root_newton(f: Callable[[float], float], x0: float, 
                    tolerance: float = 1e-10, max_iterations: int = 100) -> float:
    """Find root of function using Newton-Raphson method.
    
    Args:
        f: Function
        x0: Initial guess
        tolerance: Convergence tolerance
        max_iterations: Maximum iterations
        
    Returns:
        Approximate root
    """
    x = x0
    for _ in range(max_iterations):
        f_x = f(x)
        f_prime_x = derivative(f, x)
        
        if abs(f_prime_x) < 1e-15:
            raise ValueError("Derivative too small - method fails")
        
        x_new = x - f_x / f_prime_x
        
        if abs(x_new - x) < tolerance:
            return x_new
        
        x = x_new
    
    return x


def optimize_minimize(f: Callable[[float], float], a: float, b: float, 
                     method: str = 'ternary') -> Tuple[float, float]:
    """Find minimum of function in interval [a, b].
    
    Args:
        f: Function to minimize
        a, b: Search interval
        method: 'ternary' or 'golden_section'
        
    Returns:
        Tuple of (x_min, f_min)
    """
    if method == 'golden_section':
        phi = (1 + math.sqrt(5)) / 2
        resphi = 2 - phi
        
        tol = 1e-5
        while abs(b - a) > tol:
            x1 = a + resphi * (b - a)
            x2 = b - resphi * (b - a)
            
            if f(x1) < f(x2):
                b = x2
            else:
                a = x1
    else:  # ternary
        while abs(b - a) > 1e-5:
            m1 = a + (b - a) / 3
            m2 = b - (b - a) / 3
            
            if f(m1) > f(m2):
                a = m1
            else:
                b = m2
    
    x_min = (a + b) / 2
    return (x_min, f(x_min))
