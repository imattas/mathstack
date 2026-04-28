"""
Advanced algebraic operations including polynomial handling,
equation solving, simplification, and factorization.
"""

import math
import re
from typing import List, Dict, Tuple, Union, Optional
from fractions import Fraction


class Polynomial:
    """Represents and manipulates polynomial expressions.
    
    A polynomial is stored as a dictionary mapping exponents to coefficients.
    For example: 3x^2 + 2x + 1 is stored as {2: 3, 1: 2, 0: 1}
    """
    
    def __init__(self, coefficients: Dict[int, float] = None, **kwargs):
        """Initialize a polynomial.
        
        Args:
            coefficients: Dict mapping exponents to coefficients
            **kwargs: Alternative way to specify coefficients (e.g., x2=3, x1=2, x0=1)
        """
        self.coefficients = coefficients or {}
        
        # Handle kwargs format (x2=3 means coefficient for x^2)
        for key, value in kwargs.items():
            if key.startswith('x'):
                try:
                    exponent = int(key[1:])
                    self.coefficients[exponent] = value
                except ValueError:
                    pass
        
        # Remove zero coefficients
        self.coefficients = {k: v for k, v in self.coefficients.items() if v != 0}
    
    def __repr__(self):
        """String representation of polynomial."""
        if not self.coefficients:
            return "0"
        
        terms = []
        for exp in sorted(self.coefficients.keys(), reverse=True):
            coeff = self.coefficients[exp]
            if coeff == 0:
                continue
            
            sign = "+" if coeff > 0 else "-"
            coeff = abs(coeff)
            
            if exp == 0:
                terms.append(f"{sign}{coeff}")
            elif exp == 1:
                if coeff == 1:
                    terms.append(f"{sign}x")
                else:
                    terms.append(f"{sign}{coeff}x")
            else:
                if coeff == 1:
                    terms.append(f"{sign}x^{exp}")
                else:
                    terms.append(f"{sign}{coeff}x^{exp}")
        
        result = " ".join(terms)
        if result.startswith("+"):
            result = result[1:].strip()
        return result
    
    def __add__(self, other):
        """Add two polynomials."""
        if isinstance(other, (int, float)):
            other = Polynomial({0: other})
        result = Polynomial(self.coefficients.copy())
        for exp, coeff in other.coefficients.items():
            result.coefficients[exp] = result.coefficients.get(exp, 0) + coeff
        return result
    
    def __sub__(self, other):
        """Subtract two polynomials."""
        if isinstance(other, (int, float)):
            other = Polynomial({0: other})
        result = Polynomial(self.coefficients.copy())
        for exp, coeff in other.coefficients.items():
            result.coefficients[exp] = result.coefficients.get(exp, 0) - coeff
        return result
    
    def __mul__(self, other):
        """Multiply two polynomials."""
        if isinstance(other, (int, float)):
            result = Polynomial()
            for exp, coeff in self.coefficients.items():
                result.coefficients[exp] = coeff * other
            return result
        
        result = Polynomial()
        for exp1, coeff1 in self.coefficients.items():
            for exp2, coeff2 in other.coefficients.items():
                new_exp = exp1 + exp2
                result.coefficients[new_exp] = result.coefficients.get(new_exp, 0) + coeff1 * coeff2
        return result
    
    def evaluate(self, x: float) -> float:
        """Evaluate the polynomial at a given x value."""
        return sum(coeff * (x ** exp) for exp, coeff in self.coefficients.items())
    
    def derivative(self) -> 'Polynomial':
        """Calculate the derivative of the polynomial."""
        result = Polynomial()
        for exp, coeff in self.coefficients.items():
            if exp > 0:
                result.coefficients[exp - 1] = coeff * exp
        return result
    
    def degree(self) -> int:
        """Get the degree of the polynomial."""
        return max(self.coefficients.keys()) if self.coefficients else 0


def solve_quadratic(a: float, b: float, c: float) -> Tuple[complex, complex]:
    """Solve quadratic equation ax^2 + bx + c = 0.
    
    Args:
        a, b, c: Coefficients of quadratic equation
        
    Returns:
        Tuple of two solutions
    """
    if a == 0:
        if b == 0:
            raise ValueError("Not a valid equation")
        return (-c / b, -c / b)
    
    discriminant = b ** 2 - 4 * a * c
    sqrt_discriminant = discriminant ** 0.5
    
    x1 = (-b + sqrt_discriminant) / (2 * a)
    x2 = (-b - sqrt_discriminant) / (2 * a)
    
    return (x1, x2)


def solve_cubic(a: float, b: float, c: float, d: float) -> List[complex]:
    """Solve cubic equation ax^3 + bx^2 + cx + d = 0.
    Uses Cardano's formula.
    
    Args:
        a, b, c, d: Coefficients of cubic equation
        
    Returns:
        List of three solutions
    """
    if a == 0:
        return solve_quadratic(b, c, d)
    
    # Normalize
    b, c, d = b / a, c / a, d / a
    
    # Convert to depressed cubic t^3 + pt + q = 0
    p = c - (b ** 2) / 3
    q = d + (2 * b ** 3) / 27 - (b * c) / 3
    
    # Apply Cardano's formula
    discriminant = -(4 * p ** 3 + 27 * q ** 2)
    
    term1 = (q / 2) ** 2 + (p / 3) ** 3
    sqrt_term = term1 ** 0.5
    
    C = ((-q / 2 + sqrt_term) ** (1/3)) if (-q / 2 + sqrt_term) != 0 else 0
    if C == 0:
        C = ((-q / 2 - sqrt_term) ** (1/3)) if (-q / 2 - sqrt_term) != 0 else 0
    
    omega = (-1 + 1j * math.sqrt(3)) / 2
    
    solutions = []
    for k in range(3):
        t = omega ** k * C + (-p / 3) / (omega ** k * C) - b / 3 if C != 0 else -b / 3
        solutions.append(t)
    
    return solutions


def simplify(expression: str) -> str:
    """Simplify algebraic expression.
    
    Args:
        expression: String representation of expression
        
    Returns:
        Simplified expression
    """
    # Remove spaces
    expr = expression.replace(" ", "")
    
    # Handle basic arithmetic
    try:
        # Evaluate if it's just numbers
        if re.match(r'^[\d+\-*/().]*$', expr):
            result = eval(expr)
            return str(result)
    except:
        pass
    
    return expr


def factor_expression(expression: str) -> str:
    """Factor algebraic expression.
    
    Args:
        expression: String representation of expression
        
    Returns:
        Factored form
    """
    # Basic factoring for simple cases
    expr = expression.replace(" ", "")
    
    # Factor out common terms
    if "+" in expr or "-" in expr:
        terms = re.split(r'(?=[+-])', expr)
        terms = [t for t in terms if t]
        
        # Find GCD of coefficients
        coefficients = []
        for term in terms:
            match = re.match(r'^([+-]?\d+)', term)
            if match:
                coefficients.append(int(match.group(1)))
        
        if coefficients:
            from mathcore.core.arithmetic import gcd as calc_gcd
            common_gcd = coefficients[0]
            for coeff in coefficients[1:]:
                common_gcd = calc_gcd(common_gcd, coeff)
            
            if common_gcd > 1:
                factored_terms = []
                for term in terms:
                    match = re.match(r'^([+-]?\d+)(.*)', term)
                    if match:
                        coeff = int(match.group(1))
                        rest = match.group(2)
                        factored_terms.append(f"{coeff // common_gcd}{rest}")
                
                return f"{common_gcd}({'+'.join(factored_terms)})"
    
    return expr


def expand_expression(expression: str) -> str:
    """Expand algebraic expression.
    
    Args:
        expression: String representation of expression
        
    Returns:
        Expanded form
    """
    expr = expression.replace(" ", "")
    
    # Simple expansion for (a+b)^n where n is small
    match = re.search(r'\(([^)]+)\)\^(\d+)', expr)
    if match:
        base = match.group(1)
        power = int(match.group(2))
        
        try:
            result = eval(f"({base}) ** {power}")
            return str(result)
        except:
            pass
    
    return expr
