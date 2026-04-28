"""
Complex Numbers Module
Comprehensive complex number operations and analysis.
"""

import math
from typing import Tuple, Union


class Complex:
    """Advanced complex number class with full operations."""
    
    def __init__(self, real: float = 0, imag: float = 0):
        """Initialize complex number z = real + imag*i"""
        self.real = real
        self.imag = imag
    
    def __repr__(self) -> str:
        if self.imag >= 0:
            return f"({self.real} + {self.imag}i)"
        return f"({self.real} - {abs(self.imag)}i)"
    
    def __add__(self, other: 'Complex') -> 'Complex':
        """Add two complex numbers."""
        if isinstance(other, (int, float)):
            other = Complex(other, 0)
        return Complex(self.real + other.real, self.imag + other.imag)
    
    def __sub__(self, other: 'Complex') -> 'Complex':
        """Subtract complex numbers."""
        if isinstance(other, (int, float)):
            other = Complex(other, 0)
        return Complex(self.real - other.real, self.imag - other.imag)
    
    def __mul__(self, other: 'Complex') -> 'Complex':
        """Multiply complex numbers."""
        if isinstance(other, (int, float)):
            other = Complex(other, 0)
        # (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        real = self.real * other.real - self.imag * other.imag
        imag = self.real * other.imag + self.imag * other.real
        return Complex(real, imag)
    
    def __truediv__(self, other: 'Complex') -> 'Complex':
        """Divide complex numbers."""
        if isinstance(other, (int, float)):
            other = Complex(other, 0)
        # (a + bi) / (c + di) = ((a + bi)(c - di)) / (c² + d²)
        denominator = other.real ** 2 + other.imag ** 2
        if abs(denominator) < 1e-15:
            raise ValueError("Division by zero")
        
        numerator = self * Complex(other.real, -other.imag)
        return Complex(numerator.real / denominator, numerator.imag / denominator)
    
    def __pow__(self, n: int) -> 'Complex':
        """Raise to integer power."""
        if n == 0:
            return Complex(1, 0)
        if n < 0:
            return Complex(1, 0) / (self ** (-n))
        
        result = Complex(1, 0)
        for _ in range(n):
            result = result * self
        return result
    
    def conjugate(self) -> 'Complex':
        """Return complex conjugate."""
        return Complex(self.real, -self.imag)
    
    def magnitude(self) -> float:
        """Calculate |z| = √(a² + b²)"""
        return math.sqrt(self.real ** 2 + self.imag ** 2)
    
    def argument(self) -> float:
        """Calculate arg(z) in radians."""
        return math.atan2(self.imag, self.real)
    
    def argument_degrees(self) -> float:
        """Calculate arg(z) in degrees."""
        return math.degrees(self.argument())
    
    def polar_form(self) -> Tuple[float, float]:
        """Return (magnitude, angle_in_radians)."""
        return (self.magnitude(), self.argument())
    
    @staticmethod
    def from_polar(r: float, theta: float) -> 'Complex':
        """Create complex number from polar form.
        
        Args:
            r: Magnitude
            theta: Angle in radians
        """
        return Complex(r * math.cos(theta), r * math.sin(theta))
    
    def sqrt(self) -> 'Complex':
        """Calculate square root."""
        r, theta = self.polar_form()
        sqrt_r = math.sqrt(r)
        return Complex.from_polar(sqrt_r, theta / 2)
    
    def exp(self) -> 'Complex':
        """Calculate e^z = e^(a+bi) = e^a * (cos(b) + i*sin(b))"""
        exp_a = math.exp(self.real)
        return Complex(exp_a * math.cos(self.imag), exp_a * math.sin(self.imag))
    
    def ln(self) -> 'Complex':
        """Calculate natural logarithm."""
        r, theta = self.polar_form()
        if r <= 0:
            raise ValueError("Cannot take logarithm of non-positive magnitude")
        return Complex(math.log(r), theta)
    
    def sin(self) -> 'Complex':
        """Calculate sin(z)."""
        # sin(a+bi) = sin(a)cosh(b) + i*cos(a)sinh(b)
        sin_a = math.sin(self.real)
        cos_a = math.cos(self.real)
        sinh_b = math.sinh(self.imag)
        cosh_b = math.cosh(self.imag)
        return Complex(sin_a * cosh_b, cos_a * sinh_b)
    
    def cos(self) -> 'Complex':
        """Calculate cos(z)."""
        # cos(a+bi) = cos(a)cosh(b) - i*sin(a)sinh(b)
        sin_a = math.sin(self.real)
        cos_a = math.cos(self.real)
        sinh_b = math.sinh(self.imag)
        cosh_b = math.cosh(self.imag)
        return Complex(cos_a * cosh_b, -sin_a * sinh_b)
    
    def tan(self) -> 'Complex':
        """Calculate tan(z)."""
        sin_z = self.sin()
        cos_z = self.cos()
        if abs(cos_z.magnitude()) < 1e-15:
            raise ValueError("tan(z) undefined")
        return sin_z / cos_z


class QuadraticFormula:
    """Solve quadratic equations with complex coefficients."""
    
    @staticmethod
    def solve(a: Complex, b: Complex, c: Complex) -> Tuple[Complex, Complex]:
        """Solve az² + bz + c = 0 using quadratic formula.
        
        Args:
            a, b, c: Complex coefficients
            
        Returns:
            Tuple of two solutions
        """
        if abs(a.magnitude()) < 1e-15:
            raise ValueError("Coefficient a cannot be zero")
        
        discriminant = b * b - 4 * a * c
        sqrt_disc = discriminant.sqrt()
        
        z1 = (-b + sqrt_disc) / (2 * a)
        z2 = (-b - sqrt_disc) / (2 * a)
        
        return (z1, z2)


class ComplexAnalysis:
    """Complex analysis utilities."""
    
    @staticmethod
    def is_real(z: Complex, tolerance: float = 1e-10) -> bool:
        """Check if complex number is effectively real."""
        return abs(z.imag) < tolerance
    
    @staticmethod
    def is_imaginary(z: Complex, tolerance: float = 1e-10) -> bool:
        """Check if complex number is purely imaginary."""
        return abs(z.real) < tolerance
    
    @staticmethod
    def distance(z1: Complex, z2: Complex) -> float:
        """Calculate distance between two complex numbers."""
        diff = z1 - z2
        return diff.magnitude()
    
    @staticmethod
    def roots_of_unity(n: int) -> list:
        """Calculate nth roots of unity."""
        roots = []
        for k in range(n):
            theta = 2 * math.pi * k / n
            roots.append(Complex.from_polar(1, theta))
        return roots
    
    @staticmethod
    def mandelbrot_iteration(c: Complex, max_iter: int = 100) -> int:
        """Calculate Mandelbrot set iteration count for point c."""
        z = Complex(0, 0)
        for n in range(max_iter):
            if z.magnitude() > 2:
                return n
            z = z * z + c
        return max_iter
    
    @staticmethod
    def julia_set_iteration(z: Complex, c: Complex, max_iter: int = 100) -> int:
        """Calculate Julia set iteration count."""
        for n in range(max_iter):
            if z.magnitude() > 2:
                return n
            z = z * z + c
        return max_iter
