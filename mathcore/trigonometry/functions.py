"""Core trigonometric functions and angle utilities."""

import math
from typing import Tuple


_EPS = 1e-12


def sin(angle_radians: float) -> float:
    """Return sine of angle in radians."""
    return math.sin(angle_radians)


def cos(angle_radians: float) -> float:
    """Return cosine of angle in radians."""
    return math.cos(angle_radians)


def tan(angle_radians: float) -> float:
    """Return tangent of angle in radians."""
    return math.tan(angle_radians)


def csc(angle_radians: float) -> float:
    """Return cosecant of angle in radians."""
    s = sin(angle_radians)
    if abs(s) < _EPS:
        raise ValueError("csc undefined when sin(theta) = 0")
    return 1.0 / s


def sec(angle_radians: float) -> float:
    """Return secant of angle in radians."""
    c = cos(angle_radians)
    if abs(c) < _EPS:
        raise ValueError("sec undefined when cos(theta) = 0")
    return 1.0 / c


def cot(angle_radians: float) -> float:
    """Return cotangent of angle in radians."""
    t = tan(angle_radians)
    if abs(t) < _EPS:
        raise ValueError("cot undefined when tan(theta) = 0")
    return 1.0 / t


def asin(value: float) -> float:
    """Return inverse sine in radians."""
    return math.asin(value)


def acos(value: float) -> float:
    """Return inverse cosine in radians."""
    return math.acos(value)


def atan(value: float) -> float:
    """Return inverse tangent in radians."""
    return math.atan(value)


def radians(angle_degrees: float) -> float:
    """Convert degrees to radians."""
    return math.radians(angle_degrees)


def degrees(angle_radians: float) -> float:
    """Convert radians to degrees."""
    return math.degrees(angle_radians)


def normalize_angle_radians(angle_radians: float) -> float:
    """Normalize angle into [0, 2*pi)."""
    two_pi = 2.0 * math.pi
    return angle_radians % two_pi


def normalize_angle_degrees(angle_degrees: float) -> float:
    """Normalize angle into [0, 360)."""
    return angle_degrees % 360.0


def unit_circle_point(angle_radians: float) -> Tuple[float, float]:
    """Return (x, y) point on unit circle for angle."""
    return (cos(angle_radians), sin(angle_radians))


def taylor_sin(angle_radians: float, terms: int = 10) -> float:
    """Approximate sin(angle) via Taylor series around 0."""
    x = normalize_angle_radians(angle_radians)
    total = 0.0
    for n in range(terms):
        sign = -1.0 if n % 2 else 1.0
        total += sign * (x ** (2 * n + 1)) / math.factorial(2 * n + 1)
    return total


def taylor_cos(angle_radians: float, terms: int = 10) -> float:
    """Approximate cos(angle) via Taylor series around 0."""
    x = normalize_angle_radians(angle_radians)
    total = 0.0
    for n in range(terms):
        sign = -1.0 if n % 2 else 1.0
        total += sign * (x ** (2 * n)) / math.factorial(2 * n)
    return total
