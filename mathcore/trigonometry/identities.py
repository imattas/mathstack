"""Identity helpers for trigonometric validation and checks."""

import math


_EPS = 1e-10


def verify_pythagorean_identity(angle_radians: float, tolerance: float = _EPS) -> bool:
    """Check if sin^2(theta) + cos^2(theta) = 1 within tolerance."""
    lhs = math.sin(angle_radians) ** 2 + math.cos(angle_radians) ** 2
    return abs(lhs - 1.0) <= tolerance


def verify_sum_difference_identity(alpha_radians: float, beta_radians: float, tolerance: float = _EPS) -> bool:
    """Verify sin(alpha+beta) identity numerically."""
    lhs = math.sin(alpha_radians + beta_radians)
    rhs = math.sin(alpha_radians) * math.cos(beta_radians) + math.cos(alpha_radians) * math.sin(beta_radians)
    return abs(lhs - rhs) <= tolerance


def verify_double_angle_identity(angle_radians: float, tolerance: float = _EPS) -> bool:
    """Verify cos(2x)=cos^2(x)-sin^2(x) numerically."""
    lhs = math.cos(2.0 * angle_radians)
    rhs = math.cos(angle_radians) ** 2 - math.sin(angle_radians) ** 2
    return abs(lhs - rhs) <= tolerance
