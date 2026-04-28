"""Triangle trigonometry helpers (laws of sines/cosines, area formulas)."""

import math


_EPS = 1e-12


def pythagorean_hypotenuse(leg_a: float, leg_b: float) -> float:
    """Return hypotenuse length from two right-triangle legs."""
    if leg_a < 0 or leg_b < 0:
        raise ValueError("Leg lengths must be non-negative")
    return math.sqrt(leg_a * leg_a + leg_b * leg_b)


def pythagorean_leg(hypotenuse: float, other_leg: float) -> float:
    """Return missing right-triangle leg from hypotenuse and another leg."""
    if hypotenuse <= 0 or other_leg < 0 or other_leg > hypotenuse:
        raise ValueError("Invalid side lengths for right triangle")
    return math.sqrt(hypotenuse * hypotenuse - other_leg * other_leg)


def law_of_sines_side(known_side: float, known_angle_radians: float, target_angle_radians: float) -> float:
    """Solve unknown side via law of sines: a/sin(A)=b/sin(B)."""
    denom = math.sin(known_angle_radians)
    if abs(denom) < _EPS:
        raise ValueError("Known angle cannot have sin(angle)=0")
    return known_side * math.sin(target_angle_radians) / denom


def law_of_sines_angle(known_side: float, known_angle_radians: float, target_side: float) -> float:
    """Solve unknown angle via law of sines (principal solution)."""
    denom = known_side
    if abs(denom) < _EPS:
        raise ValueError("Known side must be non-zero")
    value = target_side * math.sin(known_angle_radians) / denom
    if value < -1.0 or value > 1.0:
        raise ValueError("No real triangle satisfies these dimensions")
    return math.asin(value)


def law_of_cosines_side(side_b: float, side_c: float, included_angle_radians: float) -> float:
    """Solve missing side a via law of cosines."""
    if side_b <= 0 or side_c <= 0:
        raise ValueError("Side lengths must be positive")
    value = side_b * side_b + side_c * side_c - 2.0 * side_b * side_c * math.cos(included_angle_radians)
    if value < 0 and value > -_EPS:
        value = 0.0
    if value < 0:
        raise ValueError("Invalid geometry for law of cosines")
    return math.sqrt(value)


def law_of_cosines_angle(side_a: float, side_b: float, side_c: float) -> float:
    """Solve angle A opposite side a via law of cosines."""
    if side_a <= 0 or side_b <= 0 or side_c <= 0:
        raise ValueError("Side lengths must be positive")
    denom = 2.0 * side_b * side_c
    if abs(denom) < _EPS:
        raise ValueError("Invalid denominator in law of cosines")
    value = (side_b * side_b + side_c * side_c - side_a * side_a) / denom
    value = max(-1.0, min(1.0, value))
    return math.acos(value)


def triangle_area_sas(side_a: float, side_b: float, included_angle_radians: float) -> float:
    """Compute triangle area from two sides and included angle."""
    if side_a <= 0 or side_b <= 0:
        raise ValueError("Sides must be positive")
    return 0.5 * side_a * side_b * math.sin(included_angle_radians)


def triangle_area_heron(side_a: float, side_b: float, side_c: float) -> float:
    """Compute triangle area using Heron's formula."""
    if side_a <= 0 or side_b <= 0 or side_c <= 0:
        raise ValueError("Sides must be positive")
    s = (side_a + side_b + side_c) / 2.0
    value = s * (s - side_a) * (s - side_b) * (s - side_c)
    if value < 0 and value > -_EPS:
        value = 0.0
    if value < 0:
        raise ValueError("Triangle inequality violated")
    return math.sqrt(value)
