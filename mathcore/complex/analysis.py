"""
Complex Analysis — from scratch, zero external dependencies.
Includes: Laurent series, residues, contour integration, conformal mappings,
winding number, Riemann zeta function, Gamma function (complex), Mobius transforms.
"""

import math
from typing import Callable, List, Tuple, Optional
from mathcore.complex.numbers import Complex


# ---------------------------------------------------------------------------
# Complex Function Evaluation
# ---------------------------------------------------------------------------

def complex_exp(z: Complex) -> Complex:
    """e^z for complex z = x + iy: e^x * (cos y + i sin y)."""
    ex = math.exp(z.real)
    return Complex(ex * math.cos(z.imag), ex * math.sin(z.imag))


def complex_log(z: Complex, branch: int = 0) -> Complex:
    """Principal (branch=0) or k-th branch of ln(z)."""
    r = z.magnitude()
    if r == 0:
        raise ValueError("log(0) is undefined")
    theta = z.argument() + 2 * math.pi * branch
    return Complex(math.log(r), theta)


def complex_pow(z: Complex, w: Complex) -> Complex:
    """Compute z^w = exp(w * log(z)) for complex z, w."""
    return complex_exp(w * complex_log(z))


def complex_sin(z: Complex) -> Complex:
    """sin(z) = sin(x)cosh(y) + i cos(x)sinh(y)."""
    return Complex(
        math.sin(z.real) * math.cosh(z.imag),
        math.cos(z.real) * math.sinh(z.imag)
    )


def complex_cos(z: Complex) -> Complex:
    """cos(z) = cos(x)cosh(y) - i sin(x)sinh(y)."""
    return Complex(
        math.cos(z.real) * math.cosh(z.imag),
        -math.sin(z.real) * math.sinh(z.imag)
    )


def complex_tan(z: Complex) -> Complex:
    """tan(z) = sin(z)/cos(z)."""
    return complex_sin(z) / complex_cos(z)


def complex_sinh(z: Complex) -> Complex:
    """sinh(z) = sinh(x)cos(y) + i cosh(x)sin(y)."""
    return Complex(
        math.sinh(z.real) * math.cos(z.imag),
        math.cosh(z.real) * math.sin(z.imag)
    )


def complex_cosh(z: Complex) -> Complex:
    """cosh(z) = cosh(x)cos(y) + i sinh(x)sin(y)."""
    return Complex(
        math.cosh(z.real) * math.cos(z.imag),
        math.sinh(z.real) * math.sin(z.imag)
    )


# ---------------------------------------------------------------------------
# Contour Integration (numerical)
# ---------------------------------------------------------------------------

def contour_integrate(
    f: Callable[[Complex], Complex],
    gamma: Callable[[float], Complex],
    gamma_prime: Callable[[float], Complex],
    n_points: int = 1000
) -> Complex:
    """Numerically integrate f along a contour gamma: [0,1] -> C.

    Uses the composite Simpson's rule on the parametric integral:
        integral_C f(z) dz = integral_0^1 f(gamma(t)) * gamma'(t) dt
    """
    if n_points % 2 == 1:
        n_points += 1
    h = 1.0 / n_points
    total = Complex(0, 0)
    for i in range(n_points + 1):
        t = i * h
        fval = f(gamma(t)) * gamma_prime(t)
        weight = 1 if (i == 0 or i == n_points) else (4 if i % 2 else 2)
        total = total + fval * Complex(weight, 0)
    return total * Complex(h / 3, 0)


def circle_contour(center: Complex, radius: float):
    """Return (gamma, gamma_prime) for the circle |z - center| = radius.

    gamma(t) = center + radius * e^{2pi*i*t}, t in [0,1].
    """
    def gamma(t: float) -> Complex:
        angle = 2 * math.pi * t
        return center + Complex(radius * math.cos(angle), radius * math.sin(angle))

    def gamma_prime(t: float) -> Complex:
        angle = 2 * math.pi * t
        return Complex(-2 * math.pi * radius * math.sin(angle),
                        2 * math.pi * radius * math.cos(angle))

    return gamma, gamma_prime


def cauchy_integral_formula(
    f: Callable[[Complex], Complex],
    z0: Complex,
    radius: float,
    n_points: int = 2000
) -> Complex:
    """Compute f(z0) via the Cauchy integral formula:
        f(z0) = (1/2pi*i) * integral_C f(z)/(z-z0) dz
    """
    gamma, gamma_prime = circle_contour(z0, radius)

    def integrand(z: Complex) -> Complex:
        return f(z) / (z - z0)

    integral = contour_integrate(integrand, gamma, gamma_prime, n_points)
    # Divide by 2*pi*i
    return integral / Complex(0, 2 * math.pi)


def cauchy_nth_derivative(
    f: Callable[[Complex], Complex],
    z0: Complex,
    n: int,
    radius: float,
    n_points: int = 2000
) -> Complex:
    """Compute f^(n)(z0) via the generalized Cauchy formula:
        f^(n)(z0) = n! / (2pi*i) * integral_C f(z)/(z-z0)^{n+1} dz
    """
    gamma, gamma_prime = circle_contour(z0, radius)

    def integrand(z: Complex) -> Complex:
        denom = z - z0
        # (z - z0)^(n+1)
        power = Complex(1, 0)
        for _ in range(n + 1):
            power = power * denom
        return f(z) / power

    integral = contour_integrate(integrand, gamma, gamma_prime, n_points)
    factorial_n = math.factorial(n)
    return integral * Complex(factorial_n, 0) / Complex(0, 2 * math.pi)


# ---------------------------------------------------------------------------
# Residues
# ---------------------------------------------------------------------------

def residue_simple_pole(
    f: Callable[[Complex], Complex],
    z0: Complex,
    eps: float = 1e-7
) -> Complex:
    """Estimate the residue of f at a simple pole z0 numerically:
        Res(f, z0) = lim_{z->z0} (z - z0) * f(z)
    Evaluated at z0 + eps for numerical stability.
    """
    z = z0 + Complex(eps, eps)
    return (z - z0) * f(z)


def residue_by_contour(
    f: Callable[[Complex], Complex],
    z0: Complex,
    radius: float = 0.01,
    n_points: int = 2000
) -> Complex:
    """Compute residue of f at z0 via:
        Res(f, z0) = (1/2pi*i) * integral_{small circle around z0} f(z) dz
    """
    gamma, gamma_prime = circle_contour(z0, radius)
    integral = contour_integrate(f, gamma, gamma_prime, n_points)
    return integral / Complex(0, 2 * math.pi)


# ---------------------------------------------------------------------------
# Winding Number
# ---------------------------------------------------------------------------

def winding_number(
    gamma: Callable[[float], Complex],
    z0: Complex,
    n_points: int = 1000
) -> int:
    """Compute the winding number of contour gamma around point z0.

    Returns an integer (number of times gamma winds around z0 counterclockwise).
    """
    total_angle = 0.0
    dt = 1.0 / n_points
    prev = gamma(0)
    for i in range(1, n_points + 1):
        curr = gamma(i * dt)
        dz = curr - z0
        prev_z = prev - z0
        # angle increment
        angle = math.atan2(dz.imag, dz.real) - math.atan2(prev_z.imag, prev_z.real)
        # Wrap to [-pi, pi]
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        total_angle += angle
        prev = curr
    return round(total_angle / (2 * math.pi))


# ---------------------------------------------------------------------------
# Laurent Series Coefficients
# ---------------------------------------------------------------------------

def laurent_coefficients(
    f: Callable[[Complex], Complex],
    z0: Complex,
    n_neg: int,
    n_pos: int,
    radius: float = 0.5,
    n_points: int = 2000
) -> List[Tuple[int, Complex]]:
    """Compute Laurent series coefficients c_n for -n_neg <= n <= n_pos:

        f(z) = sum_n c_n (z - z0)^n

    Each c_n = (1/2pi*i) * integral f(z)/(z-z0)^{n+1} dz.
    """
    gamma, gamma_prime = circle_contour(z0, radius)
    coeffs = []
    for n in range(-n_neg, n_pos + 1):
        def integrand(z: Complex, _n=n) -> Complex:
            denom = z - z0
            power = Complex(1, 0)
            exp = _n + 1
            if exp >= 0:
                for _ in range(exp):
                    power = power * denom
            else:
                for _ in range(-exp):
                    power = power / denom
            return f(z) / power
        integral = contour_integrate(integrand, gamma, gamma_prime, n_points)
        c_n = integral / Complex(0, 2 * math.pi)
        coeffs.append((n, c_n))
    return coeffs


# ---------------------------------------------------------------------------
# Conformal Mappings (Mobius / Linear Fractional Transforms)
# ---------------------------------------------------------------------------

class MobiusTransform:
    """Mobius (linear fractional) transformation: T(z) = (az + b)/(cz + d)."""

    def __init__(self, a: Complex, b: Complex, c: Complex, d: Complex):
        det = a * d - b * c
        if det.magnitude() < 1e-15:
            raise ValueError("Degenerate Mobius transform (ad - bc = 0)")
        self.a, self.b, self.c, self.d = a, b, c, d

    def __call__(self, z: Complex) -> Complex:
        return (self.a * z + self.b) / (self.c * z + self.d)

    def inverse(self) -> "MobiusTransform":
        """Return the inverse transform T^{-1}."""
        return MobiusTransform(self.d, -self.b, -self.c, self.a)

    def compose(self, other: "MobiusTransform") -> "MobiusTransform":
        """Return self composed with other: (self ∘ other)(z) = self(other(z))."""
        a = self.a * other.a + self.b * other.c
        b = self.a * other.b + self.b * other.d
        c = self.c * other.a + self.d * other.c
        d = self.c * other.b + self.d * other.d
        return MobiusTransform(a, b, c, d)

    def fixed_points(self) -> List[Complex]:
        """Return fixed points of the transform (solutions to T(z) = z)."""
        # c*z^2 + (d-a)*z - b = 0
        if self.c.magnitude() < 1e-15:
            # a*z + b = d*z => z = b/(d-a)
            if (self.d - self.a).magnitude() < 1e-15:
                return []
            return [self.b / (self.d - self.a)]
        # Quadratic: c*z^2 + (d-a)*z - b = 0
        A = self.c
        B = self.d - self.a
        C = -self.b
        disc = B * B - A * C * Complex(4, 0)
        sqrt_disc = disc.sqrt()
        z1 = (-B + sqrt_disc) / (A * Complex(2, 0))
        z2 = (-B - sqrt_disc) / (A * Complex(2, 0))
        return [z1, z2]

    def cross_ratio(self, z1: Complex, z2: Complex, z3: Complex) -> Complex:
        """Return the cross-ratio (z1,z2;z3,infinity) = (z1-z3)/(z2-z3)."""
        return (z1 - z3) / (z2 - z3)

    @staticmethod
    def from_three_points(
        z1: Complex, z2: Complex, z3: Complex,
        w1: Complex, w2: Complex, w3: Complex
    ) -> "MobiusTransform":
        """Construct the unique Mobius transform mapping z1->w1, z2->w2, z3->w3."""
        # Use cross-ratio: (w,w1;w2,w3) = (z,z1;z2,z3)
        # Expand to get T(z) = (az+b)/(cz+d) in terms of the three pairs.
        # Cross-ratio maps: a = z2 - z3, b = z1*(z3-z2), c = z2-z1, d = z3*(z1-z2)
        # Then compose with inverse for w.
        def cross_ratio_map(p: Complex, p1: Complex, p2: Complex, p3: Complex):
            a = p2 - p3
            b = p1 * (p3 - p2)
            c = p2 - p1
            d = p3 * (p1 - p2)
            return MobiusTransform(a, b, c, d)

        T_z = cross_ratio_map(z1, z1, z2, z3)
        T_w = cross_ratio_map(w1, w1, w2, w3)
        return T_w.inverse().compose(T_z)


# ---------------------------------------------------------------------------
# Riemann Zeta Function (numerical)
# ---------------------------------------------------------------------------

def riemann_zeta(s: Complex, n_terms: int = 200) -> Complex:
    """Compute the Riemann zeta function zeta(s) via the Euler-Maclaurin formula.

    For Re(s) > 1, uses the direct series sum.
    For Re(s) <= 1 (not s=1), uses the functional equation:
        zeta(s) = 2^s * pi^(s-1) * sin(pi*s/2) * Gamma(1-s) * zeta(1-s)
    """
    if s.real > 1:
        total = Complex(0, 0)
        for n in range(1, n_terms + 1):
            # n^(-s) = exp(-s * log(n))
            log_n = math.log(n)
            term = complex_exp(s * Complex(-log_n, 0))
            total = total + term
        return total
    elif abs(s.real - 1) < 1e-10 and abs(s.imag) < 1e-10:
        raise ValueError("zeta(1) is a pole (diverges)")
    else:
        # Functional equation
        one_minus_s = Complex(1, 0) - s
        zeta_1ms = riemann_zeta(one_minus_s, n_terms)
        pi_s = complex_exp(s * Complex(math.log(math.pi), 0))
        sin_half = complex_sin(s * Complex(math.pi / 2, 0))
        gamma_1ms = complex_gamma(one_minus_s)
        two_s = complex_exp(s * Complex(math.log(2), 0))
        return two_s * pi_s * Complex(1 / math.pi, 0) * sin_half * gamma_1ms * zeta_1ms


def complex_gamma(z: Complex, n_terms: int = 50) -> Complex:
    """Compute Gamma(z) for complex z using the Lanczos approximation."""
    # Lanczos approximation with g=7
    g = 7
    p = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]
    if z.real < 0.5:
        # Reflection formula: Gamma(z)*Gamma(1-z) = pi/sin(pi*z)
        return Complex(math.pi, 0) / (complex_sin(Complex(math.pi, 0) * z) * complex_gamma(Complex(1, 0) - z))
    z = z - Complex(1, 0)
    x = Complex(p[0], 0)
    for i in range(1, g + 2):
        x = x + Complex(p[i], 0) / (z + Complex(i, 0))
    t = z + Complex(g + 0.5, 0)
    sqrt_2pi = Complex(math.sqrt(2 * math.pi), 0)
    return sqrt_2pi * complex_pow(t, z + Complex(0.5, 0)) * complex_exp(-t) * x


# ---------------------------------------------------------------------------
# Analytic Continuation & Series
# ---------------------------------------------------------------------------

def taylor_complex(
    f: Callable[[Complex], Complex],
    z0: Complex,
    n_terms: int,
    radius: float = 0.5,
    n_points: int = 2000
) -> List[Complex]:
    """Return Taylor series coefficients [c0, c1, ..., c_{n-1}] of f around z0.

    c_n = f^(n)(z0) / n!  computed via contour integration.
    """
    coeffs = []
    for n in range(n_terms):
        cn = cauchy_nth_derivative(f, z0, n, radius, n_points) / Complex(math.factorial(n), 0)
        coeffs.append(cn)
    return coeffs


def evaluate_taylor(coeffs: List[Complex], z0: Complex, z: Complex) -> Complex:
    """Evaluate a Taylor series sum c_n * (z - z0)^n at point z."""
    w = z - z0
    result = Complex(0, 0)
    power = Complex(1, 0)
    for c in coeffs:
        result = result + c * power
        power = power * w
    return result


def argument_principle_zeros_minus_poles(
    f: Callable[[Complex], Complex],
    center: Complex,
    radius: float,
    n_points: int = 2000
) -> int:
    """Return N - P (zeros minus poles of f inside contour) via argument principle:
        N - P = (1/2pi*i) * integral_C f'(z)/f(z) dz
    Approximates f' by finite differences.
    """
    eps = 1e-6

    def log_deriv(z: Complex) -> Complex:
        fz = f(z)
        fprime = (f(z + Complex(eps, 0)) - f(z - Complex(eps, 0))) / Complex(2 * eps, 0)
        return fprime / fz

    gamma, gamma_prime = circle_contour(center, radius)
    integral = contour_integrate(log_deriv, gamma, gamma_prime, n_points)
    result = integral / Complex(0, 2 * math.pi)
    return round(result.real)
