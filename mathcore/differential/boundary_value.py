"""
Boundary Value Problems — from scratch, zero external dependencies.
Includes: shooting method (linear & nonlinear), finite differences,
finite elements (1D), Sturm-Liouville eigenvalue problems,
Chebyshev collocation, Green's functions.
"""

import math
from typing import List, Tuple, Callable, Optional


# ---------------------------------------------------------------------------
# Shooting Method
# ---------------------------------------------------------------------------

def shooting_method_linear(
    p: Callable[[float], float],
    q: Callable[[float], float],
    r: Callable[[float], float],
    a: float,
    b: float,
    alpha: float,
    beta: float,
    n_steps: int = 1000
) -> Tuple[List[float], List[float]]:
    """Solve the linear BVP y'' + p(x)y' + q(x)y = r(x), y(a)=alpha, y(b)=beta.

    Uses the superposition principle with two IVP solutions.
    Returns (x_values, y_values).
    """
    h = (b - a) / n_steps

    def rk4_system(f, y0, x0, x1, h):
        """Solve y' = f(x, y) (system) from x0 to x1."""
        y = list(y0)
        x = x0
        while x < x1 - h / 2:
            k1 = [h * fi for fi in f(x, y)]
            k2 = [h * fi for fi in f(x + h / 2, [y[i] + k1[i] / 2 for i in range(len(y))])]
            k3 = [h * fi for fi in f(x + h / 2, [y[i] + k2[i] / 2 for i in range(len(y))])]
            k4 = [h * fi for fi in f(x + h, [y[i] + k3[i] for i in range(len(y))])]
            y = [y[i] + (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6 for i in range(len(y))]
            x = min(x + h, x1)
        return y

    # Convert to first-order system: u = [y, y']
    def f_system(x, u):
        return [u[1], r(x) - p(x) * u[1] - q(x) * u[0]]

    # Solve y1: y1(a) = alpha, y1'(a) = 0
    # Solve y2: y2(a) = 0, y2'(a) = 1
    y1_vals = [alpha]
    y2_vals = [0.0]
    y1_prime_vals = [0.0]
    y2_prime_vals = [1.0]
    xs = [a]

    y1 = [alpha, 0.0]
    y2 = [0.0, 1.0]
    x = a
    for _ in range(n_steps):
        x_new = x + h
        y1 = rk4_system(f_system, y1, x, x_new, h)
        y2 = rk4_system(f_system, y2, x, x_new, h)
        xs.append(x_new)
        y1_vals.append(y1[0])
        y2_vals.append(y2[0])
        x = x_new

    # y = y1 + c * y2, enforce y(b) = beta
    c = (beta - y1_vals[-1]) / y2_vals[-1] if abs(y2_vals[-1]) > 1e-14 else 0.0
    y_vals = [y1_vals[i] + c * y2_vals[i] for i in range(len(xs))]
    return xs, y_vals


def shooting_method_nonlinear(
    f: Callable[[float, float, float], float],
    a: float,
    b: float,
    alpha: float,
    beta: float,
    n_steps: int = 1000,
    tol: float = 1e-8,
    max_iter: int = 50
) -> Tuple[List[float], List[float]]:
    """Solve the nonlinear BVP y'' = f(x, y, y'), y(a)=alpha, y(b)=beta.

    Uses Newton's method for the shooting parameter.
    Returns (x_values, y_values).
    """
    h = (b - a) / n_steps

    def integrate(slope: float) -> Tuple[List[float], List[float]]:
        y, yp = alpha, slope
        xs, ys = [a], [y]
        x = a
        for _ in range(n_steps):
            # RK4 step
            k1y = h * yp
            k1z = h * f(x, y, yp)
            k2y = h * (yp + k1z / 2)
            k2z = h * f(x + h / 2, y + k1y / 2, yp + k1z / 2)
            k3y = h * (yp + k2z / 2)
            k3z = h * f(x + h / 2, y + k2y / 2, yp + k2z / 2)
            k4y = h * (yp + k3z)
            k4z = h * f(x + h, y + k3y, yp + k3z)
            y = y + (k1y + 2 * k2y + 2 * k3y + k4y) / 6
            yp = yp + (k1z + 2 * k2z + 2 * k3z + k4z) / 6
            x += h
            xs.append(x)
            ys.append(y)
        return xs, ys

    # Secant method on the shooting function F(s) = y(b; s) - beta
    s0 = 0.0
    s1 = (beta - alpha) / (b - a)
    _, ys0 = integrate(s0)
    F0 = ys0[-1] - beta
    for _ in range(max_iter):
        _, ys1 = integrate(s1)
        F1 = ys1[-1] - beta
        if abs(F1) < tol:
            return integrate(s1)
        if abs(F1 - F0) < 1e-14:
            break
        s_new = s1 - F1 * (s1 - s0) / (F1 - F0)
        s0, F0 = s1, F1
        s1 = s_new
    return integrate(s1)


# ---------------------------------------------------------------------------
# Finite Difference Method
# ---------------------------------------------------------------------------

def finite_difference_bvp(
    p: Callable[[float], float],
    q: Callable[[float], float],
    r: Callable[[float], float],
    a: float,
    b: float,
    alpha: float,
    beta: float,
    n: int = 100
) -> Tuple[List[float], List[float]]:
    """Solve y'' + p(x)y' + q(x)y = r(x), y(a)=alpha, y(b)=beta via FDM.

    Returns (x_values, y_values).
    """
    h = (b - a) / (n + 1)
    xs = [a + i * h for i in range(n + 2)]

    # Build tridiagonal system
    # Central differences: y''_i ~ (y_{i-1} - 2y_i + y_{i+1}) / h^2
    #                      y'_i  ~ (y_{i+1} - y_{i-1}) / (2h)
    lower = []
    main = []
    upper = []
    rhs = []

    for i in range(1, n + 1):
        x = xs[i]
        pi = p(x)
        qi = q(x)
        ri = r(x)
        a_coeff = 1 / h ** 2 - pi / (2 * h)
        b_coeff = -2 / h ** 2 + qi
        c_coeff = 1 / h ** 2 + pi / (2 * h)
        rhs_val = ri

        if i == 1:
            rhs_val -= a_coeff * alpha
        if i == n:
            rhs_val -= c_coeff * beta

        lower.append(a_coeff if i > 1 else 0.0)
        main.append(b_coeff)
        upper.append(c_coeff if i < n else 0.0)
        rhs.append(rhs_val)

    # Thomas algorithm (tridiagonal solver)
    y_inner = _thomas(lower, main, upper, rhs)
    y_vals = [alpha] + y_inner + [beta]
    return xs, y_vals


def finite_difference_bvp_nonlinear(
    f: Callable[[float, float, float], float],
    df_dy: Callable[[float, float, float], float],
    df_dyp: Callable[[float, float, float], float],
    a: float,
    b: float,
    alpha: float,
    beta: float,
    n: int = 100,
    tol: float = 1e-8,
    max_iter: int = 50
) -> Tuple[List[float], List[float]]:
    """Solve nonlinear BVP y'' = f(x, y, y') via FDM + Newton iteration.

    Returns (x_values, y_values).
    """
    h = (b - a) / (n + 1)
    xs = [a + i * h for i in range(n + 2)]
    y = [alpha + (beta - alpha) * i / (n + 1) for i in range(n + 2)]

    for _ in range(max_iter):
        # Build residual and Jacobian (tridiagonal)
        residual = []
        lower, main, upper = [], [], []
        for i in range(1, n + 1):
            x = xs[i]
            yi = y[i]
            yip = (y[i + 1] - y[i - 1]) / (2 * h)
            yipp = (y[i - 1] - 2 * y[i] + y[i + 1]) / h ** 2
            F_i = yipp - f(x, yi, yip)
            residual.append(-F_i)

            # Jacobian entries
            dF_low = 1 / h ** 2 + df_dyp(x, yi, yip) / (2 * h)  # d/d y_{i-1}
            dF_main = -2 / h ** 2 - df_dy(x, yi, yip)            # d/d y_i
            dF_up = 1 / h ** 2 - df_dyp(x, yi, yip) / (2 * h)   # d/d y_{i+1}

            lower.append(dF_low if i > 1 else 0.0)
            main.append(dF_main)
            upper.append(dF_up if i < n else 0.0)

        delta = _thomas(lower, main, upper, residual)
        for i in range(1, n + 1):
            y[i] += delta[i - 1]
        if max(abs(d) for d in delta) < tol:
            break

    return xs, y


def _thomas(lower: List[float], main: List[float], upper: List[float], rhs: List[float]) -> List[float]:
    """Thomas algorithm for tridiagonal systems."""
    n = len(main)
    c = list(upper)
    d = list(rhs)
    m = list(main)
    l = list(lower)

    for i in range(1, n):
        w = l[i] / m[i - 1] if abs(m[i - 1]) > 1e-14 else 0.0
        m[i] -= w * c[i - 1]
        d[i] -= w * d[i - 1]

    x = [0.0] * n
    x[-1] = d[-1] / m[-1] if abs(m[-1]) > 1e-14 else 0.0
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / m[i] if abs(m[i]) > 1e-14 else 0.0
    return x


# ---------------------------------------------------------------------------
# Finite Element Method (1D, Linear Elements)
# ---------------------------------------------------------------------------

def fem_1d(
    k: Callable[[float], float],
    q: Callable[[float], float],
    f: Callable[[float], float],
    a: float,
    b: float,
    alpha: float,
    beta: float,
    n_elements: int = 50
) -> Tuple[List[float], List[float]]:
    """Solve -(k(x)y')' + q(x)y = f(x), y(a)=alpha, y(b)=beta via FEM.

    Uses linear (P1) Galerkin elements with 2-point Gaussian quadrature.
    Returns (x_values, y_values).
    """
    n_nodes = n_elements + 1
    h = (b - a) / n_elements
    xs = [a + i * h for i in range(n_nodes)]

    # Gaussian quadrature weights and points on [0, 1]
    gp = [(0.5 - 1 / (2 * math.sqrt(3)), 0.5), (0.5 + 1 / (2 * math.sqrt(3)), 0.5)]
    gw = [0.5, 0.5]

    # Assemble stiffness K and load F (interior nodes only)
    n_interior = n_nodes - 2
    K_full = [[0.0] * n_nodes for _ in range(n_nodes)]
    F_full = [0.0] * n_nodes

    for e in range(n_elements):
        x_l, x_r = xs[e], xs[e + 1]
        h_e = x_r - x_l

        for (xi, _), w in zip(gp, gw):
            x = x_l + xi * h_e
            ke = k(x)
            qe = q(x)
            fe = f(x)
            # Shape functions: N1 = 1 - xi, N2 = xi
            N = [1 - xi, xi]
            dN = [-1.0 / h_e, 1.0 / h_e]
            for i in range(2):
                for j in range(2):
                    K_full[e + i][e + j] += w * h_e * (ke * dN[i] * dN[j] + qe * N[i] * N[j])
                F_full[e + i] += w * h_e * fe * N[i]

    # Apply Dirichlet BCs: fix y[0] = alpha, y[-1] = beta
    rhs = []
    for i in range(1, n_nodes - 1):
        rhs.append(F_full[i] - K_full[i][0] * alpha - K_full[i][-1] * beta)

    K_int = [[K_full[i + 1][j + 1] for j in range(n_interior)] for i in range(n_interior)]

    # Solve K_int * y_int = rhs
    y_int = _gauss_solve(K_int, rhs)
    y_vals = [alpha] + y_int + [beta]
    return xs, y_vals


def _gauss_solve(A: List[List[float]], b: List[float]) -> List[float]:
    """Gaussian elimination with partial pivoting."""
    n = len(b)
    aug = [A[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        if abs(aug[col][col]) < 1e-14:
            continue
        for row in range(col + 1, n):
            f = aug[row][col] / aug[col][col]
            aug[row] = [aug[row][c] - f * aug[col][c] for c in range(n + 1)]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (aug[i][n] - sum(aug[i][j] * x[j] for j in range(i + 1, n))) / aug[i][i] if abs(aug[i][i]) > 1e-14 else 0.0
    return x


# ---------------------------------------------------------------------------
# Sturm-Liouville Eigenvalue Problem
# ---------------------------------------------------------------------------

def sturm_liouville_eigenvalues(
    p: Callable[[float], float],
    q: Callable[[float], float],
    w: Callable[[float], float],
    a: float,
    b: float,
    n_eigenvalues: int = 5,
    n_grid: int = 200
) -> Tuple[List[float], List[List[float]]]:
    """Compute eigenvalues/eigenfunctions of -(p y')' + q y = lambda w y.

    Uses the finite difference discretisation + power-shifted inverse iteration.
    Returns (eigenvalues, eigenvectors).
    """
    h = (b - a) / (n_grid + 1)
    xs = [a + i * h for i in range(1, n_grid + 1)]

    # Build generalised eigenvalue problem K v = lambda M v
    # K_ij = stiffness, M_ij = mass (diagonal, using midpoint weights)
    K = [[0.0] * n_grid for _ in range(n_grid)]
    M = [w(xs[i]) * h for i in range(n_grid)]  # Diagonal mass lumping

    for i in range(n_grid):
        x = xs[i]
        p_mid_right = p(x + h / 2) if i < n_grid - 1 else p(b - h / 4)
        p_mid_left = p(x - h / 2) if i > 0 else p(a + h / 4)
        K[i][i] = (p_mid_right + p_mid_left) / h ** 2 + q(x)
        if i > 0:
            K[i][i - 1] = -p_mid_left / h ** 2
            K[i - 1][i] = -p_mid_left / h ** 2

    # Find smallest n_eigenvalues eigenvalues via inverse iteration with shifts
    eigenvalues = []
    eigenvectors = []
    # Approximate eigenvalues using Gershgorin: lambda_k ~ (k*pi/(b-a))^2
    for k in range(1, n_eigenvalues + 1):
        shift = (k * math.pi / (b - a)) ** 2 * 0.9
        # (K - shift*M) v = M v_prev
        K_shifted = [[K[i][j] - (shift * M[i] if i == j else 0.0) for j in range(n_grid)] for i in range(n_grid)]
        v = [math.sin(k * math.pi * (xs[i] - a) / (b - a)) for i in range(n_grid)]
        # Normalise
        norm = math.sqrt(sum(v[i] ** 2 * M[i] for i in range(n_grid)))
        v = [vi / norm for vi in v] if norm > 1e-14 else v
        lam = shift
        for _ in range(100):
            # Mv = M * v
            Mv = [M[i] * v[i] for i in range(n_grid)]
            try:
                v_new = _gauss_solve(K_shifted, Mv)
            except Exception:
                break
            # Rayleigh quotient update
            norm_new = math.sqrt(sum(v_new[i] ** 2 * M[i] for i in range(n_grid)))
            if norm_new < 1e-14:
                break
            v_new = [vi / norm_new for vi in v_new]
            lam_new = shift + norm_new / sum(v_new[i] * Mv[i] for i in range(n_grid)) if sum(v_new[i] * Mv[i] for i in range(n_grid)) != 0 else lam
            if abs(lam_new - lam) < 1e-10:
                lam = lam_new
                v = v_new
                break
            lam = lam_new
            v = v_new
        eigenvalues.append(lam)
        eigenvectors.append([0.0] + v + [0.0])

    return eigenvalues, eigenvectors


# ---------------------------------------------------------------------------
# Chebyshev Collocation
# ---------------------------------------------------------------------------

def chebyshev_collocation(
    p: Callable[[float], float],
    q: Callable[[float], float],
    r: Callable[[float], float],
    a: float,
    b: float,
    alpha: float,
    beta: float,
    n: int = 20
) -> Tuple[List[float], List[float]]:
    """Solve y'' + p(x)y' + q(x)y = r(x) via Chebyshev pseudospectral collocation.

    Returns (x_values, y_values) at Chebyshev nodes.
    """
    # Chebyshev nodes on [-1, 1]
    theta = [math.pi * k / n for k in range(n + 1)]
    xi = [math.cos(th) for th in theta]  # xi[0]=1, xi[n]=-1

    # Map from [a,b] to [-1,1]: x = (b+a)/2 + (b-a)/2 * t
    mid = (b + a) / 2
    half = (b - a) / 2

    # Chebyshev differentiation matrix D on [-1,1]
    D = _chebyshev_D(n)
    D2 = _mat_mul_2d(D, D)

    # Map derivatives: dy/dx = (1/half)*dy/dt, d2y/dx2 = (1/half^2)*d2y/dt2
    scale1 = 1.0 / half
    scale2 = 1.0 / half ** 2

    # Interior nodes: indices 1..n-1
    n_int = n - 1
    # System: (scale2*D2 + diag(p_i)*scale1*D + diag(q_i)) y_int = r_i - BC_contributions
    # Boundary: y[0] = beta (t=+1 -> x=b), y[n] = alpha (t=-1 -> x=a)
    # Note: xi[0]=1 -> x=b, xi[n]=-1 -> x=a

    xs = [mid + half * xi[k] for k in range(n + 1)]
    p_vals = [p(xs[k]) for k in range(n + 1)]
    q_vals = [q(xs[k]) for k in range(n + 1)]
    r_vals = [r(xs[k]) for k in range(n + 1)]

    # Build system for interior nodes 1..n-1
    A_mat = []
    rhs = []
    for i in range(1, n):
        row = []
        for j in range(1, n):
            val = scale2 * D2[i][j] + scale1 * p_vals[i] * D[i][j] + (q_vals[i] if i == j else 0.0)
            row.append(val)
        # RHS: r_i - boundary terms
        bc = (scale2 * D2[i][0] + scale1 * p_vals[i] * D[i][0]) * beta + \
             (scale2 * D2[i][n] + scale1 * p_vals[i] * D[i][n]) * alpha
        rhs.append(r_vals[i] - bc - q_vals[i] * 0)  # q term already in diagonal
        A_mat.append(row)

    y_int = _gauss_solve(A_mat, rhs)
    y_full = [beta] + y_int + [alpha]
    return xs, y_full


def _chebyshev_D(n: int) -> List[List[float]]:
    """Chebyshev spectral differentiation matrix of size (n+1) x (n+1)."""
    xi = [math.cos(math.pi * k / n) for k in range(n + 1)]
    D = [[0.0] * (n + 1) for _ in range(n + 1)]

    def c(k):
        return 2.0 if k == 0 or k == n else 1.0

    for i in range(n + 1):
        for j in range(n + 1):
            if i != j:
                D[i][j] = c(i) / c(j) * (-1) ** (i + j) / (xi[i] - xi[j]) if abs(xi[i] - xi[j]) > 1e-14 else 0.0
        D[i][i] = 0.0  # Set by row sum = 0

    for i in range(n + 1):
        D[i][i] = -sum(D[i][j] for j in range(n + 1) if j != i)
    return D


def _mat_mul_2d(A, B):
    n = len(A)
    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)] for i in range(n)]


# ---------------------------------------------------------------------------
# Green's Function Method (Simple BVP)
# ---------------------------------------------------------------------------

def greens_function_bvp(
    r: Callable[[float], float],
    a: float,
    b: float,
    alpha: float,
    beta: float,
    n: int = 1000
) -> Tuple[List[float], List[float]]:
    """Solve y'' = r(x), y(a)=alpha, y(b)=beta via Green's function.

    The particular solution is integrated numerically.
    Returns (x_values, y_values).
    """
    h = (b - a) / n
    xs = [a + i * h for i in range(n + 1)]

    # Homogeneous solution: y_h = alpha*(b-x)/(b-a) + beta*(x-a)/(b-a)
    y_h = [alpha * (b - xs[i]) / (b - a) + beta * (xs[i] - a) / (b - a) for i in range(n + 1)]

    # Particular solution via Green's function: y_p(x) = integral G(x,s)*r(s) ds
    # G(x,s) = (s-a)*(b-x)/(b-a) for s <= x, (x-a)*(b-s)/(b-a) for s > x
    y_p = []
    for i, x in enumerate(xs):
        # Numerical integration using trapezoidal rule
        g_vals = []
        for j, s in enumerate(xs):
            if s <= x:
                g = (s - a) * (b - x) / (b - a)
            else:
                g = (x - a) * (b - s) / (b - a)
            g_vals.append(g * r(s))
        # Trapezoidal rule
        integral = h * (g_vals[0] / 2 + sum(g_vals[1:-1]) + g_vals[-1] / 2)
        y_p.append(integral)

    y_vals = [y_h[i] + y_p[i] for i in range(n + 1)]
    return xs, y_vals
