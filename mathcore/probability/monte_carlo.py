"""
Monte Carlo Methods — from scratch, zero external dependencies.
Includes: MC integration, importance sampling, rejection sampling,
bootstrap, Latin hypercube, option pricing, variance reduction.
"""

import math
import random
from typing import List, Tuple, Callable, Optional


# ---------------------------------------------------------------------------
# Monte Carlo Integration
# ---------------------------------------------------------------------------

def monte_carlo_integrate(
    f: Callable[[float], float],
    a: float,
    b: float,
    n_samples: int = 100000
) -> Tuple[float, float]:
    """Estimate integral of f over [a, b] by plain Monte Carlo.

    Returns (estimate, standard_error).
    """
    samples = [f(random.uniform(a, b)) for _ in range(n_samples)]
    mean = sum(samples) / n_samples
    variance = sum((s - mean) ** 2 for s in samples) / (n_samples - 1)
    se = math.sqrt(variance / n_samples)
    return (b - a) * mean, (b - a) * se


def monte_carlo_integrate_nd(
    f: Callable[[List[float]], float],
    bounds: List[Tuple[float, float]],
    n_samples: int = 100000
) -> Tuple[float, float]:
    """Multi-dimensional Monte Carlo integration over a hyper-rectangle.

    Returns (estimate, standard_error).
    """
    d = len(bounds)
    volume = 1.0
    for a, b in bounds:
        volume *= (b - a)
    samples = [f([random.uniform(bounds[k][0], bounds[k][1]) for k in range(d)])
               for _ in range(n_samples)]
    mean = sum(samples) / n_samples
    variance = sum((s - mean) ** 2 for s in samples) / (n_samples - 1)
    se = math.sqrt(variance / n_samples)
    return volume * mean, volume * se


def quasi_monte_carlo_halton(
    f: Callable[[List[float]], float],
    bounds: List[Tuple[float, float]],
    n_samples: int = 10000
) -> float:
    """Quasi-Monte Carlo integration using Halton low-discrepancy sequences."""
    d = len(bounds)
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][:d]

    def halton(index: int, base: int) -> float:
        result, f_val = 0.0, 1.0
        while index > 0:
            f_val /= base
            result += f_val * (index % base)
            index //= base
        return result

    volume = 1.0
    for a, b in bounds:
        volume *= (b - a)

    total = 0.0
    for i in range(1, n_samples + 1):
        point = [bounds[k][0] + (bounds[k][1] - bounds[k][0]) * halton(i, primes[k])
                 for k in range(d)]
        total += f(point)
    return volume * total / n_samples


# ---------------------------------------------------------------------------
# Importance Sampling
# ---------------------------------------------------------------------------

def importance_sampling(
    f: Callable[[float], float],
    proposal_sampler: Callable[[], float],
    proposal_pdf: Callable[[float], float],
    target_pdf: Callable[[float], float],
    n_samples: int = 100000
) -> Tuple[float, float]:
    """Importance sampling estimator: E[f(X)] where X ~ target.

    Returns (estimate, standard_error).
    Samples from proposal, reweights by target/proposal.
    """
    weighted_vals = []
    for _ in range(n_samples):
        x = proposal_sampler()
        w = target_pdf(x) / max(proposal_pdf(x), 1e-300)
        weighted_vals.append(f(x) * w)
    mean = sum(weighted_vals) / n_samples
    variance = sum((v - mean) ** 2 for v in weighted_vals) / (n_samples - 1)
    return mean, math.sqrt(variance / n_samples)


# ---------------------------------------------------------------------------
# Rejection Sampling
# ---------------------------------------------------------------------------

def rejection_sampling(
    target_pdf: Callable[[float], float],
    proposal_sampler: Callable[[], float],
    proposal_pdf: Callable[[float], float],
    M: float,
    n_samples: int = 10000
) -> List[float]:
    """Rejection sampling from target using proposal envelope M*q(x) >= p(x).

    Returns list of n_samples accepted samples.
    """
    samples = []
    while len(samples) < n_samples:
        x = proposal_sampler()
        u = random.uniform(0, M * proposal_pdf(x))
        if u <= target_pdf(x):
            samples.append(x)
    return samples


def slice_sampling(
    log_target: Callable[[float], float],
    initial: float,
    n_samples: int = 10000,
    width: float = 1.0,
    burn_in: int = 500
) -> List[float]:
    """Univariate slice sampler. Returns n_samples samples after burn-in."""
    x = initial
    samples = []
    for i in range(n_samples + burn_in):
        # Sample uniformly below the current function value
        log_y = log_target(x) + math.log(random.random() + 1e-300)
        # Find a stepping-out interval [L, R]
        L = x - random.uniform(0, width)
        R = L + width
        # Shrinkage sampling
        while True:
            x_new = random.uniform(L, R)
            if log_target(x_new) >= log_y:
                x = x_new
                break
            if x_new < x:
                L = x_new
            else:
                R = x_new
        if i >= burn_in:
            samples.append(x)
    return samples


# ---------------------------------------------------------------------------
# Bootstrap Resampling
# ---------------------------------------------------------------------------

def bootstrap(
    data: List[float],
    statistic: Callable[[List[float]], float],
    n_bootstrap: int = 10000
) -> Tuple[float, float, Tuple[float, float]]:
    """Non-parametric bootstrap.

    Returns (mean_bootstrap_stat, std_bootstrap_stat, 95%_CI).
    """
    n = len(data)
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = [data[random.randint(0, n - 1)] for _ in range(n)]
        boot_stats.append(statistic(sample))
    mean_stat = sum(boot_stats) / n_bootstrap
    std_stat = math.sqrt(sum((s - mean_stat) ** 2 for s in boot_stats) / (n_bootstrap - 1))
    sorted_stats = sorted(boot_stats)
    ci_lo = sorted_stats[int(0.025 * n_bootstrap)]
    ci_hi = sorted_stats[int(0.975 * n_bootstrap)]
    return mean_stat, std_stat, (ci_lo, ci_hi)


def parametric_bootstrap(
    sampler: Callable[[int], List[float]],
    statistic: Callable[[List[float]], float],
    n: int,
    n_bootstrap: int = 10000
) -> Tuple[float, float, Tuple[float, float]]:
    """Parametric bootstrap: sample from a parametric distribution.

    Returns (mean_stat, std_stat, 95%_CI).
    """
    boot_stats = [statistic(sampler(n)) for _ in range(n_bootstrap)]
    mean_stat = sum(boot_stats) / n_bootstrap
    std_stat = math.sqrt(sum((s - mean_stat) ** 2 for s in boot_stats) / (n_bootstrap - 1))
    sorted_stats = sorted(boot_stats)
    return (mean_stat, std_stat,
            (sorted_stats[int(0.025 * n_bootstrap)], sorted_stats[int(0.975 * n_bootstrap)]))


# ---------------------------------------------------------------------------
# Latin Hypercube Sampling
# ---------------------------------------------------------------------------

def latin_hypercube_sample(
    d: int,
    n: int,
    bounds: Optional[List[Tuple[float, float]]] = None
) -> List[List[float]]:
    """Generate a Latin Hypercube Sample of n points in d dimensions.

    Each dimension is stratified into n equal intervals.
    Returns list of n points, each of length d.
    """
    if bounds is None:
        bounds = [(0.0, 1.0)] * d
    sample = []
    for j in range(d):
        a, b = bounds[j]
        # One point per stratum
        strata = [(a + (b - a) * (i + random.random()) / n) for i in range(n)]
        random.shuffle(strata)
        sample.append(strata)
    return [[sample[j][i] for j in range(d)] for i in range(n)]


# ---------------------------------------------------------------------------
# Variance Reduction
# ---------------------------------------------------------------------------

def antithetic_variates(
    f: Callable[[float], float],
    a: float,
    b: float,
    n_samples: int = 50000
) -> Tuple[float, float]:
    """Antithetic variates estimator for integral of f over [a, b].

    Pairs U and 1-U to reduce variance. Returns (estimate, se).
    """
    estimates = []
    for _ in range(n_samples):
        u = random.random()
        x1 = a + (b - a) * u
        x2 = a + (b - a) * (1 - u)
        estimates.append((f(x1) + f(x2)) / 2)
    mean = sum(estimates) / n_samples
    variance = sum((e - mean) ** 2 for e in estimates) / (n_samples - 1)
    return (b - a) * mean, (b - a) * math.sqrt(variance / n_samples)


def control_variates(
    f: Callable[[float], float],
    g: Callable[[float], float],
    g_mean: float,
    a: float,
    b: float,
    n_samples: int = 50000
) -> Tuple[float, float]:
    """Control variates estimator. g is a control variate with known mean g_mean.

    Estimates E[f(X)] using f_hat = f(X) - c*(g(X) - E[g(X)]).
    Returns (estimate, se).
    """
    fs, gs = [], []
    for _ in range(n_samples):
        x = random.uniform(a, b)
        fs.append(f(x))
        gs.append(g(x))
    mean_f = sum(fs) / n_samples
    mean_g = sum(gs) / n_samples
    cov = sum((fs[i] - mean_f) * (gs[i] - mean_g) for i in range(n_samples)) / (n_samples - 1)
    var_g = sum((gi - mean_g) ** 2 for gi in gs) / (n_samples - 1)
    c = cov / var_g if var_g > 0 else 0.0
    adjusted = [fs[i] - c * (gs[i] - g_mean) for i in range(n_samples)]
    mean_adj = sum(adjusted) / n_samples
    var_adj = sum((v - mean_adj) ** 2 for v in adjusted) / (n_samples - 1)
    return (b - a) * mean_adj, (b - a) * math.sqrt(var_adj / n_samples)


def stratified_sampling(
    f: Callable[[float], float],
    a: float,
    b: float,
    n_strata: int = 10,
    n_per_stratum: int = 1000
) -> Tuple[float, float]:
    """Stratified sampling estimator.

    Returns (estimate, se).
    """
    width = (b - a) / n_strata
    estimates = []
    for k in range(n_strata):
        lo = a + k * width
        hi = lo + width
        stratum_samples = [f(random.uniform(lo, hi)) for _ in range(n_per_stratum)]
        mean_k = sum(stratum_samples) / n_per_stratum
        estimates.append(width * mean_k)
    total = sum(estimates)
    # SE from individual stratum estimates
    overall_mean = total / (b - a)  # per-unit mean
    var_estimates = sum((e / width - overall_mean) ** 2 for e in estimates) / (n_strata * (n_strata - 1))
    se = (b - a) * math.sqrt(var_estimates / n_strata)
    return total, se


# ---------------------------------------------------------------------------
# Financial Monte Carlo
# ---------------------------------------------------------------------------

def black_scholes_mc(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    n_paths: int = 100000
) -> Tuple[float, float]:
    """Price a European option via Monte Carlo (GBM paths).

    Returns (price, standard_error).
    """
    payoffs = []
    for _ in range(n_paths):
        Z = random.gauss(0, 1)
        ST = S0 * math.exp((r - 0.5 * sigma ** 2) * T + sigma * math.sqrt(T) * Z)
        if option_type == "call":
            payoffs.append(max(ST - K, 0))
        else:
            payoffs.append(max(K - ST, 0))
    discount = math.exp(-r * T)
    mean_payoff = sum(payoffs) / n_paths
    std_payoff = math.sqrt(sum((p - mean_payoff) ** 2 for p in payoffs) / (n_paths - 1))
    return discount * mean_payoff, discount * std_payoff / math.sqrt(n_paths)


def black_scholes_analytic(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call"
) -> float:
    """Analytic Black-Scholes price for European option."""
    def _norm_cdf(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    d1 = (math.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        return S0 * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S0 * _norm_cdf(-d1)


def asian_option_mc(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int = 252,
    n_paths: int = 50000,
    average: str = "arithmetic"
) -> Tuple[float, float]:
    """Price an Asian call option via Monte Carlo. Returns (price, se)."""
    dt = T / n_steps
    payoffs = []
    for _ in range(n_paths):
        S = S0
        path = [S]
        for _ in range(n_steps):
            Z = random.gauss(0, 1)
            S *= math.exp((r - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * Z)
            path.append(S)
        if average == "arithmetic":
            avg = sum(path) / len(path)
        else:
            log_sum = sum(math.log(s) for s in path)
            avg = math.exp(log_sum / len(path))
        payoffs.append(max(avg - K, 0))
    discount = math.exp(-r * T)
    mean_p = sum(payoffs) / n_paths
    std_p = math.sqrt(sum((p - mean_p) ** 2 for p in payoffs) / (n_paths - 1))
    return discount * mean_p, discount * std_p / math.sqrt(n_paths)


def value_at_risk(
    returns: List[float],
    confidence: float = 0.95,
    method: str = "historical"
) -> float:
    """Estimate Value at Risk (VaR) at given confidence level.

    method: 'historical' or 'parametric'
    """
    if method == "historical":
        sorted_returns = sorted(returns)
        idx = int((1 - confidence) * len(sorted_returns))
        return -sorted_returns[idx]
    else:
        n = len(returns)
        mean = sum(returns) / n
        std = math.sqrt(sum((r - mean) ** 2 for r in returns) / (n - 1))
        # z-score for confidence level
        z = _probit(confidence)
        return -(mean - z * std)


def _probit(p: float) -> float:
    """Inverse normal CDF (probit function)."""
    if p <= 0 or p >= 1:
        raise ValueError("p must be in (0,1)")
    t = math.sqrt(-2 * math.log(min(p, 1 - p)))
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]
    approx = t - (c[0] + c[1] * t + c[2] * t ** 2) / (1 + d[0] * t + d[1] * t ** 2 + d[2] * t ** 3)
    return approx if p >= 0.5 else -approx


def expected_shortfall(
    returns: List[float],
    confidence: float = 0.95
) -> float:
    """Conditional Value at Risk (CVaR / Expected Shortfall)."""
    sorted_returns = sorted(returns)
    cutoff_idx = int((1 - confidence) * len(sorted_returns))
    tail = sorted_returns[:cutoff_idx + 1]
    return -sum(tail) / len(tail) if tail else 0.0


# ---------------------------------------------------------------------------
# Simulation Utilities
# ---------------------------------------------------------------------------

def simulate_pi(n_samples: int = 1000000) -> float:
    """Estimate pi via Monte Carlo (dartboard method)."""
    inside = sum(1 for _ in range(n_samples) if random.random() ** 2 + random.random() ** 2 <= 1)
    return 4 * inside / n_samples


def simulate_birthday_problem(n_people: int, n_simulations: int = 10000) -> float:
    """Estimate probability of a shared birthday in a group of n_people."""
    shared = 0
    for _ in range(n_simulations):
        birthdays = [random.randint(0, 364) for _ in range(n_people)]
        if len(set(birthdays)) < n_people:
            shared += 1
    return shared / n_simulations
