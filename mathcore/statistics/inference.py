"""
Statistical Inference — from scratch, zero external dependencies.
Includes: t-tests, ANOVA, chi-squared, Mann-Whitney, Kolmogorov-Smirnov,
confidence intervals, power analysis, effect sizes, Bayesian inference.
"""

import math
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Helper: Numerical CDF approximations
# ---------------------------------------------------------------------------

def _erf(x: float) -> float:
    """Error function via Horner's method (Abramowitz & Stegun 7.1.26)."""
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1 / (1 + 0.3275911 * x)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
                + t * (-1.453152027 + t * 1.061405429))))
    return sign * (1 - poly * math.exp(-x * x))


def _normal_cdf(x: float) -> float:
    return 0.5 * (1 + _erf(x / math.sqrt(2)))


def _normal_ppf(p: float) -> float:
    """Inverse normal CDF (probit) via rational approximation."""
    if p <= 0 or p >= 1:
        raise ValueError("p must be in (0, 1)")
    if p < 0.5:
        return -_rational_ppf(math.sqrt(-2 * math.log(p)))
    return _rational_ppf(math.sqrt(-2 * math.log(1 - p)))


def _rational_ppf(t: float) -> float:
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]
    return t - (c[0] + c[1] * t + c[2] * t ** 2) / (1 + d[0] * t + d[1] * t ** 2 + d[2] * t ** 3)


def _t_cdf(t: float, df: int) -> float:
    """CDF of Student's t-distribution via regularised incomplete beta."""
    x = df / (df + t * t)
    # Regularised incomplete beta I(x; df/2, 1/2) via continued fraction
    half_df = df / 2.0
    ib = _regularised_incomplete_beta(x, half_df, 0.5)
    if t >= 0:
        return 1 - 0.5 * ib
    return 0.5 * ib


def _regularised_incomplete_beta(x: float, a: float, b: float, max_iter: int = 200) -> float:
    """Regularised incomplete beta I(x;a,b) via continued fraction (Lentz)."""
    if x < 0 or x > 1:
        raise ValueError("x must be in [0,1]")
    if x == 0:
        return 0.0
    if x == 1:
        return 1.0
    # Use symmetry relation when x > (a+1)/(a+b+2)
    if x > (a + 1) / (a + b + 2):
        return 1 - _regularised_incomplete_beta(1 - x, b, a, max_iter)
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    # Lentz's continued fraction
    tiny = 1e-300
    f = tiny
    C, D = f, 0.0
    for m in range(max_iter):
        for num_idx in range(2):
            if num_idx == 0:
                if m == 0:
                    d = 1.0
                else:
                    d = -(a + m - 1) * (a + b + m - 1) * x / ((a + 2 * m - 2) * (a + 2 * m - 1))
            else:
                d = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
            D = 1 + d * D
            if abs(D) < tiny:
                D = tiny
            C = 1 + d / C
            if abs(C) < tiny:
                C = tiny
            D = 1 / D
            delta = C * D
            f *= delta
            if abs(delta - 1) < 1e-10:
                return front * f
    return front * f


def _chi2_cdf(x: float, k: float) -> float:
    """CDF of chi-squared distribution with k degrees of freedom."""
    if x <= 0:
        return 0.0
    return _regularised_lower_gamma(k / 2, x / 2)


def _regularised_lower_gamma(a: float, x: float, max_iter: int = 300) -> float:
    """Regularised lower incomplete gamma P(a, x) via series expansion."""
    if x < 0:
        return 0.0
    if x == 0:
        return 0.0
    log_p = a * math.log(x) - x - math.lgamma(a)
    term = 1.0 / a
    total = term
    for n in range(1, max_iter):
        term *= x / (a + n)
        total += term
        if abs(term) < 1e-14 * abs(total):
            break
    return min(math.exp(log_p) * total, 1.0)


def _f_cdf(f: float, d1: float, d2: float) -> float:
    """CDF of F-distribution with d1, d2 degrees of freedom."""
    if f <= 0:
        return 0.0
    x = d1 * f / (d1 * f + d2)
    return _regularised_incomplete_beta(x, d1 / 2, d2 / 2)


# ---------------------------------------------------------------------------
# One-sample t-test
# ---------------------------------------------------------------------------

def ttest_one_sample(
    data: List[float],
    popmean: float = 0.0,
    alternative: str = "two-sided"
) -> Tuple[float, float]:
    """One-sample t-test: H0: mean(data) == popmean.

    Returns (t_statistic, p_value).
    alternative: 'two-sided', 'greater', 'less'
    """
    n = len(data)
    if n < 2:
        raise ValueError("Need at least 2 observations")
    mean = sum(data) / n
    var = sum((x - mean) ** 2 for x in data) / (n - 1)
    se = math.sqrt(var / n)
    if se == 0:
        raise ValueError("Standard error is zero — all values identical")
    t = (mean - popmean) / se
    p = _t_pvalue(t, n - 1, alternative)
    return t, p


def _t_pvalue(t: float, df: int, alternative: str) -> float:
    if alternative == "two-sided":
        return 2 * min(_t_cdf(t, df), 1 - _t_cdf(t, df))
    elif alternative == "greater":
        return 1 - _t_cdf(t, df)
    elif alternative == "less":
        return _t_cdf(t, df)
    raise ValueError(f"Unknown alternative: {alternative}")


# ---------------------------------------------------------------------------
# Two-sample t-tests
# ---------------------------------------------------------------------------

def ttest_independent(
    a: List[float],
    b: List[float],
    equal_var: bool = True,
    alternative: str = "two-sided"
) -> Tuple[float, float]:
    """Two-sample independent t-test.

    equal_var=True  => pooled (Student's t)
    equal_var=False => Welch's t
    Returns (t_statistic, p_value).
    """
    na, nb = len(a), len(b)
    mean_a = sum(a) / na
    mean_b = sum(b) / nb
    var_a = sum((x - mean_a) ** 2 for x in a) / (na - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (nb - 1)
    if equal_var:
        sp2 = ((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2)
        se = math.sqrt(sp2 * (1 / na + 1 / nb))
        df = na + nb - 2
    else:
        se = math.sqrt(var_a / na + var_b / nb)
        # Welch-Satterthwaite df
        num = (var_a / na + var_b / nb) ** 2
        den = (var_a / na) ** 2 / (na - 1) + (var_b / nb) ** 2 / (nb - 1)
        df = int(num / den)
    t = (mean_a - mean_b) / se
    p = _t_pvalue(t, df, alternative)
    return t, p


def ttest_paired(
    a: List[float],
    b: List[float],
    alternative: str = "two-sided"
) -> Tuple[float, float]:
    """Paired t-test. Returns (t_statistic, p_value)."""
    if len(a) != len(b):
        raise ValueError("Paired sequences must have equal length")
    diffs = [x - y for x, y in zip(a, b)]
    return ttest_one_sample(diffs, 0.0, alternative)


# ---------------------------------------------------------------------------
# ANOVA
# ---------------------------------------------------------------------------

def one_way_anova(*groups: List[float]) -> Tuple[float, float]:
    """One-way ANOVA (Fisher). Returns (F_statistic, p_value)."""
    k = len(groups)
    if k < 2:
        raise ValueError("Need at least 2 groups")
    n_total = sum(len(g) for g in groups)
    grand_mean = sum(sum(g) for g in groups) / n_total
    # Between-group sum of squares
    ss_between = sum(len(g) * (sum(g) / len(g) - grand_mean) ** 2 for g in groups)
    # Within-group sum of squares
    ss_within = sum(
        sum((x - sum(g) / len(g)) ** 2 for x in g) for g in groups
    )
    df_between = k - 1
    df_within = n_total - k
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    if ms_within == 0:
        raise ValueError("Within-group variance is zero")
    F = ms_between / ms_within
    p = 1 - _f_cdf(F, df_between, df_within)
    return F, p


def two_way_anova(
    data: List[List[List[float]]],
) -> dict:
    """Two-way ANOVA with replication.

    data[i][j] = list of observations for factor A level i, factor B level j.
    Returns dict with F/p for main effects A, B and interaction AB.
    """
    a = len(data)
    b = len(data[0])
    n = len(data[0][0])  # replications per cell (must be equal)
    N = a * b * n
    grand_total = sum(data[i][j][k] for i in range(a) for j in range(b) for k in range(n))
    grand_mean = grand_total / N
    cf = grand_total ** 2 / N  # correction factor

    ss_total = sum(data[i][j][k] ** 2 for i in range(a) for j in range(b) for k in range(n)) - cf

    row_totals = [sum(data[i][j][k] for j in range(b) for k in range(n)) for i in range(a)]
    col_totals = [sum(data[i][j][k] for i in range(a) for k in range(n)) for j in range(b)]
    cell_totals = [[sum(data[i][j]) for j in range(b)] for i in range(a)]

    ss_a = sum(t ** 2 for t in row_totals) / (b * n) - cf
    ss_b = sum(t ** 2 for t in col_totals) / (a * n) - cf
    ss_ab = sum(cell_totals[i][j] ** 2 for i in range(a) for j in range(b)) / n - cf - ss_a - ss_b
    ss_error = ss_total - ss_a - ss_b - ss_ab

    df_a, df_b, df_ab, df_e = a - 1, b - 1, (a - 1) * (b - 1), a * b * (n - 1)
    ms_a, ms_b, ms_ab = ss_a / df_a, ss_b / df_b, ss_ab / df_ab
    ms_e = ss_error / df_e if df_e > 0 else 1

    Fa, Fb, Fab = ms_a / ms_e, ms_b / ms_e, ms_ab / ms_e
    return {
        "F_A": Fa, "p_A": 1 - _f_cdf(Fa, df_a, df_e),
        "F_B": Fb, "p_B": 1 - _f_cdf(Fb, df_b, df_e),
        "F_AB": Fab, "p_AB": 1 - _f_cdf(Fab, df_ab, df_e),
        "ss_total": ss_total, "ss_error": ss_error,
    }


# ---------------------------------------------------------------------------
# Chi-squared Test
# ---------------------------------------------------------------------------

def chi2_goodness_of_fit(
    observed: List[float],
    expected: List[float]
) -> Tuple[float, float]:
    """Chi-squared goodness-of-fit test. Returns (chi2_stat, p_value)."""
    if len(observed) != len(expected):
        raise ValueError("observed and expected must have equal length")
    chi2 = sum((o - e) ** 2 / e for o, e in zip(observed, expected) if e > 0)
    df = len(observed) - 1
    p = 1 - _chi2_cdf(chi2, df)
    return chi2, p


def chi2_independence(contingency: List[List[float]]) -> Tuple[float, float, int]:
    """Chi-squared test of independence for a contingency table.

    Returns (chi2_stat, p_value, df).
    """
    r, c = len(contingency), len(contingency[0])
    row_sums = [sum(contingency[i]) for i in range(r)]
    col_sums = [sum(contingency[i][j] for i in range(r)) for j in range(c)]
    n = sum(row_sums)
    chi2 = 0.0
    for i in range(r):
        for j in range(c):
            expected = row_sums[i] * col_sums[j] / n
            if expected > 0:
                chi2 += (contingency[i][j] - expected) ** 2 / expected
    df = (r - 1) * (c - 1)
    p = 1 - _chi2_cdf(chi2, df)
    return chi2, p, df


# ---------------------------------------------------------------------------
# Non-parametric Tests
# ---------------------------------------------------------------------------

def mann_whitney_u(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Mann-Whitney U test (two-sided). Returns (U_statistic, p_value)."""
    na, nb = len(a), len(b)
    U = sum(1 for x in a for y in b if x > y) + 0.5 * sum(1 for x in a for y in b if x == y)
    mu = na * nb / 2
    sigma = math.sqrt(na * nb * (na + nb + 1) / 12)
    z = (U - mu) / sigma
    p = 2 * (1 - _normal_cdf(abs(z)))
    return U, p


def wilcoxon_signed_rank(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Wilcoxon signed-rank test for paired samples. Returns (W, p_value)."""
    diffs = [x - y for x, y in zip(a, b) if x != y]
    n = len(diffs)
    abs_diffs = sorted(abs(d) for d in diffs)
    ranks = {}
    i = 0
    while i < n:
        val = abs_diffs[i]
        j = i
        while j < n and abs_diffs[j] == val:
            j += 1
        avg_rank = (i + j + 1) / 2.0
        ranks[val] = avg_rank
        i = j
    W_plus = sum(ranks[abs(d)] for d in diffs if d > 0)
    W_minus = sum(ranks[abs(d)] for d in diffs if d < 0)
    W = min(W_plus, W_minus)
    mu_w = n * (n + 1) / 4
    sigma_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (W - mu_w) / sigma_w
    p = 2 * (1 - _normal_cdf(abs(z)))
    return W, p


def kruskal_wallis(*groups: List[float]) -> Tuple[float, float]:
    """Kruskal-Wallis H-test (non-parametric ANOVA). Returns (H, p_value)."""
    k = len(groups)
    all_data = [(x, i) for i, g in enumerate(groups) for x in g]
    all_data.sort(key=lambda t: t[0])
    n = len(all_data)
    # Assign ranks (average for ties)
    ranks = [0.0] * n
    i = 0
    while i < n:
        val = all_data[i][0]
        j = i
        while j < n and all_data[j][0] == val:
            j += 1
        avg_rank = (i + j + 1) / 2.0
        for idx in range(i, j):
            ranks[idx] = avg_rank
        i = j
    # Group rank sums
    group_ranks = [[] for _ in range(k)]
    for idx, (_, g) in enumerate(all_data):
        group_ranks[g].append(ranks[idx])
    H = (12 / (n * (n + 1))) * sum(
        len(gr) * (sum(gr) / len(gr)) ** 2 for gr in group_ranks
    ) - 3 * (n + 1)
    p = 1 - _chi2_cdf(H, k - 1)
    return H, p


def kolmogorov_smirnov_one_sample(
    data: List[float],
    cdf: callable
) -> Tuple[float, float]:
    """One-sample Kolmogorov-Smirnov test. Returns (D_statistic, p_value)."""
    n = len(data)
    sorted_data = sorted(data)
    D = 0.0
    for i, x in enumerate(sorted_data):
        ecdf_above = (i + 1) / n
        ecdf_below = i / n
        theoretical = cdf(x)
        D = max(D, abs(ecdf_above - theoretical), abs(ecdf_below - theoretical))
    # Approximated p-value via Kolmogorov distribution
    t = D * (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n))
    p = 2 * sum(
        (-1) ** (k - 1) * math.exp(-2 * k * k * t * t)
        for k in range(1, 100)
    )
    p = max(0.0, min(1.0, p))
    return D, p


def kolmogorov_smirnov_two_sample(
    a: List[float],
    b: List[float]
) -> Tuple[float, float]:
    """Two-sample KS test. Returns (D_statistic, p_value)."""
    na, nb = len(a), len(b)
    all_vals = sorted(set(a + b))
    def ecdf_a(x):
        return sum(1 for v in a if v <= x) / na
    def ecdf_b(x):
        return sum(1 for v in b if v <= x) / nb
    D = max(abs(ecdf_a(x) - ecdf_b(x)) for x in all_vals)
    n_eff = math.sqrt(na * nb / (na + nb))
    t = D * (n_eff + 0.12 + 0.11 / n_eff)
    p = 2 * sum((-1) ** (k - 1) * math.exp(-2 * k * k * t * t) for k in range(1, 100))
    p = max(0.0, min(1.0, p))
    return D, p


# ---------------------------------------------------------------------------
# Confidence Intervals
# ---------------------------------------------------------------------------

def confidence_interval_mean(
    data: List[float],
    confidence: float = 0.95
) -> Tuple[float, float]:
    """Confidence interval for the population mean (t-distribution)."""
    n = len(data)
    mean = sum(data) / n
    var = sum((x - mean) ** 2 for x in data) / (n - 1)
    se = math.sqrt(var / n)
    alpha = 1 - confidence
    # Find t_critical via bisection on the t CDF
    t_crit = _t_critical(1 - alpha / 2, n - 1)
    margin = t_crit * se
    return mean - margin, mean + margin


def confidence_interval_proportion(
    successes: int,
    n: int,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    p_hat = successes / n
    z = _normal_ppf(1 - (1 - confidence) / 2)
    denom = 1 + z ** 2 / n
    center = (p_hat + z ** 2 / (2 * n)) / denom
    half_width = z * math.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2)) / denom
    return center - half_width, center + half_width


def _t_critical(p: float, df: int, tol: float = 1e-8) -> float:
    """Inverse t-distribution CDF via bisection."""
    lo, hi = 0.0, 1e6
    for _ in range(100):
        mid = (lo + hi) / 2
        if _t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# Effect Sizes & Power
# ---------------------------------------------------------------------------

def cohens_d(a: List[float], b: List[float]) -> float:
    """Cohen's d effect size for two independent groups."""
    na, nb = len(a), len(b)
    mean_a = sum(a) / na
    mean_b = sum(b) / nb
    var_a = sum((x - mean_a) ** 2 for x in a) / (na - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (nb - 1)
    pooled_sd = math.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    return (mean_a - mean_b) / pooled_sd


def cohens_f(groups: List[List[float]]) -> float:
    """Cohen's f effect size for one-way ANOVA."""
    k = len(groups)
    ns = [len(g) for g in groups]
    n_total = sum(ns)
    means = [sum(g) / len(g) for g in groups]
    grand_mean = sum(sum(g) for g in groups) / n_total
    eta2 = sum(n * (m - grand_mean) ** 2 for n, m in zip(ns, means))
    ss_total = sum((x - grand_mean) ** 2 for g in groups for x in g)
    if ss_total == 0:
        return 0.0
    return math.sqrt(eta2 / (ss_total - eta2))


def power_ttest(
    effect_size: float,
    n: int,
    alpha: float = 0.05,
    alternative: str = "two-sided"
) -> float:
    """Statistical power of a two-sample t-test given effect size d and n per group."""
    df = 2 * n - 2
    t_crit = _t_critical(1 - alpha / (2 if alternative == "two-sided" else 1), df)
    ncp = effect_size * math.sqrt(n / 2)  # non-centrality parameter
    # Power = P(|T| > t_crit | ncp) — approximate via normal
    power = 1 - _normal_cdf(t_crit - ncp) + _normal_cdf(-t_crit - ncp)
    return max(0.0, min(1.0, power))


def sample_size_ttest(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
    alternative: str = "two-sided"
) -> int:
    """Required sample size per group for a two-sample t-test."""
    for n in range(2, 10000):
        if power_ttest(effect_size, n, alpha, alternative) >= power:
            return n
    return 10000


# ---------------------------------------------------------------------------
# Bayesian Inference
# ---------------------------------------------------------------------------

def bayesian_beta_binomial(
    successes: int,
    trials: int,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0
) -> Tuple[float, float, float]:
    """Bayesian update for binomial likelihood with Beta prior.

    Returns (posterior_alpha, posterior_beta, posterior_mean).
    """
    alpha_post = alpha_prior + successes
    beta_post = beta_prior + (trials - successes)
    mean = alpha_post / (alpha_post + beta_post)
    return alpha_post, beta_post, mean


def bayesian_normal_normal(
    data: List[float],
    prior_mean: float = 0.0,
    prior_var: float = 1.0,
    likelihood_var: float = 1.0
) -> Tuple[float, float]:
    """Bayesian update for normal likelihood with known variance and normal prior.

    Returns (posterior_mean, posterior_variance).
    """
    n = len(data)
    sample_mean = sum(data) / n
    post_var = 1 / (1 / prior_var + n / likelihood_var)
    post_mean = post_var * (prior_mean / prior_var + n * sample_mean / likelihood_var)
    return post_mean, post_var


def credible_interval(
    posterior_alpha: float,
    posterior_beta: float,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """HDI credible interval for a Beta(alpha, beta) posterior via bisection."""
    lower_p = (1 - confidence) / 2
    upper_p = 1 - lower_p

    def beta_ppf(p: float) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2
            if _regularised_incomplete_beta(mid, posterior_alpha, posterior_beta) < p:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    return beta_ppf(lower_p), beta_ppf(upper_p)
