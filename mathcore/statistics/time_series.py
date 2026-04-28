"""
Time Series Analysis — from scratch, zero external dependencies.
Includes: moving averages (SMA/EMA/WMA), ACF/PACF, differencing, AR/MA/ARMA,
Holt-Winters, seasonal decomposition, ADF test, spectral density.
"""

import math
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Moving Averages
# ---------------------------------------------------------------------------

def simple_moving_average(x: List[float], window: int) -> List[float]:
    """Simple (equally-weighted) moving average."""
    if window < 1:
        raise ValueError("window must be >= 1")
    return [sum(x[i:i + window]) / window for i in range(len(x) - window + 1)]


def exponential_moving_average(x: List[float], alpha: float) -> List[float]:
    """Exponential moving average with smoothing factor alpha in (0, 1]."""
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    ema = [x[0]]
    for xi in x[1:]:
        ema.append(alpha * xi + (1 - alpha) * ema[-1])
    return ema


def weighted_moving_average(x: List[float], weights: List[float]) -> List[float]:
    """Weighted moving average with custom weights (automatically normalised)."""
    w = len(weights)
    total = sum(weights)
    weights = [wi / total for wi in weights]
    return [sum(x[i + j] * weights[j] for j in range(w)) for i in range(len(x) - w + 1)]


def double_exponential_smoothing(
    x: List[float], alpha: float, beta: float
) -> Tuple[List[float], List[float]]:
    """Holt's double exponential smoothing. Returns (level, trend)."""
    if len(x) < 2:
        raise ValueError("Need at least 2 observations")
    level = [x[0]]
    trend = [x[1] - x[0]]
    for xi in x[1:]:
        l = alpha * xi + (1 - alpha) * (level[-1] + trend[-1])
        t = beta * (l - level[-1]) + (1 - beta) * trend[-1]
        level.append(l)
        trend.append(t)
    return level, trend


def holt_winters(
    x: List[float],
    alpha: float,
    beta: float,
    gamma: float,
    period: int,
    n_forecast: int = 0
) -> Tuple[List[float], List[float]]:
    """Holt-Winters triple exponential smoothing (additive seasonality).

    Returns (smoothed_values, forecast).
    """
    n = len(x)
    if n < period:
        raise ValueError(f"Need at least {period} observations")
    # Initialise
    level = sum(x[:period]) / period
    trend = (sum(x[period:2 * period]) - sum(x[:period])) / period ** 2
    seasonal = [x[i] - level for i in range(period)]
    result = []
    for i in range(n):
        s_idx = i % period
        prev_level, prev_trend = level, trend
        level = alpha * (x[i] - seasonal[s_idx]) + (1 - alpha) * (prev_level + prev_trend)
        trend = beta * (level - prev_level) + (1 - beta) * prev_trend
        seasonal[s_idx] = gamma * (x[i] - level) + (1 - gamma) * seasonal[s_idx]
        result.append(level + trend + seasonal[s_idx])
    forecast = [
        level + (h + 1) * trend + seasonal[(n + h) % period]
        for h in range(n_forecast)
    ]
    return result, forecast


# ---------------------------------------------------------------------------
# Autocorrelation & PACF
# ---------------------------------------------------------------------------

def acf(x: List[float], max_lag: int) -> List[float]:
    """Autocorrelation function at lags 0, 1, ..., max_lag."""
    n = len(x)
    mean = sum(x) / n
    variance = sum((xi - mean) ** 2 for xi in x) / n
    if variance == 0:
        return [1.0] + [0.0] * max_lag
    correlations = [1.0]
    for lag in range(1, max_lag + 1):
        cov = sum((x[i] - mean) * (x[i + lag] - mean) for i in range(n - lag)) / n
        correlations.append(cov / variance)
    return correlations


def pacf(x: List[float], max_lag: int) -> List[float]:
    """Partial autocorrelation function via Yule-Walker equations."""
    acf_vals = acf(x, max_lag)
    result = [1.0, acf_vals[1] if max_lag >= 1 else 1.0]
    phi = [[0.0] * (max_lag + 1) for _ in range(max_lag + 1)]
    phi[1][1] = acf_vals[1]
    for k in range(2, max_lag + 1):
        num = acf_vals[k] - sum(phi[k - 1][j] * acf_vals[k - j] for j in range(1, k))
        den = 1 - sum(phi[k - 1][j] * acf_vals[j] for j in range(1, k))
        phi[k][k] = num / den if abs(den) > 1e-14 else 0.0
        for j in range(1, k):
            phi[k][j] = phi[k - 1][j] - phi[k][k] * phi[k - 1][k - j]
        result.append(phi[k][k])
    return result[:max_lag + 1]


def ljung_box_test(x: List[float], lags: int) -> Tuple[float, float]:
    """Ljung-Box Q test for autocorrelation. Returns (Q_statistic, p_value)."""
    n = len(x)
    acf_vals = acf(x, lags)
    Q = n * (n + 2) * sum(
        acf_vals[k] ** 2 / (n - k) for k in range(1, lags + 1)
    )
    # p-value from chi-squared(lags) distribution
    from mathcore.statistics.inference import _chi2_cdf
    p = 1 - _chi2_cdf(Q, lags)
    return Q, p


# ---------------------------------------------------------------------------
# Differencing & Stationarity
# ---------------------------------------------------------------------------

def difference(x: List[float], d: int = 1) -> List[float]:
    """Apply d-th order differencing to time series x."""
    result = list(x)
    for _ in range(d):
        result = [result[i + 1] - result[i] for i in range(len(result) - 1)]
    return result


def seasonal_difference(x: List[float], period: int) -> List[float]:
    """Seasonal differencing: x[t] - x[t - period]."""
    return [x[i] - x[i - period] for i in range(period, len(x))]


def adf_test(x: List[float], lags: int = 1) -> Tuple[float, float]:
    """Augmented Dickey-Fuller test for unit root.

    H0: unit root (non-stationary).
    Returns (ADF_statistic, approximate_p_value).
    p-value is approximated via MacKinnon (1994) response surface.
    """
    n = len(x)
    # Build regression: Delta_x_t = alpha + beta*x_{t-1} + sum gamma_j * Delta_x_{t-j}
    dy = [x[i + 1] - x[i] for i in range(n - 1)]
    # Response variable: dy[lags:]
    y_reg = dy[lags:]
    # Regressors: intercept, x[lags:-1], and lagged differences
    X_reg = []
    for t in range(lags, len(dy)):
        row = [1.0, x[t]]  # intercept, x_{t-1}
        for j in range(1, lags + 1):
            row.append(dy[t - j])  # lagged differences
        X_reg.append(row)
    if not X_reg:
        raise ValueError("Not enough observations for ADF test")
    # OLS fit
    from mathcore.statistics.regression import OLSRegression, _design_matrix, _solve_ols
    beta = _solve_ols(X_reg, y_reg)
    # ADF statistic = beta[1] / SE(beta[1])
    y_pred = [sum(beta[j] * X_reg[i][j] for j in range(len(beta))) for i in range(len(y_reg))]
    residuals = [y_reg[i] - y_pred[i] for i in range(len(y_reg))]
    n_obs = len(y_reg)
    k = len(beta)
    s2 = sum(r ** 2 for r in residuals) / (n_obs - k)
    # (X^T X)^{-1} — element [1,1] gives variance of beta[1]
    XtX = [[sum(X_reg[i][a] * X_reg[i][b] for i in range(n_obs)) for b in range(k)] for a in range(k)]
    # Use numeric estimate: SE via Cholesky
    from mathcore.statistics.regression import _cholesky, _solve_cholesky
    try:
        L = _cholesky(XtX)
        e1 = [1.0 if j == 1 else 0.0 for j in range(k)]
        inv_col = _solve_cholesky(L, e1)
        e1_back = [1.0 if j == 1 else 0.0 for j in range(k)]
        var_beta1 = sum(inv_col[j] * e1_back[j] for j in range(k)) * s2
        se_beta1 = math.sqrt(abs(var_beta1))
        adf_stat = beta[1] / se_beta1 if se_beta1 > 0 else 0.0
    except Exception:
        adf_stat = 0.0
    # MacKinnon (1994) approximate critical values interpolation
    # Critical values for no-trend model: [-3.43, -2.86, -2.57] at [1%, 5%, 10%]
    # Approximate p-value: linear interpolation heuristic
    cv = [(-3.43, 0.01), (-2.86, 0.05), (-2.57, 0.10)]
    if adf_stat <= cv[0][0]:
        p_val = 0.01
    elif adf_stat >= 0:
        p_val = 0.99
    else:
        for i in range(len(cv) - 1):
            if cv[i][0] >= adf_stat >= cv[i + 1][0]:
                slope = (cv[i + 1][1] - cv[i][1]) / (cv[i + 1][0] - cv[i][0])
                p_val = cv[i][1] + slope * (adf_stat - cv[i][0])
                break
        else:
            p_val = 0.50
    return adf_stat, max(0.01, min(0.99, p_val))


# ---------------------------------------------------------------------------
# AR / MA / ARMA Models
# ---------------------------------------------------------------------------

class ARModel:
    """Autoregressive model AR(p) fitted by Yule-Walker equations."""

    def __init__(self, p: int):
        self.p = p
        self.phi_: Optional[List[float]] = None
        self.intercept_: float = 0.0
        self._x: Optional[List[float]] = None

    def fit(self, x: List[float]) -> "ARModel":
        self._x = x
        n = len(x)
        mean = sum(x) / n
        self.intercept_ = mean
        xc = [xi - mean for xi in x]
        # Yule-Walker: R * phi = r
        acf_vals = acf(xc, self.p)
        # Build Toeplitz correlation matrix
        R = [[acf_vals[abs(i - j)] for j in range(self.p)] for i in range(self.p)]
        r = [acf_vals[k + 1] for k in range(self.p)]
        # Solve via Levinson-Durbin
        self.phi_ = self._levinson_durbin(R, r)
        return self

    def _levinson_durbin(self, R: List[List[float]], r: List[float]) -> List[float]:
        """Solve the Yule-Walker system via Levinson-Durbin recursion."""
        p = len(r)
        phi = [r[0] / R[0][0]] if R[0][0] != 0 else [0.0]
        for k in range(1, p):
            num = r[k] - sum(phi[j] * R[k - 1 - j][0] if k - 1 - j >= 0 else 0 for j in range(k))
            # Simplified: use direct solve for small p
            pass
        # Fallback: Gaussian elimination
        n = len(r)
        aug = [R[i][:] + [r[i]] for i in range(n)]
        for col in range(n):
            pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
            aug[col], aug[pivot] = aug[pivot], aug[col]
            if abs(aug[col][col]) < 1e-14:
                continue
            for row in range(n):
                if row != col:
                    factor = aug[row][col] / aug[col][col]
                    aug[row] = [aug[row][c] - factor * aug[col][c] for c in range(n + 1)]
        return [aug[i][n] / aug[i][i] if abs(aug[i][i]) > 1e-14 else 0.0 for i in range(n)]

    def predict(self, steps: int) -> List[float]:
        """Forecast `steps` periods ahead."""
        history = list(self._x[-self.p:])
        preds = []
        for _ in range(steps):
            val = self.intercept_ + sum(self.phi_[j] * (history[-(j + 1)] - self.intercept_) for j in range(self.p))
            preds.append(val)
            history.append(val)
        return preds

    def fitted_values(self) -> List[float]:
        """In-sample fitted values."""
        x = self._x
        result = list(x[:self.p])
        for t in range(self.p, len(x)):
            val = self.intercept_ + sum(self.phi_[j] * (x[t - j - 1] - self.intercept_) for j in range(self.p))
            result.append(val)
        return result

    def residuals(self) -> List[float]:
        fitted = self.fitted_values()
        return [self._x[i] - fitted[i] for i in range(len(self._x))]

    def aic(self) -> float:
        n = len(self._x)
        res = self.residuals()[self.p:]
        sigma2 = sum(r ** 2 for r in res) / len(res)
        return n * math.log(sigma2 + 1e-15) + 2 * self.p


class MAModel:
    """Moving Average model MA(q) fitted via approximate MLE (iterative)."""

    def __init__(self, q: int, max_iter: int = 200, tol: float = 1e-6):
        self.q = q
        self.max_iter = max_iter
        self.tol = tol
        self.theta_: Optional[List[float]] = None
        self.intercept_: float = 0.0
        self._x: Optional[List[float]] = None

    def fit(self, x: List[float]) -> "MAModel":
        self._x = x
        n = len(x)
        self.intercept_ = sum(x) / n
        # Initialise theta = 0
        theta = [0.0] * self.q
        # Conditional sum-of-squares
        for _ in range(self.max_iter):
            eps = [0.0] * n
            for t in range(n):
                eps[t] = x[t] - self.intercept_ - sum(theta[j] * eps[t - j - 1] for j in range(min(self.q, t)))
            # Gradient step
            grad = [0.0] * self.q
            for j in range(self.q):
                grad[j] = -2 * sum(eps[t] * eps[t - j - 1] for t in range(j + 1, n) if t - j - 1 >= 0)
            step = 1e-4
            theta_new = [theta[j] - step * grad[j] for j in range(self.q)]
            if max(abs(theta_new[j] - theta[j]) for j in range(self.q)) < self.tol:
                theta = theta_new
                break
            theta = theta_new
        self.theta_ = theta
        return self

    def predict(self, steps: int) -> List[float]:
        """Forecast steps periods ahead (errors assumed 0 for future)."""
        x = self._x
        n = len(x)
        eps = [0.0] * n
        for t in range(n):
            eps[t] = x[t] - self.intercept_ - sum(self.theta_[j] * eps[t - j - 1] for j in range(min(self.q, t)))
        preds = []
        for h in range(steps):
            if h == 0:
                preds.append(self.intercept_ + sum(self.theta_[j] * eps[n - j - 1] for j in range(self.q)))
            else:
                preds.append(self.intercept_)
        return preds


class ARMAModel:
    """ARMA(p, q) model via conditional least squares."""

    def __init__(self, p: int, q: int, max_iter: int = 300, tol: float = 1e-6):
        self.p = p
        self.q = q
        self.max_iter = max_iter
        self.tol = tol
        self.phi_: Optional[List[float]] = None
        self.theta_: Optional[List[float]] = None
        self.intercept_: float = 0.0
        self._x: Optional[List[float]] = None

    def fit(self, x: List[float]) -> "ARMAModel":
        self._x = x
        n = len(x)
        self.intercept_ = sum(x) / n
        phi = [0.1] * self.p
        theta = [0.1] * self.q
        for _ in range(self.max_iter):
            eps = [0.0] * n
            for t in range(n):
                ar = sum(phi[j] * (x[t - j - 1] - self.intercept_) for j in range(min(self.p, t)))
                ma = sum(theta[j] * eps[t - j - 1] for j in range(min(self.q, t)))
                eps[t] = x[t] - self.intercept_ - ar - ma
            # Numerical gradient update (finite differences)
            step = 1e-5
            loss = sum(e ** 2 for e in eps)

            def compute_loss(phi_, theta_):
                eps_ = [0.0] * n
                for t in range(n):
                    ar_ = sum(phi_[j] * (x[t - j - 1] - self.intercept_) for j in range(min(self.p, t)))
                    ma_ = sum(theta_[j] * eps_[t - j - 1] for j in range(min(self.q, t)))
                    eps_[t] = x[t] - self.intercept_ - ar_ - ma_
                return sum(e ** 2 for e in eps_)

            phi_new = list(phi)
            for j in range(self.p):
                phi_up = phi[:]; phi_up[j] += step
                grad = (compute_loss(phi_up, theta) - loss) / step
                phi_new[j] -= 1e-3 * grad

            theta_new = list(theta)
            for j in range(self.q):
                theta_up = theta[:]; theta_up[j] += step
                grad = (compute_loss(phi, theta_up) - loss) / step
                theta_new[j] -= 1e-3 * grad

            if (max(abs(phi_new[j] - phi[j]) for j in range(self.p)) +
                    max(abs(theta_new[j] - theta[j]) for j in range(self.q))) < self.tol:
                phi, theta = phi_new, theta_new
                break
            phi, theta = phi_new, theta_new
        self.phi_ = phi
        self.theta_ = theta
        return self

    def predict(self, steps: int) -> List[float]:
        x = self._x
        n = len(x)
        eps = [0.0] * n
        for t in range(n):
            ar = sum(self.phi_[j] * (x[t - j - 1] - self.intercept_) for j in range(min(self.p, t)))
            ma = sum(self.theta_[j] * eps[t - j - 1] for j in range(min(self.q, t)))
            eps[t] = x[t] - self.intercept_ - ar - ma
        ext_x = list(x)
        ext_eps = list(eps)
        preds = []
        for h in range(steps):
            t = n + h
            ar = sum(self.phi_[j] * (ext_x[t - j - 1] - self.intercept_) for j in range(self.p))
            ma = sum(self.theta_[j] * ext_eps[t - j - 1] for j in range(self.q) if t - j - 1 < n)
            val = self.intercept_ + ar + ma
            preds.append(val)
            ext_x.append(val)
            ext_eps.append(0.0)
        return preds


# ---------------------------------------------------------------------------
# Seasonal Decomposition (additive)
# ---------------------------------------------------------------------------

def seasonal_decompose(
    x: List[float],
    period: int
) -> Tuple[List[float], List[float], List[float]]:
    """Additive seasonal decomposition: x = trend + seasonal + residual.

    Returns (trend, seasonal, residual).
    """
    n = len(x)
    if n < 2 * period:
        raise ValueError(f"Need at least {2 * period} observations")
    # 1. Trend: centred moving average
    half = period // 2
    trend = []
    for i in range(n):
        if i < half or i >= n - half:
            trend.append(float("nan"))
        else:
            trend.append(sum(x[i - half:i + half + 1]) / (2 * half + 1))
    # 2. Detrend
    detrended = [x[i] - trend[i] if not math.isnan(trend[i]) else float("nan") for i in range(n)]
    # 3. Average seasonal pattern
    seasonal_avg = []
    for s in range(period):
        vals = [detrended[i] for i in range(s, n, period) if not math.isnan(detrended[i])]
        seasonal_avg.append(sum(vals) / len(vals) if vals else 0.0)
    # Normalise seasonal so it sums to zero
    mean_s = sum(seasonal_avg) / period
    seasonal_avg = [v - mean_s for v in seasonal_avg]
    seasonal = [seasonal_avg[i % period] for i in range(n)]
    # 4. Residual
    residual = [x[i] - (trend[i] if not math.isnan(trend[i]) else 0) - seasonal[i] for i in range(n)]
    return trend, seasonal, residual


# ---------------------------------------------------------------------------
# Spectral Analysis
# ---------------------------------------------------------------------------

def periodogram(x: List[float]) -> Tuple[List[float], List[float]]:
    """Return (frequencies, power) periodogram via DFT."""
    from mathcore.complex.transforms import rfft, fft_frequencies, _next_power_of_2
    n_orig = len(x)
    X = rfft(x)
    n_fft = _next_power_of_2(n_orig)
    freqs = fft_frequencies(n_fft)
    power = [abs(v.real) ** 2 + abs(v.imag) ** 2 for v in X]
    return freqs[:len(X)], power


def dominant_frequency(x: List[float]) -> Tuple[float, float]:
    """Return (dominant_frequency, power) of the strongest spectral peak."""
    freqs, power = periodogram(x)
    # Skip DC component (freq=0)
    best_idx = max(range(1, len(power)), key=lambda i: power[i])
    return freqs[best_idx], power[best_idx]


def band_pass_filter(x: List[float], low_freq: float, high_freq: float) -> List[float]:
    """Apply a rectangular band-pass filter in the frequency domain."""
    from mathcore.complex.transforms import fft, ifft, _next_power_of_2
    n = len(x)
    n_fft = _next_power_of_2(n)
    from mathcore.complex.numbers import Complex
    padded = x + [0.0] * (n_fft - n)
    X = fft(padded)
    filtered = []
    for k, Xk in enumerate(X):
        freq = k / n_fft
        if low_freq <= freq <= high_freq or (1 - high_freq) <= freq <= (1 - low_freq):
            filtered.append(Xk)
        else:
            filtered.append(Complex(0, 0))
    result = ifft(filtered)
    return [v.real for v in result[:n]]
