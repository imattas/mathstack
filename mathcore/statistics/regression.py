"""
Advanced Regression — from scratch, zero external dependencies.
Includes: OLS (multiple), polynomial, ridge, lasso (coordinate descent),
logistic regression, k-fold cross-validation, VIF, stepwise selection.
"""

import math
from typing import List, Tuple, Optional, Callable


# ---------------------------------------------------------------------------
# Helper: Basic linear algebra (no numpy)
# ---------------------------------------------------------------------------

def _dot(a: List[float], b: List[float]) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


def _mat_mul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    rows_A, cols_A = len(A), len(A[0])
    cols_B = len(B[0])
    return [
        [sum(A[i][k] * B[k][j] for k in range(cols_A)) for j in range(cols_B)]
        for i in range(rows_A)
    ]


def _transpose(A: List[List[float]]) -> List[List[float]]:
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


def _cholesky(A: List[List[float]]) -> List[List[float]]:
    """Lower Cholesky decomposition A = L L^T."""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = A[i][i] - s
                if val <= 0:
                    val = 1e-12  # numerical stability
                L[i][j] = math.sqrt(val)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L


def _solve_cholesky(L: List[List[float]], b: List[float]) -> List[float]:
    """Solve L L^T x = b."""
    n = len(b)
    # Forward substitution L y = b
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(L[i][k] * y[k] for k in range(i))) / L[i][i]
    # Back substitution L^T x = y
    x = [0.0] * n
    LT = _transpose(L)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(LT[i][k] * x[k] for k in range(i + 1, n))) / LT[i][i]
    return x


def _solve_ols(X: List[List[float]], y: List[float], ridge: float = 0.0) -> List[float]:
    """Solve the normal equations (X^T X + ridge*I) beta = X^T y."""
    Xt = _transpose(X)
    XtX = _mat_mul(Xt, [[yi] for yi in y])
    XtX_mat = _mat_mul(Xt, X)
    n_cols = len(XtX_mat)
    for i in range(n_cols):
        XtX_mat[i][i] += ridge
    Xty = [sum(Xt[i][j] * y[j] for j in range(len(y))) for i in range(n_cols)]
    L = _cholesky(XtX_mat)
    return _solve_cholesky(L, Xty)


def _design_matrix(X: List[List[float]], intercept: bool = True) -> List[List[float]]:
    """Add intercept column to design matrix if requested."""
    if intercept:
        return [[1.0] + list(row) for row in X]
    return [list(row) for row in X]


# ---------------------------------------------------------------------------
# Ordinary Least Squares (Multiple Linear Regression)
# ---------------------------------------------------------------------------

class OLSRegression:
    """Multiple linear regression via Ordinary Least Squares (normal equations)."""

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.coef_: Optional[List[float]] = None
        self.intercept_: float = 0.0
        self._X: Optional[List[List[float]]] = None
        self._y: Optional[List[float]] = None

    def fit(self, X: List[List[float]], y: List[float]) -> "OLSRegression":
        self._X = X
        self._y = y
        Xd = _design_matrix(X, self.fit_intercept)
        beta = _solve_ols(Xd, y)
        if self.fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta
        return self

    def predict(self, X: List[List[float]]) -> List[float]:
        return [self.intercept_ + _dot(self.coef_, row) for row in X]

    def r_squared(self) -> float:
        y_pred = self.predict(self._X)
        y_mean = sum(self._y) / len(self._y)
        ss_res = sum((y - yp) ** 2 for y, yp in zip(self._y, y_pred))
        ss_tot = sum((y - y_mean) ** 2 for y in self._y)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def adjusted_r_squared(self) -> float:
        n = len(self._y)
        p = len(self.coef_)
        r2 = self.r_squared()
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)

    def residuals(self) -> List[float]:
        y_pred = self.predict(self._X)
        return [y - yp for y, yp in zip(self._y, y_pred)]

    def mse(self) -> float:
        res = self.residuals()
        return sum(r ** 2 for r in res) / len(res)

    def rmse(self) -> float:
        return math.sqrt(self.mse())

    def coefficients_with_names(self, feature_names: Optional[List[str]] = None) -> dict:
        names = feature_names or [f"x{i}" for i in range(len(self.coef_))]
        result = {"intercept": self.intercept_}
        result.update(dict(zip(names, self.coef_)))
        return result


# ---------------------------------------------------------------------------
# Polynomial Regression
# ---------------------------------------------------------------------------

class PolynomialRegression:
    """Polynomial regression of degree d: fit y = sum_i beta_i * x^i."""

    def __init__(self, degree: int = 2, fit_intercept: bool = True):
        self.degree = degree
        self._ols = OLSRegression(fit_intercept=fit_intercept)

    def _poly_features(self, x: List[float]) -> List[List[float]]:
        return [[xi ** d for d in range(1, self.degree + 1)] for xi in x]

    def fit(self, x: List[float], y: List[float]) -> "PolynomialRegression":
        self._ols.fit(self._poly_features(x), y)
        return self

    def predict(self, x: List[float]) -> List[float]:
        return self._ols.predict(self._poly_features(x))

    def r_squared(self) -> float:
        return self._ols.r_squared()

    @property
    def coef_(self) -> List[float]:
        return [self._ols.intercept_] + self._ols.coef_


# ---------------------------------------------------------------------------
# Ridge Regression (L2 regularisation)
# ---------------------------------------------------------------------------

class RidgeRegression:
    """Ridge regression: minimise ||y - Xb||^2 + alpha * ||b||^2."""

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_: Optional[List[float]] = None
        self.intercept_: float = 0.0
        self._X: Optional[List[List[float]]] = None
        self._y: Optional[List[float]] = None

    def fit(self, X: List[List[float]], y: List[float]) -> "RidgeRegression":
        self._X, self._y = X, y
        if self.fit_intercept:
            # Centre X and y before fitting (do not penalise intercept)
            n = len(y)
            x_means = [sum(X[i][j] for i in range(n)) / n for j in range(len(X[0]))]
            y_mean = sum(y) / n
            Xc = [[X[i][j] - x_means[j] for j in range(len(X[0]))] for i in range(n)]
            yc = [yi - y_mean for yi in y]
            beta = _solve_ols(Xc, yc, ridge=self.alpha)
            self.coef_ = beta
            self.intercept_ = y_mean - _dot(x_means, beta)
        else:
            beta = _solve_ols(X, y, ridge=self.alpha)
            self.coef_ = beta
            self.intercept_ = 0.0
        return self

    def predict(self, X: List[List[float]]) -> List[float]:
        return [self.intercept_ + _dot(self.coef_, row) for row in X]

    def r_squared(self) -> float:
        y_pred = self.predict(self._X)
        y_mean = sum(self._y) / len(self._y)
        ss_res = sum((y - yp) ** 2 for y, yp in zip(self._y, y_pred))
        ss_tot = sum((y - y_mean) ** 2 for y in self._y)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


# ---------------------------------------------------------------------------
# Lasso Regression (L1 regularisation via coordinate descent)
# ---------------------------------------------------------------------------

class LassoRegression:
    """Lasso regression: minimise (1/2n)||y - Xb||^2 + alpha * ||b||_1."""

    def __init__(self, alpha: float = 1.0, max_iter: int = 1000, tol: float = 1e-6):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_: Optional[List[float]] = None
        self.intercept_: float = 0.0
        self._X: Optional[List[List[float]]] = None
        self._y: Optional[List[float]] = None

    def fit(self, X: List[List[float]], y: List[float]) -> "LassoRegression":
        self._X, self._y = X, y
        n, p = len(y), len(X[0])
        # Centre and scale
        x_means = [sum(X[i][j] for i in range(n)) / n for j in range(p)]
        x_stds = [math.sqrt(sum((X[i][j] - x_means[j]) ** 2 for i in range(n)) / n + 1e-12) for j in range(p)]
        y_mean = sum(y) / n
        Xn = [[(X[i][j] - x_means[j]) / x_stds[j] for j in range(p)] for i in range(n)]
        yn = [yi - y_mean for yi in y]
        beta = [0.0] * p
        # Coordinate descent
        for _ in range(self.max_iter):
            beta_old = beta[:]
            for j in range(p):
                residual = [yn[i] - sum(beta[k] * Xn[i][k] for k in range(p) if k != j) for i in range(n)]
                rho = sum(Xn[i][j] * residual[i] for i in range(n)) / n
                # Soft threshold
                if rho > self.alpha:
                    beta[j] = rho - self.alpha
                elif rho < -self.alpha:
                    beta[j] = rho + self.alpha
                else:
                    beta[j] = 0.0
            if max(abs(b - bo) for b, bo in zip(beta, beta_old)) < self.tol:
                break
        # Unscale
        self.coef_ = [beta[j] / x_stds[j] for j in range(p)]
        self.intercept_ = y_mean - sum(self.coef_[j] * x_means[j] for j in range(p))
        return self

    def predict(self, X: List[List[float]]) -> List[float]:
        return [self.intercept_ + _dot(self.coef_, row) for row in X]


# ---------------------------------------------------------------------------
# Logistic Regression (via gradient descent)
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1 + ex)


class LogisticRegression:
    """Binary logistic regression via gradient descent with L2 regularisation."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_iter: int = 1000,
        tol: float = 1e-6,
        C: float = 1.0  # inverse regularisation strength
    ):
        self.lr = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.C = C
        self.coef_: Optional[List[float]] = None
        self.intercept_: float = 0.0

    def fit(self, X: List[List[float]], y: List[int]) -> "LogisticRegression":
        n, p = len(y), len(X[0])
        beta = [0.0] * p
        b0 = 0.0
        for _ in range(self.max_iter):
            # Compute gradients
            preds = [_sigmoid(b0 + _dot(beta, X[i])) for i in range(n)]
            errors = [preds[i] - y[i] for i in range(n)]
            grad_b0 = sum(errors) / n
            grad_beta = [
                sum(errors[i] * X[i][j] for i in range(n)) / n + beta[j] / (self.C * n)
                for j in range(p)
            ]
            # Update
            b0_new = b0 - self.lr * grad_b0
            beta_new = [beta[j] - self.lr * grad_beta[j] for j in range(p)]
            if max(abs(beta_new[j] - beta[j]) for j in range(p)) < self.tol:
                beta, b0 = beta_new, b0_new
                break
            beta, b0 = beta_new, b0_new
        self.coef_ = beta
        self.intercept_ = b0
        return self

    def predict_proba(self, X: List[List[float]]) -> List[float]:
        return [_sigmoid(self.intercept_ + _dot(self.coef_, row)) for row in X]

    def predict(self, X: List[List[float]], threshold: float = 0.5) -> List[int]:
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]

    def accuracy(self, X: List[List[float]], y: List[int]) -> float:
        preds = self.predict(X)
        return sum(1 for p, yi in zip(preds, y) if p == yi) / len(y)

    def log_loss(self, X: List[List[float]], y: List[int]) -> float:
        probs = self.predict_proba(X)
        eps = 1e-15
        return -sum(
            yi * math.log(max(p, eps)) + (1 - yi) * math.log(max(1 - p, eps))
            for yi, p in zip(y, probs)
        ) / len(y)


# ---------------------------------------------------------------------------
# k-fold Cross Validation
# ---------------------------------------------------------------------------

def k_fold_cross_validate(
    model,
    X: List[List[float]],
    y: List[float],
    k: int = 5,
    metric: str = "r2"
) -> Tuple[float, float]:
    """k-fold cross-validation. Returns (mean_score, std_score).

    metric: 'r2', 'mse', 'rmse'
    """
    n = len(y)
    fold_size = n // k
    indices = list(range(n))
    scores = []
    for fold in range(k):
        val_idx = indices[fold * fold_size:(fold + 1) * fold_size]
        train_idx = [i for i in indices if i not in set(val_idx)]
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_val = [X[i] for i in val_idx]
        y_val = [y[i] for i in val_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        if metric == "r2":
            y_mean = sum(y_val) / len(y_val)
            ss_res = sum((yv - yp) ** 2 for yv, yp in zip(y_val, y_pred))
            ss_tot = sum((yv - y_mean) ** 2 for yv in y_val)
            score = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        elif metric in ("mse", "rmse"):
            mse = sum((yv - yp) ** 2 for yv, yp in zip(y_val, y_pred)) / len(y_val)
            score = math.sqrt(mse) if metric == "rmse" else mse
        else:
            raise ValueError(f"Unknown metric: {metric}")
        scores.append(score)
    mean = sum(scores) / k
    std = math.sqrt(sum((s - mean) ** 2 for s in scores) / k)
    return mean, std


# ---------------------------------------------------------------------------
# VIF (Variance Inflation Factor)
# ---------------------------------------------------------------------------

def vif(X: List[List[float]]) -> List[float]:
    """Compute VIF for each predictor in X.

    VIF_j = 1/(1 - R^2_j) where R^2_j is R-squared regressing X_j on the rest.
    """
    p = len(X[0])
    result = []
    for j in range(p):
        y_j = [X[i][j] for i in range(len(X))]
        X_rest = [[X[i][k] for k in range(p) if k != j] for i in range(len(X))]
        model = OLSRegression()
        model.fit(X_rest, y_j)
        r2 = model.r_squared()
        result.append(1 / (1 - r2) if r2 < 1 else float("inf"))
    return result


# ---------------------------------------------------------------------------
# Stepwise Selection (Forward / Backward)
# ---------------------------------------------------------------------------

def forward_stepwise(
    X: List[List[float]],
    y: List[float],
    criterion: str = "aic"
) -> List[int]:
    """Forward stepwise feature selection by AIC or BIC.

    Returns list of selected feature indices.
    """
    n = len(y)
    remaining = list(range(len(X[0])))
    selected = []

    def _criterion_score(indices):
        X_sub = [[X[i][j] for j in indices] for i in range(n)]
        model = OLSRegression()
        model.fit(X_sub, y)
        rss = sum(r ** 2 for r in model.residuals())
        k = len(indices) + 1  # +1 for intercept
        log_lik = -n / 2 * math.log(rss / n + 1e-15)
        if criterion == "aic":
            return 2 * k - 2 * log_lik
        return k * math.log(n) - 2 * log_lik  # BIC

    current_score = float("inf")
    while remaining:
        best_score, best_feat = float("inf"), None
        for feat in remaining:
            candidate = selected + [feat]
            score = _criterion_score(candidate)
            if score < best_score:
                best_score, best_feat = score, feat
        if best_score < current_score:
            selected.append(best_feat)
            remaining.remove(best_feat)
            current_score = best_score
        else:
            break
    return selected
