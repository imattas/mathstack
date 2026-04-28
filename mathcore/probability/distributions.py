"""
Probability Distributions Module
Normal, Binomial, Poisson, Exponential, Uniform distributions and their properties.
"""

import math
from typing import Tuple, List


class NormalDistribution:
    """Normal (Gaussian) Distribution."""
    
    def __init__(self, mean: float = 0, std_dev: float = 1):
        """Initialize Normal distribution.
        
        Args:
            mean: Mean (μ)
            std_dev: Standard deviation (σ)
        """
        if std_dev <= 0:
            raise ValueError("Standard deviation must be positive")
        self.mean = mean
        self.std_dev = std_dev
    
    def pdf(self, x: float) -> float:
        """Probability Density Function."""
        z = (x - self.mean) / self.std_dev
        return (1 / (self.std_dev * math.sqrt(2 * math.pi))) * math.exp(-z**2 / 2)
    
    def cdf(self, x: float) -> float:
        """Cumulative Distribution Function (approximation)."""
        z = (x - self.mean) / self.std_dev
        # Approximation using error function
        return 0.5 * (1 + self._erf(z / math.sqrt(2)))
    
    @staticmethod
    def _erf(x: float) -> float:
        """Error function approximation."""
        # Chebyshev approximation
        a1 =  0.254829592
        a2 = -0.284496736
        a3 =  1.421413741
        a4 = -1.453152027
        a5 =  1.061405429
        p  =  0.3275911
        
        sign = 1 if x >= 0 else -1
        x = abs(x)
        
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
        
        return sign * y
    
    def quantile(self, p: float) -> float:
        """Inverse CDF (quantile function)."""
        if not 0 < p < 1:
            raise ValueError("Probability must be between 0 and 1")
        return self.mean + self.std_dev * self._inverse_normal(p)
    
    @staticmethod
    def _inverse_normal(p: float) -> float:
        """Approximation of inverse normal CDF."""
        if p < 0.5:
            t = math.sqrt(-2.0 * math.log(p))
            return -(((2.515517 + 0.802853 * t + 0.010328 * t**2) /
                     (1.0 + 1.432788 * t + 0.189269 * t**2 + 0.001308 * t**3)))
        else:
            t = math.sqrt(-2.0 * math.log(1 - p))
            return (((2.515517 + 0.802853 * t + 0.010328 * t**2) /
                    (1.0 + 1.432788 * t + 0.189269 * t**2 + 0.001308 * t**3)))


class BinomialDistribution:
    """Binomial Distribution."""
    
    def __init__(self, n: int, p: float):
        """Initialize Binomial distribution.
        
        Args:
            n: Number of trials
            p: Probability of success
        """
        if not 0 < p < 1:
            raise ValueError("Probability must be between 0 and 1")
        if n < 1:
            raise ValueError("Number of trials must be positive")
        self.n = n
        self.p = p
    
    def pmf(self, k: int) -> float:
        """Probability Mass Function P(X = k)."""
        if k < 0 or k > self.n:
            return 0
        
        from mathcore.core.arithmetic import binomial_coefficient
        return binomial_coefficient(self.n, k) * (self.p ** k) * ((1 - self.p) ** (self.n - k))
    
    def cdf(self, k: int) -> float:
        """Cumulative Distribution Function P(X <= k)."""
        return sum(self.pmf(i) for i in range(k + 1))
    
    def mean(self) -> float:
        """Expected value E[X] = n*p"""
        return self.n * self.p
    
    def variance(self) -> float:
        """Variance Var[X] = n*p*(1-p)"""
        return self.n * self.p * (1 - self.p)
    
    def std_dev(self) -> float:
        """Standard deviation"""
        return math.sqrt(self.variance())


class PoissonDistribution:
    """Poisson Distribution."""
    
    def __init__(self, lambda_param: float):
        """Initialize Poisson distribution.
        
        Args:
            lambda_param: Rate parameter (λ)
        """
        if lambda_param <= 0:
            raise ValueError("Lambda must be positive")
        self.lambda_param = lambda_param
    
    def pmf(self, k: int) -> float:
        """Probability Mass Function P(X = k)."""
        if k < 0:
            return 0
        
        from mathcore.core.arithmetic import factorial
        return (math.exp(-self.lambda_param) * (self.lambda_param ** k)) / factorial(k)
    
    def cdf(self, k: int) -> float:
        """Cumulative Distribution Function P(X <= k)."""
        return sum(self.pmf(i) for i in range(k + 1))
    
    def mean(self) -> float:
        """Expected value E[X] = λ"""
        return self.lambda_param
    
    def variance(self) -> float:
        """Variance Var[X] = λ"""
        return self.lambda_param
    
    def std_dev(self) -> float:
        """Standard deviation"""
        return math.sqrt(self.variance())


class ExponentialDistribution:
    """Exponential Distribution."""
    
    def __init__(self, lambda_param: float):
        """Initialize Exponential distribution.
        
        Args:
            lambda_param: Rate parameter (λ)
        """
        if lambda_param <= 0:
            raise ValueError("Lambda must be positive")
        self.lambda_param = lambda_param
    
    def pdf(self, x: float) -> float:
        """Probability Density Function."""
        if x < 0:
            return 0
        return self.lambda_param * math.exp(-self.lambda_param * x)
    
    def cdf(self, x: float) -> float:
        """Cumulative Distribution Function."""
        if x < 0:
            return 0
        return 1 - math.exp(-self.lambda_param * x)
    
    def mean(self) -> float:
        """Expected value E[X] = 1/λ"""
        return 1 / self.lambda_param
    
    def variance(self) -> float:
        """Variance Var[X] = 1/λ²"""
        return 1 / (self.lambda_param ** 2)
    
    def std_dev(self) -> float:
        """Standard deviation"""
        return math.sqrt(self.variance())


class UniformDistribution:
    """Uniform Distribution."""
    
    def __init__(self, a: float, b: float):
        """Initialize Uniform distribution.
        
        Args:
            a: Lower bound
            b: Upper bound
        """
        if a >= b:
            raise ValueError("Lower bound must be less than upper bound")
        self.a = a
        self.b = b
    
    def pdf(self, x: float) -> float:
        """Probability Density Function."""
        if self.a <= x <= self.b:
            return 1 / (self.b - self.a)
        return 0
    
    def cdf(self, x: float) -> float:
        """Cumulative Distribution Function."""
        if x < self.a:
            return 0
        if x > self.b:
            return 1
        return (x - self.a) / (self.b - self.a)
    
    def mean(self) -> float:
        """Expected value E[X] = (a+b)/2"""
        return (self.a + self.b) / 2
    
    def variance(self) -> float:
        """Variance Var[X] = (b-a)²/12"""
        return ((self.b - self.a) ** 2) / 12
    
    def std_dev(self) -> float:
        """Standard deviation"""
        return math.sqrt(self.variance())


class ChiSquaredDistribution:
    """Chi-Squared Distribution."""
    
    def __init__(self, k: int):
        """Initialize Chi-Squared distribution.
        
        Args:
            k: Degrees of freedom
        """
        if k < 1:
            raise ValueError("Degrees of freedom must be positive")
        self.k = k
    
    def pdf(self, x: float) -> float:
        """Probability Density Function."""
        if x < 0:
            return 0
        
        # Using approximation without special functions
        numerator = (x ** (self.k / 2 - 1)) * math.exp(-x / 2)
        denominator = (2 ** (self.k / 2)) * self._gamma(self.k / 2)
        return numerator / denominator
    
    @staticmethod
    def _gamma(z: float) -> float:
        """Stirling's approximation of gamma function."""
        if z == 1:
            return 1
        if z == 0.5:
            return math.sqrt(math.pi)
        return math.sqrt(2 * math.pi / z) * ((z / math.e) ** z)
    
    def mean(self) -> float:
        """Expected value E[X] = k"""
        return self.k
    
    def variance(self) -> float:
        """Variance Var[X] = 2k"""
        return 2 * self.k
    
    def std_dev(self) -> float:
        """Standard deviation"""
        return math.sqrt(self.variance())


class TDistribution:
    """Student's t-Distribution."""
    
    def __init__(self, df: float):
        """Initialize t-distribution.
        
        Args:
            df: Degrees of freedom
        """
        if df <= 0:
            raise ValueError("Degrees of freedom must be positive")
        self.df = df
    
    def pdf(self, x: float) -> float:
        """Probability Density Function."""
        # Using Stirling's approximation
        numerator = (1 + x**2 / self.df) ** (-(self.df + 1) / 2)
        denominator = math.sqrt(self.df * math.pi)
        
        # Beta function approximation
        return numerator / denominator
    
    def mean(self) -> float:
        """Expected value E[X] = 0 (if df > 1)"""
        return 0 if self.df > 1 else float('nan')
    
    def variance(self) -> float:
        """Variance Var[X] = df/(df-2) (if df > 2)"""
        if self.df > 2:
            return self.df / (self.df - 2)
        return float('inf')
