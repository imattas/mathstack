"""
Comprehensive Statistics Module
Descriptive statistics, hypothesis testing, regression, correlation, and more.
"""

import math
from typing import List, Tuple, Dict, Optional
from collections import Counter
import bisect


class DescriptiveStatistics:
    """Descriptive statistics calculations."""
    
    @staticmethod
    def mean(data: List[float]) -> float:
        """Calculate arithmetic mean."""
        if not data:
            raise ValueError("Cannot calculate mean of empty list")
        return sum(data) / len(data)
    
    @staticmethod
    def median(data: List[float]) -> float:
        """Calculate median."""
        if not data:
            raise ValueError("Cannot calculate median of empty list")
        sorted_data = sorted(data)
        n = len(sorted_data)
        if n % 2 == 1:
            return sorted_data[n // 2]
        return (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
    
    @staticmethod
    def mode(data: List[float]) -> List[float]:
        """Calculate mode (most frequent value)."""
        if not data:
            raise ValueError("Cannot calculate mode of empty list")
        counts = Counter(data)
        max_count = max(counts.values())
        return sorted([val for val, count in counts.items() if count == max_count])
    
    @staticmethod
    def variance(data: List[float], sample: bool = True) -> float:
        """Calculate variance.
        
        Args:
            data: Data points
            sample: If True, use sample variance (n-1), else population variance (n)
        """
        if not data:
            raise ValueError("Cannot calculate variance of empty list")
        mean = DescriptiveStatistics.mean(data)
        n = len(data)
        divisor = (n - 1) if sample else n
        if divisor == 0:
            return 0
        return sum((x - mean) ** 2 for x in data) / divisor
    
    @staticmethod
    def std_dev(data: List[float], sample: bool = True) -> float:
        """Calculate standard deviation."""
        return math.sqrt(DescriptiveStatistics.variance(data, sample))
    
    @staticmethod
    def quartiles(data: List[float]) -> Tuple[float, float, float]:
        """Calculate Q1, Q2 (median), Q3."""
        sorted_data = sorted(data)
        n = len(sorted_data)
        q2 = DescriptiveStatistics.median(sorted_data)
        q1 = DescriptiveStatistics.median(sorted_data[:n // 2])
        q3 = DescriptiveStatistics.median(sorted_data[(n + 1) // 2:])
        return (q1, q2, q3)
    
    @staticmethod
    def iqr(data: List[float]) -> float:
        """Calculate Interquartile Range."""
        q1, _, q3 = DescriptiveStatistics.quartiles(data)
        return q3 - q1
    
    @staticmethod
    def percentile(data: List[float], p: float) -> float:
        """Calculate pth percentile (0-100)."""
        if not 0 <= p <= 100:
            raise ValueError("Percentile must be between 0 and 100")
        sorted_data = sorted(data)
        index = (p / 100) * (len(sorted_data) - 1)
        lower = int(index)
        upper = lower + 1
        if upper >= len(sorted_data):
            return sorted_data[-1]
        weight = index - lower
        return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight
    
    @staticmethod
    def skewness(data: List[float]) -> float:
        """Calculate skewness (third moment)."""
        if len(data) < 3:
            return 0
        mean = DescriptiveStatistics.mean(data)
        std = DescriptiveStatistics.std_dev(data)
        if std == 0:
            return 0
        n = len(data)
        return (sum((x - mean) ** 3 for x in data) / n) / (std ** 3)
    
    @staticmethod
    def kurtosis(data: List[float]) -> float:
        """Calculate kurtosis (fourth moment)."""
        if len(data) < 4:
            return 0
        mean = DescriptiveStatistics.mean(data)
        std = DescriptiveStatistics.std_dev(data)
        if std == 0:
            return 0
        n = len(data)
        return (sum((x - mean) ** 4 for x in data) / n) / (std ** 4) - 3


class LinearRegression:
    """Simple and multiple linear regression."""
    
    def __init__(self):
        self.slope = None
        self.intercept = None
        self.x_data = None
        self.y_data = None
    
    def fit(self, x: List[float], y: List[float]) -> None:
        """Fit linear regression line."""
        if len(x) != len(y):
            raise ValueError("x and y must have same length")
        
        n = len(x)
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            raise ValueError("Cannot fit regression: no variance in x")
        
        self.slope = numerator / denominator
        self.intercept = y_mean - self.slope * x_mean
        self.x_data = x
        self.y_data = y
    
    def predict(self, x: float) -> float:
        """Predict y for given x."""
        if self.slope is None:
            raise ValueError("Model not fitted yet")
        return self.slope * x + self.intercept
    
    def r_squared(self) -> float:
        """Calculate R² (coefficient of determination)."""
        if self.slope is None:
            raise ValueError("Model not fitted yet")
        
        y_pred = [self.predict(x) for x in self.x_data]
        ss_res = sum((self.y_data[i] - y_pred[i]) ** 2 for i in range(len(self.y_data)))
        y_mean = sum(self.y_data) / len(self.y_data)
        ss_tot = sum((y - y_mean) ** 2 for y in self.y_data)
        
        if ss_tot == 0:
            return 0
        return 1 - (ss_res / ss_tot)
    
    def correlation_coefficient(self) -> float:
        """Calculate Pearson correlation coefficient."""
        if self.slope is None:
            raise ValueError("Model not fitted yet")
        
        n = len(self.x_data)
        x_mean = sum(self.x_data) / n
        y_mean = sum(self.y_data) / n
        
        numerator = sum((self.x_data[i] - x_mean) * (self.y_data[i] - y_mean) for i in range(n))
        denominator = math.sqrt(
            sum((x - x_mean) ** 2 for x in self.x_data) *
            sum((y - y_mean) ** 2 for y in self.y_data)
        )
        
        if denominator == 0:
            return 0
        return numerator / denominator


class HypothesisTesting:
    """Statistical hypothesis testing."""
    
    @staticmethod
    def t_statistic(sample_mean: float, population_mean: float, 
                   std_error: float) -> float:
        """Calculate t-statistic."""
        if std_error == 0:
            raise ValueError("Standard error cannot be zero")
        return (sample_mean - population_mean) / std_error
    
    @staticmethod
    def z_statistic(sample_mean: float, population_mean: float,
                   population_std: float, n: int) -> float:
        """Calculate z-statistic."""
        if population_std == 0:
            raise ValueError("Population standard deviation cannot be zero")
        return (sample_mean - population_mean) / (population_std / math.sqrt(n))
    
    @staticmethod
    def chi_square(observed: List[float], expected: List[float]) -> float:
        """Calculate chi-square statistic."""
        if len(observed) != len(expected):
            raise ValueError("observed and expected must have same length")
        
        chi_sq = 0
        for obs, exp in zip(observed, expected):
            if exp == 0:
                raise ValueError("Expected frequencies cannot be zero")
            chi_sq += ((obs - exp) ** 2) / exp
        return chi_sq
    
    @staticmethod
    def contingency_table_chi_square(table: List[List[float]]) -> float:
        """Calculate chi-square for contingency table."""
        # Calculate row and column totals
        row_totals = [sum(row) for row in table]
        col_totals = [sum(table[i][j] for i in range(len(table))) 
                      for j in range(len(table[0]))]
        grand_total = sum(row_totals)
        
        chi_sq = 0
        for i in range(len(table)):
            for j in range(len(table[0])):
                expected = (row_totals[i] * col_totals[j]) / grand_total
                if expected > 0:
                    chi_sq += ((table[i][j] - expected) ** 2) / expected
        
        return chi_sq
    
    @staticmethod
    def anova_f_statistic(groups: List[List[float]]) -> float:
        """Calculate F-statistic for ANOVA (Analysis of Variance)."""
        k = len(groups)  # number of groups
        n = sum(len(group) for group in groups)
        
        # Overall mean
        all_data = [val for group in groups for val in group]
        grand_mean = sum(all_data) / n
        
        # Between-group sum of squares
        ss_between = sum(len(group) * (sum(group) / len(group) - grand_mean) ** 2
                        for group in groups)
        
        # Within-group sum of squares
        ss_within = sum(sum((val - sum(group) / len(group)) ** 2 for val in group)
                       for group in groups)
        
        # Degrees of freedom
        df_between = k - 1
        df_within = n - k
        
        if df_within == 0:
            return 0
        
        ms_between = ss_between / df_between
        ms_within = ss_within / df_within
        
        if ms_within == 0:
            return 0
        
        return ms_between / ms_within


class CorrelationAnalysis:
    """Correlation and covariance analysis."""
    
    @staticmethod
    def covariance(x: List[float], y: List[float]) -> float:
        """Calculate covariance."""
        if len(x) != len(y):
            raise ValueError("x and y must have same length")
        
        n = len(x)
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        
        return sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n)) / (n - 1)
    
    @staticmethod
    def pearson_correlation(x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x) != len(y):
            raise ValueError("x and y must have same length")
        
        n = len(x)
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = math.sqrt(
            sum((xi - x_mean) ** 2 for xi in x) *
            sum((yi - y_mean) ** 2 for yi in y)
        )
        
        if denominator == 0:
            return 0
        return numerator / denominator
    
    @staticmethod
    def spearman_correlation(x: List[float], y: List[float]) -> float:
        """Calculate Spearman rank correlation."""
        if len(x) != len(y):
            raise ValueError("x and y must have same length")
        
        # Rank the data
        def rank_data(data):
            sorted_data = sorted(enumerate(data), key=lambda item: item[1])
            ranks = [0] * len(data)
            for rank, (idx, _) in enumerate(sorted_data, 1):
                ranks[idx] = rank
            return ranks
        
        x_ranks = rank_data(x)
        y_ranks = rank_data(y)
        
        return CorrelationAnalysis.pearson_correlation(x_ranks, y_ranks)
