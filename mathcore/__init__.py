"""
MathCore - Advanced Mathematics Library for Python

A powerful, zero-dependency library for advanced mathematical operations including:
- Symbolic algebra and simplification
- Geometric calculations and plane operations
- Line collision/intersection detection
- Advanced calculus operations
- Polynomial and equation solving
- Matrix operations and linear algebra
- Statistics and data analysis (inference, regression, time series)
- Probability distributions, Markov chains, Monte Carlo
- Optimization algorithms and linear programming
- Differential equations (ODE, PDE, BVP)
- Complex analysis and transforms (FFT, wavelets)
- Number theory and combinatorics
- Step-by-step work display for all operations
"""

__version__ = "3.1.0"
__author__ = "MathCore Contributors"
__license__ = "MIT"

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
from mathcore.core.arithmetic import (
    factorial, gcd, lcm, is_prime, prime_factors,
    fibonacci, sum_of_divisors, euler_totient,
    binomial_coefficient, combination, permutation,
    is_perfect_square, power_mod, nth_root
)
from mathcore.core.algebra import (
    Polynomial, simplify, solve_quadratic, solve_cubic,
    factor_expression, expand_expression
)
from mathcore.geometry import (
    Point, Line, Circle, Triangle, Vector, Polygon, Rectangle,
    find_line_intersection, distance, midpoint,
    line_collision_detection, circle_line_intersection, point_to_line_distance,
    angle_between, area_triangle, line_slope, line_intercept,
    rotate_point, translate_point, reflect_point, scale_point,
    rotate_shape, translate_shape, reflect_shape, scale_shape,
    circle_circle_intersection
)
from mathcore.core.calculus import (
    derivative, integral, limit, series_expansion,
    find_critical_points, second_derivative_test,
    find_root_bisection, find_root_newton, optimize_minimize,
    second_derivative
)
from mathcore.core.matrix import Matrix
from mathcore.core.advanced_linear_algebra import (
    EigenAnalysis, QRDecomposition, SingularValueDecomposition,
    NormCalculations, MatrixDecompositions
)
from mathcore.core.number_theory import (
    sieve_of_eratosthenes, segmented_sieve, is_prime_miller_rabin,
    prime_factorisation, divisors, extended_gcd, modular_inverse,
    chinese_remainder_theorem, euler_totient as euler_totient_nt,
    mobius_function, primitive_root, discrete_log_bsgs,
    legendre_symbol, jacobi_symbol, continued_fraction,
    continued_fraction_sqrt, convergents, pell_equation,
    carmichael_lambda, is_carmichael, tonelli_shanks
)
from mathcore.core.combinatorics import (
    binomial, multinomial, derangements, stirling_second, stirling_first,
    bell_number, catalan_number, narayana_number, partition_count,
    partitions, compositions, bernoulli_numbers, euler_numbers,
    gray_code, gray_code_sequence, lyndon_words, necklaces,
    permutations, combinations_iter, power_set, inclusion_exclusion,
    permanent, motzkin_number, schroder_number
)

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
from mathcore.statistics.descriptive import (
    DescriptiveStatistics, LinearRegression, HypothesisTesting,
    CorrelationAnalysis
)
from mathcore.statistics.inference import (
    ttest_one_sample, ttest_independent, ttest_paired,
    one_way_anova, chi2_goodness_of_fit, chi2_independence,
    mann_whitney_u, wilcoxon_signed_rank, kruskal_wallis,
    confidence_interval_mean, confidence_interval_proportion,
    cohens_d, power_ttest, sample_size_ttest,
    bayesian_beta_binomial, bayesian_normal_normal, credible_interval
)
from mathcore.statistics.regression import (
    OLSRegression, PolynomialRegression, RidgeRegression,
    LassoRegression, LogisticRegression,
    k_fold_cross_validate, vif, forward_stepwise
)
from mathcore.statistics.time_series import (
    simple_moving_average, exponential_moving_average, weighted_moving_average,
    double_exponential_smoothing, holt_winters,
    acf, pacf, ljung_box_test, difference, seasonal_difference, adf_test,
    ARModel, MAModel, ARMAModel,
    seasonal_decompose, periodogram, dominant_frequency, band_pass_filter
)

# ---------------------------------------------------------------------------
# Probability
# ---------------------------------------------------------------------------
from mathcore.probability.distributions import (
    NormalDistribution, BinomialDistribution, PoissonDistribution,
    ExponentialDistribution, UniformDistribution, ChiSquaredDistribution,
    TDistribution
)
from mathcore.probability.markov import (
    MarkovChain, metropolis_hastings, HiddenMarkovModel,
    simple_random_walk, random_walk_2d, brownian_motion, geometric_brownian_motion
)
from mathcore.probability.monte_carlo import (
    monte_carlo_integrate, monte_carlo_integrate_nd,
    importance_sampling, rejection_sampling, bootstrap,
    latin_hypercube_sample, antithetic_variates, stratified_sampling,
    black_scholes_mc, black_scholes_analytic, asian_option_mc,
    value_at_risk, expected_shortfall, simulate_pi
)

# ---------------------------------------------------------------------------
# Complex Numbers, Analysis & Transforms
# ---------------------------------------------------------------------------
from mathcore.complex.numbers import Complex, QuadraticFormula, ComplexAnalysis
from mathcore.complex.analysis import (
    MobiusTransform, complex_exp, complex_log, complex_pow,
    contour_integrate, cauchy_integral_formula, winding_number,
    riemann_zeta, complex_gamma
)
from mathcore.complex.transforms import (
    fft, ifft, rfft, fft_convolve, dct, haar_wavelet_transform,
    z_transform_unit_circle, stft, hann_window
)

# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------
from mathcore.optimization.algorithms import (
    GradientDescent, NewtonsMethod, ConjugateGradient,
    SimulatedAnnealing, ParticleSwarmOptimization, GeneticAlgorithm
)
from mathcore.optimization.linear_programming import (
    SimplexResult, simplex, simplex_standard_form, two_phase_simplex,
    sensitivity_analysis, branch_and_bound,
    transportation_problem, max_flow_ford_fulkerson
)

# ---------------------------------------------------------------------------
# Differential Equations
# ---------------------------------------------------------------------------
from mathcore.differential.ode_solver import (
    ODESolver, PartialDifferentialEquations, StiffODESolver
)
from mathcore.differential.boundary_value import (
    shooting_method_linear, shooting_method_nonlinear,
    finite_difference_bvp, finite_difference_bvp_nonlinear,
    fem_1d, sturm_liouville_eigenvalues,
    chebyshev_collocation, greens_function_bvp
)

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------
from mathcore.utils.step_display import (
    StepTracker, SimplificationTracker, CalculationTracker,
    EquationSolver, ArithmeticSteps, ProbabilitySolver
)

__all__ = [
    # Arithmetic
    'factorial', 'gcd', 'lcm', 'is_prime', 'prime_factors',
    'fibonacci', 'sum_of_divisors', 'euler_totient',
    'binomial_coefficient', 'combination', 'permutation',
    'is_perfect_square', 'power_mod', 'nth_root',
    # Algebra
    'Polynomial', 'simplify', 'solve_quadratic', 'solve_cubic',
    'factor_expression', 'expand_expression',
    # Geometry
    'Point', 'Line', 'Circle', 'Triangle', 'Vector',
    'find_line_intersection', 'distance', 'midpoint', 'slope',
    'line_collision_detection', 'circle_line_intersection', 'point_to_line_distance',
    # Calculus
    'derivative', 'integral', 'limit', 'series_expansion',
    'find_critical_points', 'second_derivative_test',
    'find_root_bisection', 'find_root_newton', 'optimize_minimize',
    'second_derivative',
    # Matrix & Linear Algebra
    'Matrix', 'EigenAnalysis', 'QRDecomposition', 'SingularValueDecomposition',
    'NormCalculations', 'MatrixDecompositions',
    # Number Theory
    'sieve_of_eratosthenes', 'segmented_sieve', 'is_prime_miller_rabin',
    'prime_factorisation', 'divisors', 'extended_gcd', 'modular_inverse',
    'chinese_remainder_theorem', 'euler_totient_nt', 'mobius_function',
    'primitive_root', 'discrete_log_bsgs', 'legendre_symbol', 'jacobi_symbol',
    'continued_fraction', 'continued_fraction_sqrt', 'convergents', 'pell_equation',
    'carmichael_lambda', 'is_carmichael', 'tonelli_shanks',
    # Combinatorics
    'binomial', 'multinomial', 'derangements', 'stirling_second', 'stirling_first',
    'bell_number', 'catalan_number', 'narayana_number', 'partition_count',
    'partitions', 'compositions', 'bernoulli_numbers', 'euler_numbers',
    'gray_code', 'gray_code_sequence', 'lyndon_words', 'necklaces',
    'permutations', 'combinations_iter', 'power_set', 'inclusion_exclusion',
    'permanent', 'motzkin_number', 'schroder_number',
    # Statistics
    'DescriptiveStatistics', 'LinearRegression', 'HypothesisTesting',
    'CorrelationAnalysis',
    # Statistical Inference
    'ttest_one_sample', 'ttest_independent', 'ttest_paired',
    'one_way_anova', 'chi2_goodness_of_fit', 'chi2_independence',
    'mann_whitney_u', 'wilcoxon_signed_rank', 'kruskal_wallis',
    'confidence_interval_mean', 'confidence_interval_proportion',
    'cohens_d', 'power_ttest', 'sample_size_ttest',
    'bayesian_beta_binomial', 'bayesian_normal_normal', 'credible_interval',
    # Regression
    'OLSRegression', 'PolynomialRegression', 'RidgeRegression',
    'LassoRegression', 'LogisticRegression',
    'k_fold_cross_validate', 'vif', 'forward_stepwise',
    # Time Series
    'simple_moving_average', 'exponential_moving_average', 'weighted_moving_average',
    'double_exponential_smoothing', 'holt_winters',
    'acf', 'pacf', 'ljung_box_test', 'difference', 'seasonal_difference', 'adf_test',
    'ARModel', 'MAModel', 'ARMAModel',
    'seasonal_decompose', 'periodogram', 'dominant_frequency', 'band_pass_filter',
    # Probability
    'NormalDistribution', 'BinomialDistribution', 'PoissonDistribution',
    'ExponentialDistribution', 'UniformDistribution', 'ChiSquaredDistribution',
    'TDistribution',
    # Markov & MCMC
    'MarkovChain', 'metropolis_hastings', 'HiddenMarkovModel',
    'simple_random_walk', 'random_walk_2d', 'brownian_motion', 'geometric_brownian_motion',
    # Monte Carlo
    'monte_carlo_integrate', 'monte_carlo_integrate_nd',
    'importance_sampling', 'rejection_sampling', 'bootstrap',
    'latin_hypercube_sample', 'antithetic_variates', 'stratified_sampling',
    'black_scholes_mc', 'black_scholes_analytic', 'asian_option_mc',
    'value_at_risk', 'expected_shortfall', 'simulate_pi',
    # Complex Numbers, Analysis & Transforms
    'Complex', 'QuadraticFormula', 'ComplexAnalysis',
    'MobiusTransform', 'complex_exp', 'complex_log', 'complex_pow',
    'contour_integrate', 'cauchy_integral_formula', 'winding_number',
    'riemann_zeta', 'complex_gamma',
    'fft', 'ifft', 'rfft', 'fft_convolve', 'dct', 'haar_wavelet_transform',
    'z_transform_unit_circle', 'stft', 'hann_window',
    # Optimization
    'GradientDescent', 'NewtonsMethod', 'ConjugateGradient',
    'SimulatedAnnealing', 'ParticleSwarmOptimization', 'GeneticAlgorithm',
    # Linear Programming
    'SimplexResult', 'simplex', 'simplex_standard_form', 'two_phase_simplex',
    'sensitivity_analysis', 'branch_and_bound',
    'transportation_problem', 'max_flow_ford_fulkerson',
    # Differential Equations
    'ODESolver', 'PartialDifferentialEquations', 'StiffODESolver',
    # Boundary Value Problems
    'shooting_method_linear', 'shooting_method_nonlinear',
    'finite_difference_bvp', 'finite_difference_bvp_nonlinear',
    'fem_1d', 'sturm_liouville_eigenvalues',
    'chebyshev_collocation', 'greens_function_bvp',
    # Utils
    'StepTracker', 'SimplificationTracker', 'CalculationTracker',
    'EquationSolver', 'ArithmeticSteps', 'ProbabilitySolver',
]
