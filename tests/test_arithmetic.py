"""Unit tests for arithmetic module."""

import math
import pytest
from mathcore.core.arithmetic import (
    factorial, gcd, lcm, is_prime, prime_factors,
    fibonacci, sum_of_divisors, euler_totient,
    binomial_coefficient, combination, permutation,
    nth_root, is_perfect_square, power_mod
)


class TestArithmetic:
    """Tests for arithmetic functions."""
    
    def test_factorial(self):
        assert factorial(0) == 1
        assert factorial(1) == 1
        assert factorial(5) == 120
        assert factorial(10) == 3628800
    
    def test_factorial_invalid(self):
        with pytest.raises(ValueError):
            factorial(-1)
    
    def test_gcd(self):
        assert gcd(48, 18) == 6
        assert gcd(100, 50) == 50
        assert gcd(17, 19) == 1
    
    def test_lcm(self):
        assert lcm(12, 18) == 36
        assert lcm(4, 6) == 12
    
    def test_is_prime(self):
        assert is_prime(2) == True
        assert is_prime(17) == True
        assert is_prime(1) == False
        assert is_prime(4) == False
        assert is_prime(97) == True
    
    def test_prime_factors(self):
        assert prime_factors(12) == [2, 2, 3]
        assert prime_factors(17) == [17]
        assert prime_factors(100) == [2, 2, 5, 5]
    
    def test_fibonacci(self):
        assert fibonacci(0) == 0
        assert fibonacci(1) == 1
        assert fibonacci(6) == 8
        assert fibonacci(10) == 55
    
    def test_binomial_coefficient(self):
        assert binomial_coefficient(5, 2) == 10
        assert binomial_coefficient(10, 3) == 120
        assert binomial_coefficient(5, 0) == 1
    
    def test_permutation(self):
        assert permutation(5, 2) == 20
        assert permutation(5, 3) == 60
    
    def test_nth_root(self):
        assert abs(nth_root(8, 3) - 2) < 1e-10
        assert abs(nth_root(16, 4) - 2) < 1e-10
    
    def test_power_mod(self):
        assert power_mod(2, 10, 1000) == 24
        assert power_mod(3, 5, 7) == 5


class TestNumberTheory:
    """Tests for number theory functions."""
    
    def test_sum_of_divisors(self):
        assert sum_of_divisors(6) == 12  # 1 + 2 + 3 + 6
        assert sum_of_divisors(12) == 28  # 1 + 2 + 3 + 4 + 6 + 12
    
    def test_euler_totient(self):
        assert euler_totient(1) == 1
        assert euler_totient(9) == 6
        assert euler_totient(10) == 4
    
    def test_is_perfect_square(self):
        assert is_perfect_square(16) == True
        assert is_perfect_square(17) == False
        assert is_perfect_square(25) == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
