"""
Advanced arithmetic operations and number theory functions.
Optimized for performance using efficient algorithms.
"""

import math
from functools import lru_cache
from typing import List, Tuple, Union


@lru_cache(maxsize=1024)
def factorial(n: int) -> int:
    """Calculate factorial of n using memoization for performance.
    
    Args:
        n: Non-negative integer
        
    Returns:
        n! (factorial of n)
        
    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)


@lru_cache(maxsize=1024)
def gcd(a: int, b: int) -> int:
    """Calculate Greatest Common Divisor using Euclidean algorithm.
    
    Args:
        a, b: Integers
        
    Returns:
        GCD of a and b
    """
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def lcm(a: int, b: int) -> int:
    """Calculate Least Common Multiple.
    
    Args:
        a, b: Integers
        
    Returns:
        LCM of a and b
    """
    return abs(a * b) // gcd(a, b)


@lru_cache(maxsize=10000)
def is_prime(n: int) -> bool:
    """Check if a number is prime using trial division.
    Optimized for performance.
    
    Args:
        n: Integer to check
        
    Returns:
        True if n is prime, False otherwise
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    if n < 9:
        return True
    if n % 3 == 0:
        return False
    
    limit = int(math.sqrt(n))
    for i in range(5, limit + 1, 6):
        if n % i == 0 or n % (i + 2) == 0:
            return False
    return True


def prime_factors(n: int) -> List[int]:
    """Find prime factorization of n.
    
    Args:
        n: Positive integer
        
    Returns:
        List of prime factors
    """
    if n <= 0:
        raise ValueError("Prime factorization requires positive integer")
    
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


@lru_cache(maxsize=1000)
def fibonacci(n: int) -> int:
    """Calculate nth Fibonacci number using memoization.
    
    Args:
        n: Non-negative integer
        
    Returns:
        nth Fibonacci number
    """
    if n < 0:
        raise ValueError("Fibonacci is not defined for negative numbers")
    if n == 0:
        return 0
    if n == 1:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)


def sum_of_divisors(n: int) -> int:
    """Calculate sum of all divisors of n.
    
    Args:
        n: Positive integer
        
    Returns:
        Sum of divisors
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    
    total = 0
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            total += i
            if i != n // i:
                total += n // i
    return total


def euler_totient(n: int) -> int:
    """Calculate Euler's totient function φ(n).
    Returns count of integers from 1 to n that are coprime with n.
    
    Args:
        n: Positive integer
        
    Returns:
        φ(n)
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    
    result = n
    p = 2
    while p * p <= n:
        if n % p == 0:
            while n % p == 0:
                n //= p
            result -= result // p
        p += 1
    if n > 1:
        result -= result // n
    return result


def is_perfect_square(n: int) -> bool:
    """Check if n is a perfect square.
    
    Args:
        n: Non-negative integer
        
    Returns:
        True if n is a perfect square
    """
    if n < 0:
        return False
    root = int(math.sqrt(n))
    return root * root == n


def binomial_coefficient(n: int, k: int) -> int:
    """Calculate binomial coefficient C(n, k) = n! / (k! * (n-k)!)
    
    Args:
        n, k: Non-negative integers with k <= n
        
    Returns:
        C(n, k)
    """
    if k > n or k < 0:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def power_mod(base: int, exp: int, mod: int) -> int:
    """Calculate (base^exp) % mod efficiently using modular exponentiation.
    
    Args:
        base: Base value
        exp: Exponent (non-negative)
        mod: Modulus
        
    Returns:
        (base^exp) % mod
    """
    if mod == 1:
        return 0
    result = 1
    base = base % mod
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp = exp >> 1
        base = (base * base) % mod
    return result


def nth_root(x: float, n: int) -> float:
    """Calculate the nth root of x.
    
    Args:
        x: Non-negative number
        n: Positive integer (root degree)
        
    Returns:
        nth root of x
    """
    if x < 0 and n % 2 == 0:
        raise ValueError("Cannot take even root of negative number")
    if n == 0:
        raise ValueError("Root degree must be positive")
    
    return x ** (1 / n)


def combination(n: int, r: int) -> int:
    """Calculate combinations C(n,r).
    
    Args:
        n: Total items
        r: Items to choose
        
    Returns:
        Number of combinations
    """
    return binomial_coefficient(n, r)


def permutation(n: int, r: int) -> int:
    """Calculate permutations P(n,r) = n! / (n-r)!
    
    Args:
        n: Total items
        r: Items to arrange
        
    Returns:
        Number of permutations
    """
    if r > n or r < 0:
        return 0
    result = 1
    for i in range(n, n - r, -1):
        result *= i
    return result
