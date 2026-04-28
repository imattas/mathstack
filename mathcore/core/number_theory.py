"""
Advanced Number Theory — from scratch, zero external dependencies.
Includes: Miller-Rabin, Pollard's rho, Sieve, CRT, totient, Mobius,
continued fractions, Pell's equation, discrete logarithm, primitive roots.
"""

import math
import random
from typing import List, Tuple, Optional, Dict, Generator


# ---------------------------------------------------------------------------
# Primality
# ---------------------------------------------------------------------------

def sieve_of_eratosthenes(limit: int) -> List[int]:
    """Return all primes up to `limit` using the Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i * i :: i] = bytearray(len(is_prime[i * i :: i]))
    return [i for i, v in enumerate(is_prime) if v]


def segmented_sieve(low: int, high: int) -> List[int]:
    """Return primes in [low, high] using the segmented sieve."""
    if high < 2:
        return []
    limit = int(high ** 0.5) + 1
    base_primes = sieve_of_eratosthenes(limit)
    size = high - low + 1
    is_prime = bytearray([1]) * size
    if low == 0:
        is_prime[0] = 0
    if low <= 1:
        is_prime[1 - low] = 0
    for p in base_primes:
        start = max(p * p, ((low + p - 1) // p) * p)
        for j in range(start, high + 1, p):
            is_prime[j - low] = 0
    return [low + i for i, v in enumerate(is_prime) if v]


def _miller_rabin_check(n: int, a: int) -> bool:
    """Single Miller-Rabin witness check."""
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    x = pow(a, d, n)
    if x == 1 or x == n - 1:
        return True
    for _ in range(r - 1):
        x = x * x % n
        if x == n - 1:
            return True
    return False


def is_prime_miller_rabin(n: int, deterministic: bool = True) -> bool:
    """Primality test using Miller-Rabin.

    For n < 3,317,044,064,679,887,385,961,981, the fixed witness set gives
    a *deterministic* result (no false positives).
    """
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    if n in small_primes:
        return True
    if any(n % p == 0 for p in small_primes):
        return False
    # Deterministic witnesses covering all n < 3.3 * 10^24
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    return all(_miller_rabin_check(n, a) for a in witnesses if a < n)


def next_prime(n: int) -> int:
    """Return the smallest prime strictly greater than n."""
    candidate = n + 1 if n % 2 == 0 else n + 2
    if candidate == 2:
        return 2
    if candidate % 2 == 0:
        candidate += 1
    while not is_prime_miller_rabin(candidate):
        candidate += 2
    return candidate


def nth_prime(n: int) -> int:
    """Return the n-th prime (1-indexed)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    count, candidate = 0, 1
    while count < n:
        candidate += 1
        if is_prime_miller_rabin(candidate):
            count += 1
    return candidate


def prime_counting(n: int) -> int:
    """Return pi(n) — the number of primes <= n."""
    return len(sieve_of_eratosthenes(n))


# ---------------------------------------------------------------------------
# Factorisation
# ---------------------------------------------------------------------------

def _pollard_rho(n: int) -> int:
    """Pollard's rho algorithm — returns a non-trivial factor of n."""
    if n % 2 == 0:
        return 2
    x = random.randint(2, n - 1)
    y, c, d = x, random.randint(1, n - 1), 1
    while d == 1:
        x = (x * x + c) % n
        y = (y * y + c) % n
        y = (y * y + c) % n
        d = math.gcd(abs(x - y), n)
    return d if d != n else None


def prime_factorisation(n: int) -> Dict[int, int]:
    """Return prime factorisation of n as {prime: exponent} dict."""
    if n <= 1:
        return {}
    factors: Dict[int, int] = {}

    def _factorise(m: int) -> None:
        if m == 1:
            return
        if is_prime_miller_rabin(m):
            factors[m] = factors.get(m, 0) + 1
            return
        # Try small primes first
        for p in [2, 3, 5, 7, 11, 13]:
            if m % p == 0:
                while m % p == 0:
                    factors[p] = factors.get(p, 0) + 1
                    m //= p
                _factorise(m)
                return
        # Pollard's rho
        d = None
        while d is None or d == m:
            d = _pollard_rho(m)
        _factorise(d)
        _factorise(m // d)

    _factorise(n)
    return factors


def divisors(n: int) -> List[int]:
    """Return sorted list of all positive divisors of n."""
    if n <= 0:
        raise ValueError("n must be positive")
    divs = set()
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    return sorted(divs)


def sum_of_divisors(n: int, k: int = 1) -> int:
    """Return sigma_k(n) = sum of k-th powers of divisors of n."""
    return sum(d ** k for d in divisors(n))


def is_perfect(n: int) -> bool:
    """Return True if n is a perfect number (sigma_1(n) == 2n)."""
    return n > 1 and sum_of_divisors(n) == 2 * n


def is_abundant(n: int) -> bool:
    """Return True if sum of proper divisors > n."""
    return sum_of_divisors(n) - n > n


def amicable_pairs(limit: int) -> List[Tuple[int, int]]:
    """Return all amicable pairs (a, b) with a < b <= limit."""
    s = {n: sum_of_divisors(n) - n for n in range(1, limit + 1)}
    return [(a, b) for a in range(2, limit + 1)
            if (b := s[a]) > a and b <= limit and s.get(b) == a]


# ---------------------------------------------------------------------------
# Modular Arithmetic
# ---------------------------------------------------------------------------

def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """Extended Euclidean Algorithm — returns (gcd, x, y) where ax + by = gcd."""
    if b == 0:
        return a, 1, 0
    g, x, y = extended_gcd(b, a % b)
    return g, y, x - (a // b) * y


def modular_inverse(a: int, m: int) -> int:
    """Return x such that a*x ≡ 1 (mod m). Raises ValueError if no inverse."""
    g, x, _ = extended_gcd(a % m, m)
    if g != 1:
        raise ValueError(f"No modular inverse: gcd({a}, {m}) = {g} != 1")
    return x % m


def chinese_remainder_theorem(remainders: List[int], moduli: List[int]) -> int:
    """Solve the system x ≡ r_i (mod m_i) using CRT.

    Moduli must be pairwise coprime.
    Returns the unique solution in [0, product of moduli).
    """
    if len(remainders) != len(moduli):
        raise ValueError("remainders and moduli must have the same length")
    M = 1
    for m in moduli:
        M *= m
    x = 0
    for r, m in zip(remainders, moduli):
        Mi = M // m
        x += r * Mi * modular_inverse(Mi, m)
    return x % M


def power_tower(base: int, exp: int, mod: int) -> int:
    """Compute base^exp mod m efficiently using fast exponentiation."""
    return pow(base, exp, mod)


def euler_totient(n: int) -> int:
    """Return Euler's totient phi(n) — count of integers in [1,n] coprime to n."""
    if n <= 0:
        raise ValueError("n must be positive")
    result = n
    factors = prime_factorisation(n)
    for p in factors:
        result -= result // p
    return result


def totient_sieve(limit: int) -> List[int]:
    """Return phi(i) for i in [0, limit] using a sieve."""
    phi = list(range(limit + 1))
    for i in range(2, limit + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, limit + 1, i):
                phi[j] -= phi[j] // i
    return phi


def mobius_function(n: int) -> int:
    """Return the Mobius function mu(n):
      0 if n has a squared prime factor,
     -1 if n has an odd number of distinct prime factors,
     +1 if n has an even number of distinct prime factors.
    """
    factors = prime_factorisation(n)
    if any(e > 1 for e in factors.values()):
        return 0
    return (-1) ** len(factors)


def mobius_sieve(limit: int) -> List[int]:
    """Return mu(i) for i in [0, limit] using a linear sieve."""
    mu = [0] * (limit + 1)
    mu[1] = 1
    is_prime = [True] * (limit + 1)
    primes = []
    for i in range(2, limit + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > limit:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            mu[i * p] = -mu[i]
    return mu


def primitive_root(p: int) -> int:
    """Find the smallest primitive root modulo prime p."""
    if not is_prime_miller_rabin(p):
        raise ValueError(f"{p} must be prime")
    if p == 2:
        return 1
    phi = p - 1
    factors = list(prime_factorisation(phi).keys())
    for g in range(2, p):
        if all(pow(g, phi // f, p) != 1 for f in factors):
            return g
    raise ValueError(f"No primitive root found for {p}")


def discrete_log_bsgs(g: int, h: int, p: int) -> Optional[int]:
    """Baby-step Giant-step algorithm for discrete logarithm.

    Finds x such that g^x ≡ h (mod p), or None if no solution.
    Runs in O(sqrt(p)) time and space.
    """
    m = int(p ** 0.5) + 1
    # Baby steps: build table {g^j mod p -> j}
    table = {}
    gj = 1
    for j in range(m):
        table[gj] = j
        gj = gj * g % p
    # Giant steps
    gm_inv = modular_inverse(pow(g, m, p), p)
    gamma = h
    for i in range(m):
        if gamma in table:
            x = i * m + table[gamma]
            if x > 0:
                return x
        gamma = gamma * gm_inv % p
    return None


def legendre_symbol(a: int, p: int) -> int:
    """Return the Legendre symbol (a/p) for odd prime p: 1, -1, or 0."""
    if not is_prime_miller_rabin(p) or p == 2:
        raise ValueError("p must be an odd prime")
    ls = pow(a, (p - 1) // 2, p)
    return -1 if ls == p - 1 else ls


def jacobi_symbol(a: int, n: int) -> int:
    """Return the Jacobi symbol (a/n) for odd positive n."""
    if n <= 0 or n % 2 == 0:
        raise ValueError("n must be a positive odd integer")
    a %= n
    result = 1
    while a != 0:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                result = -result
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a %= n
    return result if n == 1 else 0


# ---------------------------------------------------------------------------
# Continued Fractions
# ---------------------------------------------------------------------------

def continued_fraction(n: int, d: int) -> List[int]:
    """Return continued fraction expansion [a0; a1, a2, ...] of n/d."""
    result = []
    while d:
        result.append(n // d)
        n, d = d, n % d
    return result


def continued_fraction_sqrt(n: int) -> Tuple[int, List[int]]:
    """Return (a0, [a1, a2, ...]) periodic CF expansion of sqrt(n).

    Returns ([a0], []) if n is a perfect square.
    """
    a0 = int(n ** 0.5)
    if a0 * a0 == n:
        return a0, []
    period = []
    m, d, a = 0, 1, a0
    while True:
        m = d * a - m
        d = (n - m * m) // d
        a = (a0 + m) // d
        period.append(a)
        if a == 2 * a0:
            break
    return a0, period


def convergents(cf: List[int]) -> List[Tuple[int, int]]:
    """Return list of convergents (numerator, denominator) for a continued fraction."""
    result = []
    h_prev, h_curr = 1, cf[0]
    k_prev, k_curr = 0, 1
    result.append((h_curr, k_curr))
    for a in cf[1:]:
        h_prev, h_curr = h_curr, a * h_curr + h_prev
        k_prev, k_curr = k_curr, a * k_curr + k_prev
        result.append((h_curr, k_curr))
    return result


def pell_equation(d: int) -> Tuple[int, int]:
    """Find the fundamental solution (x, y) to x^2 - d*y^2 = 1 (Pell's equation)."""
    a0, period = continued_fraction_sqrt(d)
    if not period:
        raise ValueError(f"{d} is a perfect square; Pell's equation has no solution")
    cf = [a0] + period
    # The fundamental solution appears at index len(period)-1 (or 2*len(period)-1)
    idx = len(period) - 1
    if (idx + 1) % 2 == 1:
        idx = 2 * len(period) - 1
    # Extend CF to required length
    extended = [a0] + (period * ((idx // len(period)) + 1))
    convs = convergents(extended[:idx + 1])
    x, y = convs[-1]
    return x, y


# ---------------------------------------------------------------------------
# Special Sequences
# ---------------------------------------------------------------------------

def collatz_sequence(n: int) -> List[int]:
    """Return the Collatz (3n+1) sequence starting at n."""
    seq = [n]
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        seq.append(n)
    return seq


def collatz_length(n: int) -> int:
    """Return the length of the Collatz sequence starting at n."""
    length = 1
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        length += 1
    return length


def lucas_sequence(n: int, p: int = 1, q: int = -1) -> List[int]:
    """Return the first n terms of the Lucas sequence U(p, q).

    With p=1, q=-1 this gives the Fibonacci sequence.
    With p=3, q=2 this gives Mersenne-like sequences.
    """
    if n <= 0:
        return []
    seq = [0, 1]
    for i in range(2, n):
        seq.append(p * seq[-1] - q * seq[-2])
    return seq[:n]


def carmichael_lambda(n: int) -> int:
    """Return the Carmichael function lambda(n) — the exponent of (Z/nZ)*."""
    def _lambda_prime_power(p: int, k: int) -> int:
        if p == 2:
            if k <= 2:
                return 2 ** (k - 1)
            return 2 ** (k - 2)
        return (p - 1) * p ** (k - 1)

    factors = prime_factorisation(n)
    if not factors:
        return 1
    values = [_lambda_prime_power(p, e) for p, e in factors.items()]
    result = values[0]
    for v in values[1:]:
        result = result * v // math.gcd(result, v)
    return result


def is_carmichael(n: int) -> bool:
    """Return True if n is a Carmichael number (composite Fermat pseudoprime)."""
    if n < 2 or is_prime_miller_rabin(n):
        return False
    factors = prime_factorisation(n)
    return (len(factors) >= 3 and
            all(e == 1 for e in factors.values()) and
            all((n - 1) % (p - 1) == 0 for p in factors))


def quadratic_residues(p: int) -> List[int]:
    """Return list of quadratic residues modulo prime p."""
    return sorted({pow(x, 2, p) for x in range(1, p)})


def tonelli_shanks(n: int, p: int) -> Optional[int]:
    """Find x such that x^2 ≡ n (mod p) using Tonelli-Shanks algorithm."""
    if legendre_symbol(n, p) != 1:
        return None
    if p % 4 == 3:
        return pow(n, (p + 1) // 4, p)
    # Factor p-1 as 2^s * q
    q, s = p - 1, 0
    while q % 2 == 0:
        q //= 2
        s += 1
    # Find a quadratic non-residue
    z = 2
    while legendre_symbol(z, p) != -1:
        z += 1
    m, c, t, r = s, pow(z, q, p), pow(n, q, p), pow(n, (q + 1) // 2, p)
    while True:
        if t == 1:
            return r
        i, tmp = 1, t * t % p
        while tmp != 1:
            tmp = tmp * tmp % p
            i += 1
        b = pow(c, 1 << (m - i - 1), p)
        m, c, t, r = i, b * b % p, t * b * b % p, r * b % p
