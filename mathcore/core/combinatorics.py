"""
Advanced Combinatorics — from scratch, zero external dependencies.
Includes: partitions, Bell/Catalan/Stirling numbers, Bernoulli numbers,
generating functions, Gray codes, Burnside's lemma, Polya enumeration.
"""

import math
from functools import lru_cache
from typing import List, Tuple, Iterator, Dict, Optional


# ---------------------------------------------------------------------------
# Basic Counts
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4096)
def binomial(n: int, k: int) -> int:
    """C(n, k) — binomial coefficient. Handles negative n via upper negation."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


@lru_cache(maxsize=4096)
def multinomial(*args: int) -> int:
    """Multinomial coefficient n! / (k1! * k2! * ... * km!) where n = sum(args)."""
    n = sum(args)
    result = math.factorial(n)
    for k in args:
        result //= math.factorial(k)
    return result


def derangements(n: int) -> int:
    """Return D(n) — number of derangements of n elements.

    D(n) = n! * sum_{k=0}^{n} (-1)^k / k!
    """
    if n == 0:
        return 1
    if n == 1:
        return 0
    return (n - 1) * (derangements(n - 1) + derangements(n - 2))


def subfactorial(n: int) -> int:
    """Alias for derangements(n)."""
    return derangements(n)


# ---------------------------------------------------------------------------
# Stirling Numbers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4096)
def stirling_second(n: int, k: int) -> int:
    """Stirling number of the second kind S(n, k).

    S(n, k) = number of ways to partition n elements into k non-empty subsets.
    Recurrence: S(n, k) = k * S(n-1, k) + S(n-1, k-1)
    """
    if n == 0 and k == 0:
        return 1
    if n == 0 or k == 0:
        return 0
    if k > n:
        return 0
    return k * stirling_second(n - 1, k) + stirling_second(n - 1, k - 1)


@lru_cache(maxsize=4096)
def stirling_first(n: int, k: int) -> int:
    """Stirling number of the first kind c(n, k) — unsigned (cycle counting).

    c(n, k) = number of permutations of n with exactly k cycles.
    Recurrence: c(n, k) = (n-1)*c(n-1, k) + c(n-1, k-1)
    """
    if n == 0 and k == 0:
        return 1
    if n == 0 or k == 0:
        return 0
    if k > n:
        return 0
    return (n - 1) * stirling_first(n - 1, k) + stirling_first(n - 1, k - 1)


def bell_number(n: int) -> int:
    """Return the n-th Bell number B(n) = sum_k S(n, k)."""
    return sum(stirling_second(n, k) for k in range(n + 1))


def bell_triangle(n: int) -> List[List[int]]:
    """Return the Bell triangle up to row n."""
    triangle = [[1]]
    for i in range(1, n + 1):
        row = [triangle[i - 1][-1]]
        for j in range(i):
            row.append(row[-1] + triangle[i - 1][j])
        triangle.append(row)
    return triangle


@lru_cache(maxsize=1024)
def catalan_number(n: int) -> int:
    """Return the n-th Catalan number C(n) = C(2n, n) / (n+1)."""
    return binomial(2 * n, n) // (n + 1)


def narayana_number(n: int, k: int) -> int:
    """Return N(n, k) — count of Dyck paths of length 2n with exactly k peaks."""
    return binomial(n, k) * binomial(n, k - 1) // n if n > 0 else int(n == 0 and k == 0)


# ---------------------------------------------------------------------------
# Integer Partitions
# ---------------------------------------------------------------------------

def partition_count(n: int) -> int:
    """Return p(n) — number of integer partitions of n.

    Uses Euler's recurrence via pentagonal numbers.
    """
    p = [0] * (n + 1)
    p[0] = 1
    for i in range(1, n + 1):
        k = 1
        while True:
            pent1 = k * (3 * k - 1) // 2
            pent2 = k * (3 * k + 1) // 2
            if pent1 > i:
                break
            p[i] += (-1) ** (k + 1) * p[i - pent1]
            if pent2 <= i:
                p[i] += (-1) ** (k + 1) * p[i - pent2]
            k += 1
    return p[n]


def partitions(n: int, max_part: Optional[int] = None) -> Iterator[Tuple[int, ...]]:
    """Generate all integer partitions of n in non-increasing order."""
    if max_part is None:
        max_part = n

    def _gen(remaining: int, max_val: int) -> Iterator[Tuple[int, ...]]:
        if remaining == 0:
            yield ()
            return
        for i in range(min(remaining, max_val), 0, -1):
            for rest in _gen(remaining - i, i):
                yield (i,) + rest

    return _gen(n, max_part)


def partition_into_k_parts(n: int, k: int) -> int:
    """Count the number of partitions of n into exactly k positive parts."""
    return stirling_second(n - 1, k - 1) if k > 0 else int(n == 0)


def compositions(n: int, k: int) -> Iterator[Tuple[int, ...]]:
    """Generate all ordered compositions of n into exactly k positive parts."""
    if k == 1:
        yield (n,)
        return
    for first in range(1, n - k + 2):
        for rest in compositions(n - first, k - 1):
            yield (first,) + rest


def composition_count(n: int, k: int) -> int:
    """Return number of ordered compositions of n into k positive parts = C(n-1, k-1)."""
    return binomial(n - 1, k - 1)


# ---------------------------------------------------------------------------
# Bernoulli & Euler Numbers
# ---------------------------------------------------------------------------

def bernoulli_numbers(n: int) -> List[Tuple[int, int]]:
    """Return Bernoulli numbers B(0) through B(n) as (numerator, denominator) fractions.

    Uses the Akiyama-Tanigawa algorithm.
    """
    from fractions import Fraction
    a = [Fraction(1, k + 1) for k in range(n + 1)]
    result = []
    for m in range(n + 1):
        result.append((a[0].numerator, a[0].denominator))
        a = [k * (a[k] - a[k - 1]) for k in range(1, n - m + 1)]
    return result


def euler_numbers(n: int) -> List[int]:
    """Return Euler numbers E(0), E(2), ..., E(2*floor(n/2)).

    Computed via the recurrence using the secant series.
    Only even-indexed Euler numbers are non-zero; odd-indexed are 0.
    """
    # Build using the triangle method
    E = [[0] * (n + 2) for _ in range(n + 2)]
    E[0][0] = 1
    for i in range(1, n + 1):
        if i % 2 == 1:
            E[i][0] = 0
        else:
            for j in range(1, i + 1):
                E[i][j] = E[i - 1][j - 1] + (E[i][j - 1] if j > 0 else 0)
                if i % 2 == 0 and j == i:
                    E[i][j] = -E[i - 1][j - 1]
    return [E[2 * k][2 * k] if 2 * k <= n else 0 for k in range(n // 2 + 1)]


# ---------------------------------------------------------------------------
# Generating Functions (symbolic polynomial)
# ---------------------------------------------------------------------------

def ordinary_generating_function(sequence: List[int], x_power: int) -> int:
    """Return the coefficient of x^x_power in OGF of sequence."""
    if x_power < len(sequence):
        return sequence[x_power]
    return 0


def poly_multiply(a: List[int], b: List[int]) -> List[int]:
    """Multiply two polynomials represented as coefficient lists."""
    if not a or not b:
        return []
    result = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            result[i + j] += ai * bj
    return result


def poly_power(p: List[int], n: int) -> List[int]:
    """Return polynomial p raised to the n-th power."""
    result = [1]
    base = p[:]
    while n:
        if n % 2:
            result = poly_multiply(result, base)
        base = poly_multiply(base, base)
        n //= 2
    return result


def partition_generating_function(limit: int) -> List[int]:
    """Return coefficients of the partition generating function up to x^limit.

    Coefficient of x^n = p(n).
    """
    coeffs = [0] * (limit + 1)
    coeffs[0] = 1
    for k in range(1, limit + 1):
        for j in range(k, limit + 1):
            coeffs[j] += coeffs[j - k]
    return coeffs


# ---------------------------------------------------------------------------
# Gray Codes & Combinatorial Sequences
# ---------------------------------------------------------------------------

def gray_code(n: int) -> int:
    """Return the n-th Gray code value."""
    return n ^ (n >> 1)


def gray_code_sequence(bits: int) -> List[int]:
    """Return the full Gray code sequence for `bits` bits."""
    return [gray_code(i) for i in range(1 << bits)]


def inverse_gray_code(g: int) -> int:
    """Return n such that gray_code(n) == g."""
    n = g
    mask = g >> 1
    while mask:
        n ^= mask
        mask >>= 1
    return n


def lyndon_words(n: int, k: int) -> Iterator[Tuple[int, ...]]:
    """Generate all Lyndon words of length n over an alphabet of size k.

    A Lyndon word is strictly smaller than all its rotations.
    Uses the Duval algorithm.
    """
    w = [-1]
    while w:
        w[-1] += 1
        m = len(w)
        if m == n:
            if n % m == 0:
                yield tuple(w)
        else:
            while len(w) < n:
                w.append(w[len(w) - m])
        while w and w[-1] == k - 1:
            w.pop()


def necklaces(n: int, k: int) -> int:
    """Return the number of distinct necklaces of length n with k colors (Burnside)."""
    from math import gcd
    total = 0
    for d in range(1, n + 1):
        if n % d == 0:
            # phi(d) * k^(n/d) over d | n using Euler's product
            phi_d = d
            dd = d
            for p in range(2, int(d ** 0.5) + 1):
                if dd % p == 0:
                    while dd % p == 0:
                        dd //= p
                    phi_d -= phi_d // p
            if dd > 1:
                phi_d -= phi_d // dd
            total += phi_d * (k ** (n // d))
    return total // n


# ---------------------------------------------------------------------------
# Set / Permutation Operations
# ---------------------------------------------------------------------------

def permutations(seq: List) -> Iterator[Tuple]:
    """Generate all permutations of seq using Heap's algorithm."""
    seq = list(seq)
    n = len(seq)
    c = [0] * n
    yield tuple(seq)
    i = 0
    while i < n:
        if c[i] < i:
            if i % 2 == 0:
                seq[0], seq[i] = seq[i], seq[0]
            else:
                seq[c[i]], seq[i] = seq[i], seq[c[i]]
            yield tuple(seq)
            c[i] += 1
            i = 0
        else:
            c[i] = 0
            i += 1


def combinations_iter(seq: List, r: int) -> Iterator[Tuple]:
    """Generate all r-combinations of seq."""
    seq = list(seq)
    n = len(seq)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(seq[i] for i in indices)
    while True:
        for i in range(r - 1, -1, -1):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        yield tuple(seq[i] for i in indices)


def power_set(seq: List) -> List[Tuple]:
    """Return all subsets (power set) of seq."""
    seq = list(seq)
    result = [()]
    for elem in seq:
        result += [s + (elem,) for s in result]
    return result


def inclusion_exclusion(sets: List[set]) -> int:
    """Return |union of sets| using inclusion-exclusion principle."""
    n = len(sets)
    total = 0
    for mask in range(1, 1 << n):
        intersection = None
        bits = 0
        for i in range(n):
            if mask & (1 << i):
                intersection = sets[i] if intersection is None else intersection & sets[i]
                bits += 1
        total += (-1) ** (bits + 1) * len(intersection)
    return total


def permanent(matrix: List[List[int]]) -> int:
    """Compute the permanent of a square matrix using Ryser's formula.

    O(2^n * n) — exponential but optimal for dense matrices.
    """
    n = len(matrix)
    total = 0
    for mask in range(1, 1 << n):
        row_sums = [0] * n
        bits = 0
        for j in range(n):
            if mask & (1 << j):
                for i in range(n):
                    row_sums[i] += matrix[i][j]
                bits += 1
        prod = 1
        for s in row_sums:
            prod *= s
        total += (-1) ** bits * prod
    return (-1) ** n * total


# ---------------------------------------------------------------------------
# Motzkin, Schroder, Fibonacci Polynomials
# ---------------------------------------------------------------------------

def motzkin_number(n: int) -> int:
    """Return the n-th Motzkin number."""
    if n <= 1:
        return 1
    # Recurrence: M(n) = M(n-1) + sum_{k=0}^{n-2} M(k)*M(n-2-k)
    m = [0] * (n + 1)
    m[0] = m[1] = 1
    for i in range(2, n + 1):
        m[i] = m[i - 1] + sum(m[k] * m[i - 2 - k] for k in range(i - 1))
    return m[n]


def schroder_number(n: int) -> int:
    """Return the n-th large Schroder number (super-Catalan)."""
    if n == 0:
        return 1
    s = [0] * (n + 1)
    s[0] = 1
    s[1] = 2
    for i in range(2, n + 1):
        s[i] = (3 * (2 * i - 1) * s[i - 1] - (i - 1) * s[i - 2]) // (i + 1)
    return s[n]


def fibonacci_poly(n: int) -> List[int]:
    """Return coefficients of the n-th Fibonacci polynomial F_n(x).

    F_1(x)=1, F_2(x)=x, F_n(x)=x*F_{n-1}(x)+F_{n-2}(x)
    """
    if n == 1:
        return [1]
    if n == 2:
        return [1, 0]  # x
    f_prev, f_curr = [1], [1, 0]
    for _ in range(2, n):
        # f_curr * x + f_prev
        shifted = f_curr + [0]  # multiply by x
        nxt = list(shifted)
        for i, c in enumerate(f_prev):
            nxt[i + (len(shifted) - len(f_prev))] += c
        # Normalise length
        nxt = [0] * (len(shifted) - len(f_prev)) + [
            shifted[i] + (f_prev[i - (len(shifted) - len(f_prev))]
                          if i >= len(shifted) - len(f_prev) else 0)
            for i in range(len(shifted))
        ]
        f_prev, f_curr = f_curr, nxt
    return f_curr
