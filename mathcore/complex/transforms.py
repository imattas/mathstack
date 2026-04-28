"""
Signal Transforms — from scratch, zero external dependencies.
Includes: DFT, FFT (Cooley-Tukey), IFFT, real FFT, FFT convolution,
Z-transform (numerical), Haar wavelet transform, DCT, DST.
"""

import math
from typing import List, Tuple, Callable, Optional
from mathcore.complex.numbers import Complex


# ---------------------------------------------------------------------------
# DFT / FFT
# ---------------------------------------------------------------------------

def dft(x: List[float]) -> List[Complex]:
    """Discrete Fourier Transform (naive O(n^2) implementation).

    X[k] = sum_{n=0}^{N-1} x[n] * exp(-2*pi*i*k*n/N)
    """
    n = len(x)
    result = []
    for k in range(n):
        total = Complex(0, 0)
        for j, xj in enumerate(x):
            angle = -2 * math.pi * k * j / n
            total = total + Complex(xj * math.cos(angle), xj * math.sin(angle))
        result.append(total)
    return result


def idft(X: List[Complex]) -> List[Complex]:
    """Inverse DFT (naive O(n^2)).

    x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * exp(2*pi*i*k*n/N)
    """
    n = len(X)
    result = []
    for j in range(n):
        total = Complex(0, 0)
        for k, Xk in enumerate(X):
            angle = 2 * math.pi * k * j / n
            total = total + Xk * Complex(math.cos(angle), math.sin(angle))
        result.append(total * Complex(1 / n, 0))
    return result


def _fft_recursive(x: List[Complex], inverse: bool = False) -> List[Complex]:
    """Cooley-Tukey radix-2 FFT/IFFT (recursive). n must be a power of 2."""
    n = len(x)
    if n == 1:
        return x[:]
    even = _fft_recursive(x[::2], inverse)
    odd = _fft_recursive(x[1::2], inverse)
    sign = 1 if inverse else -1
    result = [Complex(0, 0)] * n
    for k in range(n // 2):
        angle = sign * 2 * math.pi * k / n
        w = Complex(math.cos(angle), math.sin(angle))
        t = w * odd[k]
        result[k] = even[k] + t
        result[k + n // 2] = even[k] - t
    return result


def _next_power_of_2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def fft(x: List[float]) -> List[Complex]:
    """Fast Fourier Transform (Cooley-Tukey radix-2, O(n log n)).

    Zero-pads input to the next power of 2 if necessary.
    Returns full spectrum of length 2^ceil(log2(n)).
    """
    n = _next_power_of_2(len(x))
    padded = [Complex(v, 0) for v in x] + [Complex(0, 0)] * (n - len(x))
    return _fft_recursive(padded, inverse=False)


def ifft(X: List[Complex]) -> List[Complex]:
    """Inverse FFT. Input length must be a power of 2."""
    n = len(X)
    result = _fft_recursive(X[:], inverse=True)
    return [v * Complex(1 / n, 0) for v in result]


def rfft(x: List[float]) -> List[Complex]:
    """Real FFT — returns only the non-redundant half of the spectrum.

    For a real signal of length n, the spectrum is symmetric:
    X[k] = conj(X[n-k]), so only X[0..n//2] are returned.
    """
    full = fft(x)
    return full[: len(x) // 2 + 1]


def fft_frequencies(n: int, sample_rate: float = 1.0) -> List[float]:
    """Return the frequency bins for an FFT of length n at given sample_rate."""
    return [k * sample_rate / n for k in range(n // 2 + 1)]


def fft_magnitude(X: List[Complex]) -> List[float]:
    """Return magnitude |X[k]| of each FFT bin."""
    return [v.magnitude() for v in X]


def fft_phase(X: List[Complex]) -> List[float]:
    """Return phase angle (radians) of each FFT bin."""
    return [v.argument() for v in X]


def fft_power_spectrum(x: List[float]) -> List[float]:
    """Return single-sided power spectrum of real signal x."""
    X = rfft(x)
    n = _next_power_of_2(len(x))
    ps = [v.magnitude() ** 2 / n for v in X]
    # Double all bins except DC and Nyquist (energy conservation)
    for i in range(1, len(ps) - 1):
        ps[i] *= 2
    return ps


# ---------------------------------------------------------------------------
# FFT-based Convolution & Correlation
# ---------------------------------------------------------------------------

def fft_convolve(a: List[float], b: List[float]) -> List[float]:
    """Compute the linear convolution of a and b using FFT (O(n log n)).

    Result has length len(a) + len(b) - 1.
    """
    out_len = len(a) + len(b) - 1
    n = _next_power_of_2(out_len)
    A = _fft_recursive(
        [Complex(v, 0) for v in a] + [Complex(0, 0)] * (n - len(a)), False)
    B = _fft_recursive(
        [Complex(v, 0) for v in b] + [Complex(0, 0)] * (n - len(b)), False)
    C = [ai * bi for ai, bi in zip(A, B)]
    result_full = ifft(C)
    return [v.real for v in result_full[:out_len]]


def fft_correlate(a: List[float], b: List[float]) -> List[float]:
    """Compute cross-correlation of a and b using FFT.

    corr[k] = sum_n a[n] * b[n+k]
    """
    n = _next_power_of_2(len(a) + len(b) - 1)
    A = _fft_recursive([Complex(v, 0) for v in a] + [Complex(0, 0)] * (n - len(a)), False)
    B = _fft_recursive([Complex(v, 0) for v in b] + [Complex(0, 0)] * (n - len(b)), False)
    # Cross-correlation = IFFT(conj(A) * B)
    C = [Complex(ai.real, -ai.imag) * bi for ai, bi in zip(A, B)]
    result_full = ifft(C)
    return [v.real for v in result_full]


def autocorrelate(x: List[float]) -> List[float]:
    """Compute the autocorrelation of signal x."""
    return fft_correlate(x, x)


# ---------------------------------------------------------------------------
# Discrete Cosine Transform (DCT-II, the "JPEG" DCT)
# ---------------------------------------------------------------------------

def dct(x: List[float]) -> List[float]:
    """Type-II DCT: X[k] = 2 * sum_{n=0}^{N-1} x[n] * cos(pi*k*(2n+1)/(2N))."""
    n = len(x)
    result = []
    for k in range(n):
        total = sum(x[j] * math.cos(math.pi * k * (2 * j + 1) / (2 * n))
                    for j in range(n))
        result.append(2 * total)
    return result


def idct(X: List[float]) -> List[float]:
    """Inverse Type-II DCT."""
    n = len(X)
    result = []
    for j in range(n):
        total = X[0] / 2 + sum(X[k] * math.cos(math.pi * k * (2 * j + 1) / (2 * n))
                                for k in range(1, n))
        result.append(total / n)
    return result


# ---------------------------------------------------------------------------
# Discrete Sine Transform (DST-I)
# ---------------------------------------------------------------------------

def dst(x: List[float]) -> List[float]:
    """Type-I DST: X[k] = 2 * sum_{n=0}^{N-1} x[n] * sin(pi*(k+1)*(n+1)/(N+1))."""
    n = len(x)
    result = []
    for k in range(n):
        total = sum(x[j] * math.sin(math.pi * (k + 1) * (j + 1) / (n + 1))
                    for j in range(n))
        result.append(2 * total)
    return result


# ---------------------------------------------------------------------------
# Haar Wavelet Transform
# ---------------------------------------------------------------------------

def haar_wavelet_transform(x: List[float]) -> List[float]:
    """In-place 1D Haar wavelet transform. Length must be a power of 2.

    Returns [approximation coefficients | detail coefficients] at all scales.
    """
    x = list(x)
    n = len(x)
    if n & (n - 1):
        raise ValueError("Length must be a power of 2")
    step = n
    while step > 1:
        half = step // 2
        new = [0.0] * step
        for i in range(half):
            new[i] = (x[2 * i] + x[2 * i + 1]) / math.sqrt(2)
            new[i + half] = (x[2 * i] - x[2 * i + 1]) / math.sqrt(2)
        x[:step] = new
        step = half
    return x


def inverse_haar_wavelet_transform(coeffs: List[float]) -> List[float]:
    """Inverse 1D Haar wavelet transform."""
    x = list(coeffs)
    n = len(x)
    step = 2
    while step <= n:
        half = step // 2
        orig = x[:step]
        for i in range(half):
            x[2 * i] = (orig[i] + orig[i + half]) / math.sqrt(2)
            x[2 * i + 1] = (orig[i] - orig[i + half]) / math.sqrt(2)
        step *= 2
    return x


def haar_threshold(coeffs: List[float], threshold: float) -> List[float]:
    """Apply hard thresholding to Haar coefficients (denoising)."""
    return [c if abs(c) >= threshold else 0.0 for c in coeffs]


# ---------------------------------------------------------------------------
# Z-Transform (numerical evaluation on unit circle)
# ---------------------------------------------------------------------------

def z_transform_evaluate(
    x: List[float],
    z: Complex,
    causal: bool = True
) -> Complex:
    """Evaluate the Z-transform X(z) = sum_{n=0}^{N-1} x[n] * z^{-n} at a point z.

    For causal sequences only (non-negative time indices).
    """
    result = Complex(0, 0)
    z_inv = Complex(1, 0) / z
    z_pow = Complex(1, 0)
    for xn in x:
        result = result + Complex(xn, 0) * z_pow
        z_pow = z_pow * z_inv
    return result


def z_transform_unit_circle(x: List[float], n_points: int = 512) -> List[Tuple[float, Complex]]:
    """Sample the Z-transform on the unit circle at n_points evenly spaced angles.

    This is equivalent to the DFT, returned as [(angle, X(e^{i*angle})], ...].
    """
    result = []
    for k in range(n_points):
        angle = 2 * math.pi * k / n_points
        z = Complex(math.cos(angle), math.sin(angle))
        Xz = z_transform_evaluate(x, z)
        result.append((angle, Xz))
    return result


# ---------------------------------------------------------------------------
# Short-Time Fourier Transform (STFT)
# ---------------------------------------------------------------------------

def stft(
    x: List[float],
    window_size: int,
    hop_size: int,
    window_fn: Optional[Callable[[int], List[float]]] = None
) -> List[List[Complex]]:
    """Short-Time Fourier Transform.

    Returns a list of FFT spectra, one per frame.
    window_fn(n) should return a list of n window weights (default: Hann window).
    """
    if window_fn is None:
        def window_fn(n):
            return [0.5 * (1 - math.cos(2 * math.pi * i / (n - 1))) for i in range(n)]

    win = window_fn(window_size)
    frames = []
    pos = 0
    while pos + window_size <= len(x):
        frame = [x[pos + i] * win[i] for i in range(window_size)]
        frames.append(fft(frame))
        pos += hop_size
    return frames


def hann_window(n: int) -> List[float]:
    """Return a Hann window of length n."""
    return [0.5 * (1 - math.cos(2 * math.pi * i / (n - 1))) for i in range(n)]


def hamming_window(n: int) -> List[float]:
    """Return a Hamming window of length n."""
    return [0.54 - 0.46 * math.cos(2 * math.pi * i / (n - 1)) for i in range(n)]


def blackman_window(n: int) -> List[float]:
    """Return a Blackman window of length n."""
    return [
        0.42 - 0.5 * math.cos(2 * math.pi * i / (n - 1))
        + 0.08 * math.cos(4 * math.pi * i / (n - 1))
        for i in range(n)
    ]
