"""
Markov Chains — from scratch, zero external dependencies.
Includes: discrete Markov chains, steady state, absorption, MCMC,
hidden Markov models (Viterbi/Baum-Welch), random walks.
"""

import math
import random
from typing import List, Tuple, Optional, Dict


# ---------------------------------------------------------------------------
# Markov Chain
# ---------------------------------------------------------------------------

class MarkovChain:
    """Discrete-time Markov chain defined by a stochastic transition matrix."""

    def __init__(self, transition_matrix: List[List[float]]):
        n = len(transition_matrix)
        for i, row in enumerate(transition_matrix):
            if len(row) != n:
                raise ValueError(f"Row {i} has wrong length")
            s = sum(row)
            if abs(s - 1.0) > 1e-8:
                raise ValueError(f"Row {i} does not sum to 1 (sum={s})")
        self.P = [row[:] for row in transition_matrix]
        self.n = n

    def n_step_transition(self, steps: int) -> List[List[float]]:
        """Return the n-step transition matrix P^steps."""
        result = [[1.0 if i == j else 0.0 for j in range(self.n)] for i in range(self.n)]
        base = [row[:] for row in self.P]
        while steps:
            if steps % 2:
                result = self._mat_mul(result, base)
            base = self._mat_mul(base, base)
            steps //= 2
        return result

    def state_after_n_steps(self, initial: List[float], steps: int) -> List[float]:
        """Return distribution after `steps` steps from initial distribution."""
        Pn = self.n_step_transition(steps)
        return [sum(initial[i] * Pn[i][j] for i in range(self.n)) for j in range(self.n)]

    def steady_state(self, tol: float = 1e-12, max_iter: int = 10000) -> List[float]:
        """Find the stationary distribution by power iteration."""
        dist = [1.0 / self.n] * self.n
        for _ in range(max_iter):
            new_dist = [sum(dist[i] * self.P[i][j] for i in range(self.n)) for j in range(self.n)]
            if max(abs(new_dist[j] - dist[j]) for j in range(self.n)) < tol:
                return new_dist
            dist = new_dist
        return dist

    def is_absorbing(self) -> List[int]:
        """Return indices of absorbing states (P[i][i] == 1)."""
        return [i for i in range(self.n) if abs(self.P[i][i] - 1.0) < 1e-10]

    def absorption_probabilities(self) -> Optional[List[List[float]]]:
        """For an absorbing chain, return matrix B where B[i][j] = probability
        of being absorbed into absorbing state j starting from transient state i.
        """
        absorbing = self.is_absorbing()
        transient = [i for i in range(self.n) if i not in absorbing]
        if not absorbing or not transient:
            return None
        t = len(transient)
        r = len(absorbing)
        # Q = transient->transient submatrix, R = transient->absorbing
        Q = [[self.P[transient[i]][transient[j]] for j in range(t)] for i in range(t)]
        R = [[self.P[transient[i]][absorbing[j]] for j in range(r)] for i in range(t)]
        # N = (I - Q)^{-1}
        IQ = [[int(i == j) - Q[i][j] for j in range(t)] for i in range(t)]
        N = self._mat_inv(IQ)
        # B = N * R
        return self._mat_mul(N, R)

    def expected_absorption_time(self) -> Optional[List[float]]:
        """Return expected number of steps to absorption from each transient state."""
        absorbing = self.is_absorbing()
        transient = [i for i in range(self.n) if i not in absorbing]
        if not absorbing or not transient:
            return None
        t = len(transient)
        Q = [[self.P[transient[i]][transient[j]] for j in range(t)] for i in range(t)]
        IQ = [[int(i == j) - Q[i][j] for j in range(t)] for i in range(t)]
        N = self._mat_inv(IQ)
        return [sum(N[i]) for i in range(t)]

    def period(self, state: int) -> int:
        """Return the period of a state (gcd of return times)."""
        # BFS to find all return times
        visited_depths: Dict[int, int] = {state: 0}
        queue = [(state, 0)]
        return_times = []
        while queue:
            node, depth = queue.pop(0)
            for next_state in range(self.n):
                if self.P[node][next_state] > 0:
                    if next_state == state:
                        return_times.append(depth + 1)
                    elif next_state not in visited_depths:
                        visited_depths[next_state] = depth + 1
                        queue.append((next_state, depth + 1))
        if not return_times:
            return 0
        result = return_times[0]
        for rt in return_times[1:]:
            result = math.gcd(result, rt)
        return result

    def simulate(self, initial_state: int, steps: int) -> List[int]:
        """Simulate the chain for `steps` steps. Returns sequence of states."""
        state = initial_state
        trajectory = [state]
        for _ in range(steps):
            r = random.random()
            cumulative = 0.0
            for j, p in enumerate(self.P[state]):
                cumulative += p
                if r <= cumulative:
                    state = j
                    break
            trajectory.append(state)
        return trajectory

    @staticmethod
    def _mat_mul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        n, m, p = len(A), len(A[0]), len(B[0])
        return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]

    @staticmethod
    def _mat_inv(A: List[List[float]]) -> List[List[float]]:
        """Invert a square matrix via Gauss-Jordan elimination."""
        n = len(A)
        aug = [A[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        for col in range(n):
            pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
            aug[col], aug[pivot] = aug[pivot], aug[col]
            if abs(aug[col][col]) < 1e-14:
                raise ValueError("Matrix is singular")
            scale = aug[col][col]
            aug[col] = [v / scale for v in aug[col]]
            for row in range(n):
                if row != col:
                    factor = aug[row][col]
                    aug[row] = [aug[row][c] - factor * aug[col][c] for c in range(2 * n)]
        return [[aug[i][n + j] for j in range(n)] for i in range(n)]


# ---------------------------------------------------------------------------
# MCMC: Metropolis-Hastings
# ---------------------------------------------------------------------------

def metropolis_hastings(
    log_target: callable,
    initial: List[float],
    n_samples: int,
    proposal_std: float = 0.5,
    burn_in: int = 500
) -> List[List[float]]:
    """Metropolis-Hastings MCMC sampler.

    log_target(x): log of the (unnormalised) target density.
    Returns n_samples samples after burn-in.
    """
    d = len(initial)
    current = list(initial)
    log_p_current = log_target(current)
    samples = []
    for i in range(n_samples + burn_in):
        # Gaussian proposal
        proposal = [current[j] + random.gauss(0, proposal_std) for j in range(d)]
        log_p_proposal = log_target(proposal)
        # Accept/reject
        log_alpha = log_p_proposal - log_p_current
        if math.log(random.random() + 1e-300) < log_alpha:
            current = proposal
            log_p_current = log_p_proposal
        if i >= burn_in:
            samples.append(list(current))
    return samples


def gibbs_sampling(
    conditionals: List[callable],
    initial: List[float],
    n_samples: int,
    burn_in: int = 500
) -> List[List[float]]:
    """Gibbs sampler. conditionals[j](x) should sample from p(x_j | x_{-j}).

    Returns n_samples samples after burn-in.
    """
    d = len(initial)
    current = list(initial)
    samples = []
    for i in range(n_samples + burn_in):
        for j in range(d):
            current[j] = conditionals[j](current[:])
        if i >= burn_in:
            samples.append(list(current))
    return samples


# ---------------------------------------------------------------------------
# Hidden Markov Model
# ---------------------------------------------------------------------------

class HiddenMarkovModel:
    """Hidden Markov Model with discrete observations.

    Implements:
      - forward algorithm (likelihood)
      - Viterbi decoding (most likely state sequence)
      - Baum-Welch training (EM)
    """

    def __init__(
        self,
        n_states: int,
        n_obs: int,
        A: Optional[List[List[float]]] = None,
        B: Optional[List[List[float]]] = None,
        pi: Optional[List[float]] = None
    ):
        self.n_states = n_states
        self.n_obs = n_obs
        # Transition matrix
        self.A = A or [[1.0 / n_states] * n_states for _ in range(n_states)]
        # Emission matrix
        self.B = B or [[1.0 / n_obs] * n_obs for _ in range(n_states)]
        # Initial state distribution
        self.pi = pi or [1.0 / n_states] * n_states

    def _log_sum_exp(self, log_vals: List[float]) -> float:
        max_v = max(log_vals)
        return max_v + math.log(sum(math.exp(v - max_v) for v in log_vals) + 1e-300)

    def forward(self, obs: List[int]) -> Tuple[List[List[float]], float]:
        """Forward algorithm. Returns (alpha, log_likelihood)."""
        T = len(obs)
        alpha = [[0.0] * self.n_states for _ in range(T)]
        for s in range(self.n_states):
            alpha[0][s] = self.pi[s] * self.B[s][obs[0]]
        # Normalise
        scale = [sum(alpha[0])]
        alpha[0] = [v / scale[0] for v in alpha[0]]
        for t in range(1, T):
            for s in range(self.n_states):
                alpha[t][s] = sum(alpha[t - 1][k] * self.A[k][s] for k in range(self.n_states)) * self.B[s][obs[t]]
            s_t = sum(alpha[t])
            if s_t < 1e-300:
                s_t = 1e-300
            scale.append(s_t)
            alpha[t] = [v / s_t for v in alpha[t]]
        log_likelihood = sum(math.log(max(s, 1e-300)) for s in scale)
        return alpha, log_likelihood

    def viterbi(self, obs: List[int]) -> List[int]:
        """Viterbi algorithm — most likely state sequence."""
        T = len(obs)
        log_A = [[math.log(max(self.A[i][j], 1e-300)) for j in range(self.n_states)] for i in range(self.n_states)]
        log_B = [[math.log(max(self.B[s][o], 1e-300)) for o in range(self.n_obs)] for s in range(self.n_states)]
        log_pi = [math.log(max(self.pi[s], 1e-300)) for s in range(self.n_states)]
        delta = [log_pi[s] + log_B[s][obs[0]] for s in range(self.n_states)]
        psi = [[0] * self.n_states for _ in range(T)]
        for t in range(1, T):
            new_delta = []
            for s in range(self.n_states):
                scores = [delta[k] + log_A[k][s] for k in range(self.n_states)]
                best_k = max(range(self.n_states), key=lambda k: scores[k])
                psi[t][s] = best_k
                new_delta.append(scores[best_k] + log_B[s][obs[t]])
            delta = new_delta
        # Backtrack
        path = [max(range(self.n_states), key=lambda s: delta[s])]
        for t in range(T - 1, 0, -1):
            path.insert(0, psi[t][path[0]])
        return path

    def baum_welch(self, obs: List[int], max_iter: int = 100, tol: float = 1e-6) -> float:
        """Baum-Welch EM training. Returns final log-likelihood."""
        prev_ll = float("-inf")
        for _ in range(max_iter):
            T = len(obs)
            alpha, ll = self.forward(obs)
            # Backward
            beta = [[0.0] * self.n_states for _ in range(T)]
            beta[T - 1] = [1.0 / self.n_states] * self.n_states
            for t in range(T - 2, -1, -1):
                for s in range(self.n_states):
                    beta[t][s] = sum(self.A[s][k] * self.B[k][obs[t + 1]] * beta[t + 1][k] for k in range(self.n_states))
                b_sum = sum(beta[t])
                if b_sum > 0:
                    beta[t] = [v / b_sum for v in beta[t]]
            # Gamma and xi
            gamma = [[alpha[t][s] * beta[t][s] for s in range(self.n_states)] for t in range(T)]
            for t in range(T):
                row_sum = sum(gamma[t])
                if row_sum > 0:
                    gamma[t] = [v / row_sum for v in gamma[t]]
            # Update parameters
            self.pi = gamma[0][:]
            for i in range(self.n_states):
                for j in range(self.n_states):
                    num = sum(
                        alpha[t][i] * self.A[i][j] * self.B[j][obs[t + 1]] * beta[t + 1][j]
                        for t in range(T - 1)
                    )
                    den = sum(alpha[t][i] * beta[t][i] for t in range(T - 1))
                    self.A[i][j] = num / den if den > 1e-300 else 1.0 / self.n_states
                # Renormalise row
                row_sum = sum(self.A[i])
                self.A[i] = [v / row_sum for v in self.A[i]]
                for k in range(self.n_obs):
                    num = sum(gamma[t][i] for t in range(T) if obs[t] == k)
                    den = sum(gamma[t][i] for t in range(T))
                    self.B[i][k] = num / den if den > 1e-300 else 1.0 / self.n_obs
                row_sum = sum(self.B[i])
                self.B[i] = [v / row_sum for v in self.B[i]]
            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll
        return ll


# ---------------------------------------------------------------------------
# Random Walks
# ---------------------------------------------------------------------------

def simple_random_walk(steps: int, p: float = 0.5) -> List[int]:
    """1D simple random walk: +1 with prob p, -1 otherwise."""
    pos = 0
    path = [pos]
    for _ in range(steps):
        pos += 1 if random.random() < p else -1
        path.append(pos)
    return path


def random_walk_2d(steps: int) -> List[Tuple[int, int]]:
    """2D random walk on the integer lattice."""
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    x, y = 0, 0
    path = [(x, y)]
    for _ in range(steps):
        dx, dy = random.choice(directions)
        x += dx
        y += dy
        path.append((x, y))
    return path


def brownian_motion(
    T: float,
    n_steps: int,
    mu: float = 0.0,
    sigma: float = 1.0
) -> Tuple[List[float], List[float]]:
    """Simulate Brownian motion (Wiener process with drift).

    Returns (times, values).
    """
    dt = T / n_steps
    times = [i * dt for i in range(n_steps + 1)]
    values = [0.0]
    for _ in range(n_steps):
        dW = random.gauss(0, math.sqrt(dt))
        values.append(values[-1] + mu * dt + sigma * dW)
    return times, values


def geometric_brownian_motion(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int
) -> Tuple[List[float], List[float]]:
    """Simulate Geometric Brownian Motion: dS = mu*S*dt + sigma*S*dW.

    Returns (times, prices).
    """
    dt = T / n_steps
    times = [i * dt for i in range(n_steps + 1)]
    S = [S0]
    for _ in range(n_steps):
        dW = random.gauss(0, math.sqrt(dt))
        S.append(S[-1] * math.exp((mu - 0.5 * sigma ** 2) * dt + sigma * dW))
    return times, S


def first_passage_time(chain: MarkovChain, start: int, target: int, n_simulations: int = 10000) -> float:
    """Estimate the expected first passage time from start to target by simulation."""
    total_steps = 0
    for _ in range(n_simulations):
        state = start
        steps = 0
        for _ in range(100000):
            r = random.random()
            cumulative = 0.0
            for j, p in enumerate(chain.P[state]):
                cumulative += p
                if r <= cumulative:
                    state = j
                    break
            steps += 1
            if state == target:
                break
        total_steps += steps
    return total_steps / n_simulations
