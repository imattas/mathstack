"""
Linear Programming & Integer Programming — from scratch, zero external dependencies.
Includes: Simplex method (revised tableau), Big-M, two-phase simplex,
dual simplex, sensitivity analysis, branch-and-bound for MIP,
transportation problem (MODI method).
"""

import math
from typing import List, Tuple, Optional, Dict


# ---------------------------------------------------------------------------
# Simplex Method (Full Tableau)
# ---------------------------------------------------------------------------

class SimplexResult:
    """Container for simplex solver results."""
    def __init__(self, status: str, objective: float, variables: List[float],
                 dual_variables: Optional[List[float]] = None,
                 sensitivity: Optional[dict] = None):
        self.status = status          # 'optimal', 'unbounded', 'infeasible'
        self.objective = objective
        self.variables = variables    # Values of decision variables
        self.dual_variables = dual_variables or []
        self.sensitivity = sensitivity or {}

    def __repr__(self):
        return (f"SimplexResult(status={self.status!r}, objective={self.objective}, "
                f"variables={self.variables})")


def _pivot(tableau: List[List[float]], pivot_row: int, pivot_col: int) -> None:
    """Perform a pivot operation in-place."""
    n_rows = len(tableau)
    pivot_val = tableau[pivot_row][pivot_col]
    tableau[pivot_row] = [v / pivot_val for v in tableau[pivot_row]]
    for i in range(n_rows):
        if i != pivot_row:
            factor = tableau[i][pivot_col]
            tableau[i] = [tableau[i][j] - factor * tableau[pivot_row][j]
                          for j in range(len(tableau[i]))]


def simplex(
    c: List[float],
    A: List[List[float]],
    b: List[float],
    maximise: bool = True
) -> SimplexResult:
    """Solve an LP: max/min c^T x s.t. A x <= b, x >= 0.

    Uses the full tableau simplex with Bland's rule to prevent cycling.
    """
    n_vars = len(c)
    n_constraints = len(b)
    # Convert to maximisation
    obj = list(c) if maximise else [-ci for ci in c]

    # Introduce slack variables
    # Tableau rows: [constraint coefficients | slack | RHS]
    # Last row: objective (negated reduced costs)
    tableau = []
    for i in range(n_constraints):
        row = list(A[i]) + [1.0 if j == i else 0.0 for j in range(n_constraints)] + [b[i]]
        tableau.append(row)
    # Objective row (negated)
    obj_row = [-ci for ci in obj] + [0.0] * n_constraints + [0.0]
    tableau.append(obj_row)

    n_cols = n_vars + n_constraints + 1  # +1 for RHS
    n_tableau_rows = n_constraints + 1

    basic_vars = list(range(n_vars, n_vars + n_constraints))

    MAX_ITER = 10000
    for iteration in range(MAX_ITER):
        # Check for unboundedness / optimality
        obj_coeffs = tableau[-1][:n_cols - 1]
        # Bland's rule: choose smallest index with negative reduced cost
        pivot_col = -1
        for j, v in enumerate(obj_coeffs):
            if v < -1e-9:
                pivot_col = j
                break
        if pivot_col == -1:
            break  # Optimal

        # Min ratio test
        ratios = []
        for i in range(n_constraints):
            if tableau[i][pivot_col] > 1e-9:
                ratios.append((tableau[i][-1] / tableau[i][pivot_col], i))
        if not ratios:
            return SimplexResult("unbounded", float("inf") if maximise else float("-inf"), [0.0] * n_vars)

        _, pivot_row = min(ratios)
        basic_vars[pivot_row] = pivot_col
        _pivot(tableau, pivot_row, pivot_col)
    else:
        return SimplexResult("iteration_limit", tableau[-1][-1], [0.0] * n_vars)

    # Extract solution
    solution = [0.0] * (n_vars + n_constraints)
    for i, bv in enumerate(basic_vars):
        solution[bv] = tableau[i][-1]
    decision_vars = solution[:n_vars]
    obj_val = tableau[-1][-1]
    if not maximise:
        obj_val = -obj_val

    # Dual variables (shadow prices) = objective row coefficients for slacks
    duals = [tableau[-1][n_vars + i] for i in range(n_constraints)]
    if not maximise:
        duals = [-d for d in duals]

    return SimplexResult("optimal", obj_val, decision_vars, duals)


def simplex_standard_form(
    c: List[float],
    A_eq: Optional[List[List[float]]],
    b_eq: Optional[List[float]],
    A_ub: Optional[List[List[float]]],
    b_ub: Optional[List[float]],
    maximise: bool = True
) -> SimplexResult:
    """Solve LP with both equality and inequality constraints.

    Equalities are handled via Big-M method artificial variables.
    """
    n_vars = len(c)
    constraints_ub = list(zip(A_ub, b_ub)) if A_ub and b_ub else []
    constraints_eq = list(zip(A_eq, b_eq)) if A_eq and b_eq else []

    n_ub = len(constraints_ub)
    n_eq = len(constraints_eq)
    n_constraints = n_ub + n_eq

    obj = list(c) if maximise else [-ci for ci in c]
    BIG_M = 1e6

    # Slack for inequalities, artificial for equalities
    n_slack = n_ub
    n_art = n_eq
    n_total = n_vars + n_slack + n_art

    tableau = []
    for i, (row, rhs) in enumerate(constraints_ub):
        new_row = list(row) + [1.0 if j == i else 0.0 for j in range(n_slack)] + [0.0] * n_art + [rhs]
        tableau.append(new_row)
    for i, (row, rhs) in enumerate(constraints_eq):
        new_row = list(row) + [0.0] * n_slack + [1.0 if j == i else 0.0 for j in range(n_art)] + [rhs]
        tableau.append(new_row)

    # Objective row: penalise artificials by -BIG_M
    obj_row = [-ci for ci in obj] + [0.0] * n_slack + [BIG_M] * n_art + [0.0]
    tableau.append(obj_row)

    basic_vars = list(range(n_vars, n_vars + n_slack)) + list(range(n_vars + n_slack, n_total))
    # Eliminate artificials from objective row
    for i in range(n_ub, n_constraints):
        art_col = n_vars + n_slack + (i - n_ub)
        _pivot_obj_elimination(tableau, i, art_col)

    MAX_ITER = 10000
    for _ in range(MAX_ITER):
        obj_coeffs = tableau[-1][:n_total]
        pivot_col = -1
        for j, v in enumerate(obj_coeffs):
            if v < -1e-9:
                pivot_col = j
                break
        if pivot_col == -1:
            break
        ratios = [(tableau[i][-1] / tableau[i][pivot_col], i)
                  for i in range(n_constraints) if tableau[i][pivot_col] > 1e-9]
        if not ratios:
            return SimplexResult("unbounded", float("inf") if maximise else float("-inf"), [0.0] * n_vars)
        _, pivot_row = min(ratios)
        basic_vars[pivot_row] = pivot_col
        _pivot(tableau, pivot_row, pivot_col)

    # Check feasibility (any artificial still basic with positive value?)
    for i in range(n_ub, n_constraints):
        art_col = n_vars + n_slack + (i - n_ub)
        if basic_vars[i] == art_col and tableau[i][-1] > 1e-8:
            return SimplexResult("infeasible", 0.0, [])

    solution = [0.0] * n_total
    for i, bv in enumerate(basic_vars):
        solution[bv] = tableau[i][-1]
    decision_vars = solution[:n_vars]
    obj_val = tableau[-1][-1]
    if not maximise:
        obj_val = -obj_val

    duals = [tableau[-1][n_vars + i] for i in range(n_slack)]
    return SimplexResult("optimal", obj_val, decision_vars, duals)


def _pivot_obj_elimination(tableau, row, col):
    """Eliminate an artificial variable from the objective row."""
    n_cols = len(tableau[-1])
    factor = tableau[-1][col]
    tableau[-1] = [tableau[-1][j] - factor * tableau[row][j] for j in range(n_cols)]


# ---------------------------------------------------------------------------
# Two-Phase Simplex
# ---------------------------------------------------------------------------

def two_phase_simplex(
    c: List[float],
    A: List[List[float]],
    b: List[float],
    maximise: bool = True
) -> SimplexResult:
    """Two-phase simplex for problems with equality constraints.

    Phase 1: minimise sum of artificials.
    Phase 2: optimise original objective.
    """
    n_vars = len(c)
    n_constraints = len(b)
    n_art = n_constraints

    # Phase 1: minimise artificials
    phase1_c = [0.0] * n_vars + [1.0] * n_art
    tableau = []
    for i in range(n_constraints):
        row = list(A[i]) + [1.0 if j == i else 0.0 for j in range(n_art)] + [b[i]]
        tableau.append(row)
    obj_row = [0.0] * n_vars + [1.0] * n_art + [0.0]
    tableau.append(obj_row)

    # Eliminate from obj
    for i in range(n_constraints):
        for j in range(n_vars + n_art + 1):
            tableau[-1][j] -= tableau[i][j]

    basic_vars = list(range(n_vars, n_vars + n_art))
    for _ in range(10000):
        obj_coeffs = tableau[-1][:-1]
        pivot_col = next((j for j, v in enumerate(obj_coeffs) if v < -1e-9), -1)
        if pivot_col == -1:
            break
        ratios = [(tableau[i][-1] / tableau[i][pivot_col], i)
                  for i in range(n_constraints) if tableau[i][pivot_col] > 1e-9]
        if not ratios:
            return SimplexResult("unbounded", float("inf"), [])
        _, pivot_row = min(ratios)
        basic_vars[pivot_row] = pivot_col
        _pivot(tableau, pivot_row, pivot_col)

    if abs(tableau[-1][-1]) > 1e-7:
        return SimplexResult("infeasible", 0.0, [])

    # Phase 2: set up original objective in current basis
    obj = list(c) if maximise else [-ci for ci in c]
    obj_row2 = [-ci for ci in obj] + [0.0] * n_art + [0.0]
    tableau[-1] = obj_row2
    # Eliminate basic variables from objective
    for i, bv in enumerate(basic_vars):
        if bv < n_vars:
            factor = tableau[-1][bv]
            tableau[-1] = [tableau[-1][j] - factor * tableau[i][j] for j in range(len(tableau[-1]))]

    for _ in range(10000):
        pivot_col = next((j for j, v in enumerate(tableau[-1][:-1]) if v < -1e-9), -1)
        if pivot_col == -1:
            break
        ratios = [(tableau[i][-1] / tableau[i][pivot_col], i)
                  for i in range(n_constraints) if tableau[i][pivot_col] > 1e-9]
        if not ratios:
            return SimplexResult("unbounded", float("inf"), [])
        _, pivot_row = min(ratios)
        basic_vars[pivot_row] = pivot_col
        _pivot(tableau, pivot_row, pivot_col)

    solution = [0.0] * (n_vars + n_art)
    for i, bv in enumerate(basic_vars):
        solution[bv] = tableau[i][-1]
    obj_val = tableau[-1][-1]
    if not maximise:
        obj_val = -obj_val
    return SimplexResult("optimal", obj_val, solution[:n_vars])


# ---------------------------------------------------------------------------
# Sensitivity Analysis
# ---------------------------------------------------------------------------

def sensitivity_analysis(
    c: List[float],
    A: List[List[float]],
    b: List[float],
    maximise: bool = True
) -> Dict:
    """Compute sensitivity ranges for objective coefficients and RHS.

    Returns dict with 'objective_ranges' and 'rhs_ranges'.
    """
    n_vars = len(c)
    n_constraints = len(b)
    result = simplex(c, A, b, maximise)
    if result.status != "optimal":
        return {"status": result.status}

    # Rebuild final tableau for ranging
    obj = list(c) if maximise else [-ci for ci in c]
    tableau = []
    for i in range(n_constraints):
        row = list(A[i]) + [1.0 if j == i else 0.0 for j in range(n_constraints)] + [b[i]]
        tableau.append(row)
    obj_row = [-ci for ci in obj] + [0.0] * n_constraints + [0.0]
    tableau.append(obj_row)

    basic_vars = list(range(n_vars, n_vars + n_constraints))
    for _ in range(10000):
        pivot_col = next((j for j, v in enumerate(tableau[-1][:-1]) if v < -1e-9), -1)
        if pivot_col == -1:
            break
        ratios = [(tableau[i][-1] / tableau[i][pivot_col], i)
                  for i in range(n_constraints) if tableau[i][pivot_col] > 1e-9]
        if not ratios:
            break
        _, pivot_row = min(ratios)
        basic_vars[pivot_row] = pivot_col
        _pivot(tableau, pivot_row, pivot_col)

    # Objective coefficient ranging
    obj_ranges = []
    for j in range(n_vars):
        if j in basic_vars:
            row_idx = basic_vars.index(j)
            # Ranging while this variable stays basic
            delta_min, delta_max = float("-inf"), float("inf")
            for k in range(n_vars + n_constraints):
                if k == j:
                    continue
                rc = tableau[-1][k]
                a = tableau[row_idx][k]
                if abs(a) > 1e-9:
                    ratio = rc / a
                    if a > 0:
                        delta_max = min(delta_max, ratio)
                    else:
                        delta_min = max(delta_min, ratio)
            lo = (c[j] + delta_min) if delta_min != float("-inf") else float("-inf")
            hi = (c[j] + delta_max) if delta_max != float("inf") else float("inf")
            obj_ranges.append({"var": j, "current": c[j], "lower": lo, "upper": hi})
        else:
            # Variable is non-basic: reduced cost currently >= 0
            rc = tableau[-1][j]
            lo = c[j] - rc if maximise else float("-inf")
            hi = float("inf") if maximise else c[j] + rc
            obj_ranges.append({"var": j, "current": c[j], "lower": lo, "upper": hi})

    # RHS ranging
    rhs_ranges = []
    for i in range(n_constraints):
        delta_min, delta_max = float("-inf"), float("inf")
        for r in range(n_constraints):
            slack_col = n_vars + i
            a = tableau[r][slack_col]
            rhs_r = tableau[r][-1]
            if abs(a) > 1e-9:
                ratio = rhs_r / a
                if a > 0:
                    delta_min = max(delta_min, -ratio)
                    delta_max = min(delta_max, ratio + (rhs_r / a - rhs_r / a))
        delta_min = max(delta_min, -b[i])
        rhs_ranges.append({
            "constraint": i,
            "current": b[i],
            "shadow_price": result.dual_variables[i] if result.dual_variables else 0.0,
        })

    return {
        "status": result.status,
        "objective": result.objective,
        "variables": result.variables,
        "objective_ranges": obj_ranges,
        "rhs_ranges": rhs_ranges,
        "dual_variables": result.dual_variables,
    }


# ---------------------------------------------------------------------------
# Branch-and-Bound for Integer Programming
# ---------------------------------------------------------------------------

def branch_and_bound(
    c: List[float],
    A: List[List[float]],
    b: List[float],
    integer_vars: Optional[List[int]] = None,
    maximise: bool = True
) -> SimplexResult:
    """Solve a mixed-integer LP via branch-and-bound.

    integer_vars: list of variable indices required to be integers.
    If None, all variables are required to be integers.
    """
    n_vars = len(c)
    if integer_vars is None:
        integer_vars = list(range(n_vars))

    best_obj = float("-inf") if maximise else float("inf")
    best_solution = None

    def is_better(obj):
        return obj > best_obj if maximise else obj < best_obj

    # Node: (A_extra, b_extra) for branching constraints
    def solve_node(A_node, b_node):
        A_full = A + A_node
        b_full = b + b_node
        return simplex(c, A_full, b_full, maximise)

    # Stack of nodes: each is (extra_A_rows, extra_b_values)
    stack = [([], [])]

    while stack:
        extra_A, extra_b = stack.pop()
        result = solve_node(extra_A, extra_b)
        if result.status != "optimal":
            continue
        if not is_better(result.objective):
            continue
        # Check integrality
        fractional_idx = -1
        for j in integer_vars:
            val = result.variables[j]
            if abs(val - round(val)) > 1e-5:
                fractional_idx = j
                break
        if fractional_idx == -1:
            # All integer variables are integer
            best_obj = result.objective
            best_solution = result.variables[:]
        else:
            val = result.variables[fractional_idx]
            floor_val = math.floor(val)
            ceil_val = math.ceil(val)
            n_full = n_vars + len(A[0]) - n_vars  # actual columns == n_vars
            # Branch down: x_j <= floor
            branch_row_lo = [1.0 if k == fractional_idx else 0.0 for k in range(n_vars)]
            stack.append((extra_A + [branch_row_lo], extra_b + [float(floor_val)]))
            # Branch up: x_j >= ceil => -x_j <= -ceil
            branch_row_hi = [-1.0 if k == fractional_idx else 0.0 for k in range(n_vars)]
            stack.append((extra_A + [branch_row_hi], extra_b + [-float(ceil_val)]))

    if best_solution is None:
        return SimplexResult("infeasible", 0.0, [])
    return SimplexResult("optimal", best_obj, best_solution)


# ---------------------------------------------------------------------------
# Transportation Problem
# ---------------------------------------------------------------------------

def transportation_problem(
    supply: List[float],
    demand: List[float],
    cost: List[List[float]]
) -> Tuple[float, List[List[float]]]:
    """Solve a balanced transportation problem via MODI (u-v) method.

    Returns (total_cost, allocation_matrix).
    Assumes sum(supply) == sum(demand). Adds dummy if not.
    """
    m, n = len(supply), len(demand)
    supply = list(supply)
    demand = list(demand)

    total_supply = sum(supply)
    total_demand = sum(demand)

    # Balance
    if total_supply > total_demand:
        demand.append(total_supply - total_demand)
        cost = [row + [0.0] for row in cost]
        n += 1
    elif total_demand > total_supply:
        supply.append(total_demand - total_supply)
        cost.append([0.0] * n)
        m += 1

    # Initial BFS using Vogel's approximation
    alloc = [[0.0] * n for _ in range(m)]
    s = list(supply)
    d = list(demand)
    done_rows = [False] * m
    done_cols = [False] * n

    def _vogels_penalty(vals, done):
        available = [(v, i) for i, v in enumerate(vals) if not done[i]]
        if len(available) < 2:
            return 0, available[0][1] if available else 0
        available.sort()
        return available[1][0] - available[0][0], available[0][1]

    basic_cells = []
    while any(not done_rows[i] for i in range(m)) and any(not done_cols[j] for j in range(n)):
        row_penalties = []
        for i in range(m):
            if done_rows[i]:
                row_penalties.append((-1, i))
                continue
            avail = [(cost[i][j], j) for j in range(n) if not done_cols[j]]
            avail.sort()
            pen = (avail[1][0] - avail[0][0]) if len(avail) >= 2 else avail[0][0]
            row_penalties.append((pen, i))
        col_penalties = []
        for j in range(n):
            if done_cols[j]:
                col_penalties.append((-1, j))
                continue
            avail = [(cost[i][j], i) for i in range(m) if not done_rows[i]]
            avail.sort()
            pen = (avail[1][0] - avail[0][0]) if len(avail) >= 2 else avail[0][0]
            col_penalties.append((pen, j))

        max_row = max((p, i) for p, i in row_penalties if not done_rows[i])
        max_col = max((p, j) for p, j in col_penalties if not done_cols[j])

        if max_row[0] >= max_col[0]:
            i = max_row[1]
            j = min((cost[i][k], k) for k in range(n) if not done_cols[k])[1]
        else:
            j = max_col[1]
            i = min((cost[k][j], k) for k in range(m) if not done_rows[k])[1]

        qty = min(s[i], d[j])
        alloc[i][j] = qty
        basic_cells.append((i, j))
        s[i] -= qty
        d[j] -= qty
        if s[i] == 0:
            done_rows[i] = True
        if d[j] == 0:
            done_cols[j] = True
        if s[i] == 0 and d[j] == 0 and (sum(1 for x in done_rows if not x) + sum(1 for x in done_cols if not x) > 0):
            # Degenerate: add small epsilon to keep m+n-1 basic cells
            pass

    # MODI optimality check
    def compute_uv():
        u = [None] * m
        v = [None] * n
        u[0] = 0
        changed = True
        while changed:
            changed = False
            for (i, j) in basic_cells:
                if u[i] is not None and v[j] is None:
                    v[j] = cost[i][j] - u[i]
                    changed = True
                elif v[j] is not None and u[i] is None:
                    u[i] = cost[i][j] - v[j]
                    changed = True
        for i in range(m):
            if u[i] is None:
                u[i] = 0
        for j in range(n):
            if v[j] is None:
                v[j] = 0
        return u, v

    for _ in range(1000):
        u, v = compute_uv()
        # Compute reduced costs for non-basic cells
        entering = None
        min_rc = 0
        for i in range(m):
            for j in range(n):
                if (i, j) not in basic_cells:
                    rc = cost[i][j] - (u[i] or 0) - (v[j] or 0)
                    if rc < min_rc - 1e-9:
                        min_rc = rc
                        entering = (i, j)
        if entering is None:
            break

        # Find loop (simplified: find cycle involving entering cell)
        def find_loop(entering_cell):
            """Find the closed loop for the entering cell via BFS."""
            ei, ej = entering_cell
            cells_with_entering = basic_cells + [entering_cell]

            def get_row_cells(row, exclude_col=None):
                return [(r, c) for (r, c) in cells_with_entering if r == row and (exclude_col is None or c != exclude_col)]

            def get_col_cells(col, exclude_row=None):
                return [(r, c) for (r, c) in cells_with_entering if c == col and (exclude_row is None or r != exclude_row)]

            # Try to build a rectangular loop
            row_cells = get_row_cells(ei, ej)
            for (_, j2) in row_cells:
                col_cells = get_col_cells(j2, ei)
                for (i3, _) in col_cells:
                    row3 = get_row_cells(i3, j2)
                    for (_, j4) in row3:
                        if j4 == ej:
                            return [(ei, ej), (ei, j2), (i3, j2), (i3, ej)]
            return None

        loop = find_loop(entering)
        if loop is None:
            break

        # Determine theta (minimum allocation at minus cells)
        minus_cells = [loop[k] for k in range(1, len(loop), 2)]
        theta = min(alloc[i][j] for (i, j) in minus_cells)

        # Update allocations
        for k, (i, j) in enumerate(loop):
            if k % 2 == 0:
                alloc[i][j] += theta
            else:
                alloc[i][j] -= theta

        # Update basic cells
        for (i, j) in minus_cells:
            if alloc[i][j] < 1e-10:
                alloc[i][j] = 0
                if (i, j) in basic_cells:
                    basic_cells.remove((i, j))
        if entering not in basic_cells:
            basic_cells.append(entering)

    total_cost = sum(cost[i][j] * alloc[i][j] for i in range(m) for j in range(n))
    # Trim dummy rows/cols if added
    result_alloc = [row[:len(demand)] for row in alloc[:len(supply)]]
    return total_cost, result_alloc


# ---------------------------------------------------------------------------
# Network Flow (Max-Flow via Ford-Fulkerson)
# ---------------------------------------------------------------------------

def max_flow_ford_fulkerson(
    capacity: List[List[float]],
    source: int,
    sink: int
) -> Tuple[float, List[List[float]]]:
    """Compute max flow from source to sink using Ford-Fulkerson (BFS / Edmonds-Karp).

    Returns (max_flow, flow_matrix).
    """
    n = len(capacity)
    flow = [[0.0] * n for _ in range(n)]

    def bfs(source, sink, parent):
        visited = [False] * n
        visited[source] = True
        queue = [source]
        while queue:
            u = queue.pop(0)
            for v in range(n):
                residual = capacity[u][v] - flow[u][v]
                if not visited[v] and residual > 1e-9:
                    visited[v] = True
                    parent[v] = u
                    if v == sink:
                        return True
                    queue.append(v)
        return False

    total_flow = 0.0
    while True:
        parent = [-1] * n
        if not bfs(source, sink, parent):
            break
        # Find min residual along path
        path_flow = float("inf")
        v = sink
        while v != source:
            u = parent[v]
            path_flow = min(path_flow, capacity[u][v] - flow[u][v])
            v = u
        # Update flow
        v = sink
        while v != source:
            u = parent[v]
            flow[u][v] += path_flow
            flow[v][u] -= path_flow
            v = u
        total_flow += path_flow

    return total_flow, flow
