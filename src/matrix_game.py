"""Two-player zero-sum matrix game solver via linear programming.

Solves max_pi min_omega pi^T Q omega over simplices.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog


def solve_matrix_game(Q: np.ndarray, tol: float = 1e-9):
    """Return (pi, omega, value) for the controller-maximizer / nature-minimizer game.

    Q has shape (J, L) where J = |A_cand|, L = |K_cand|.
    Controller picks rows pi in simplex_J; nature picks columns omega in simplex_L.
    Value = max_pi min_omega pi^T Q omega.

    Reformulate as LP:
        max v
        s.t. sum_j pi_j Q[j, l] >= v  for all l (for any nature column, expected >= v)
             sum_j pi_j = 1
             pi_j >= 0
    Variables: (pi_1, ..., pi_J, v). Maximize v.
    """
    J, L = Q.shape
    # linprog minimizes c^T x. We minimize -v.
    c = np.zeros(J + 1)
    c[-1] = -1.0  # minimize -v -> maximize v

    # Inequality: -sum_j pi_j Q[j,l] + v <= 0  for each column l
    A_ub = np.zeros((L, J + 1))
    A_ub[:, :J] = -Q.T
    A_ub[:, -1] = 1.0
    b_ub = np.zeros(L)

    # Equality: sum_j pi_j = 1
    A_eq = np.zeros((1, J + 1))
    A_eq[0, :J] = 1.0
    b_eq = np.array([1.0])

    # Bounds: pi_j in [0, 1], v free
    bounds = [(0.0, 1.0)] * J + [(None, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not res.success:
        # Fallback: uniform
        pi = np.ones(J) / J
        omega = np.ones(L) / L
        return pi, omega, float(pi @ Q @ omega)
    pi = res.x[:J]
    pi = np.clip(pi, 0.0, None)
    s = pi.sum()
    if s > 0:
        pi = pi / s
    else:
        pi = np.ones(J) / J
    v = float(-res.fun)

    # Solve dual for omega: min_omega max_pi pi^T Q omega
    # LP: min u; s.t. Q omega <= u * 1 ; sum omega = 1; omega >= 0
    c2 = np.zeros(L + 1)
    c2[-1] = 1.0
    A_ub2 = np.zeros((J, L + 1))
    A_ub2[:, :L] = Q
    A_ub2[:, -1] = -1.0
    b_ub2 = np.zeros(J)
    A_eq2 = np.zeros((1, L + 1))
    A_eq2[0, :L] = 1.0
    b_eq2 = np.array([1.0])
    bounds2 = [(0.0, 1.0)] * L + [(None, None)]
    res2 = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq2,
                   bounds=bounds2, method="highs")
    if res2.success:
        omega = res2.x[:L]
        omega = np.clip(omega, 0.0, None)
        if omega.sum() > 0:
            omega = omega / omega.sum()
        else:
            omega = np.ones(L) / L
    else:
        omega = np.ones(L) / L
    return pi, omega, v
