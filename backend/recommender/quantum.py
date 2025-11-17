from __future__ import annotations

import numpy as np
import pandas as pd

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import SPSA


def build_qubo(ratings: pd.Series,
               penalty: np.ndarray,
               lam: float = 0.2) -> QuadraticProgram:
    """
    Build a QUBO for 'pick a good subset of movies'.

    ratings  ->  classical predicted scores for each movie
    penalty  ->  similarity matrix between those movies
    lam      ->  trade-off between 'high rating' and 'diversity'
    """
    n = len(ratings)
    qp = QuadraticProgram("Quantum_Recommender")

    # One binary variable per movie
    for i in range(n):
        qp.binary_var(name=f"x{i}")

    # Linear term: reward high-rated movies
    linear = {f"x{i}": float(ratings.iloc[i]) for i in range(n)}

    # Quadratic term: penalize picking very similar movies together
    quadratic = {}
    for i in range(n):
        for j in range(i + 1, n):
            quadratic[(f"x{i}", f"x{j}")] = -lam * float(penalty[i, j])

    qp.maximize(linear=linear, quadratic=quadratic)
    return qp


def solve_qubo(qp: QuadraticProgram):
    """
    Solve the QUBO using QAOA on a statevector simulator.
    This is the part that actually uses Qiskit.
    """
    sampler = StatevectorSampler()
    optimizer = SPSA(maxiter=30)          # keep it small so it’s fast
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=1)

    algo = MinimumEigenOptimizer(qaoa)
    result = algo.solve(qp)

    # Convert result.x (array) into a dict: {"x0": 0, "x1": 1, ...}
    return {
        var.name: int(val)
        for var, val in zip(qp.variables, result.x)
    }


def recommended_quantum(
    classicalRecs: pd.Series,
    similarity: pd.DataFrame,
    keep: int = 3,
    lam: float = 0.2,
    max_items_qubo: int = 6,
) -> dict[str, float]:
    """
    Take classical recommendations and run a small QAOA-based
    optimization on them to pick a diverse subset.

    classicalRecs : Series(index=movie_title, values=score)
    similarity    : similarity matrix over movie titles
    keep          : max number of movies to return
    max_items_qubo: how many movies to feed into QAOA at once
    """
    if classicalRecs.empty:
        raise ValueError("No classical recommendations provided")

    # Use the best few classical items for the quantum step
    classicalRecs = classicalRecs.sort_values(ascending=False)
    small = classicalRecs.head(max_items_qubo)
    items = small.index.tolist()

    # Build penalty matrix for just those items
    penalty = similarity.loc[items, items].to_numpy()

    # Build and solve the QUBO with QAOA (real Qiskit part)
    qp = build_qubo(small, penalty, lam)
    solution = solve_qubo(qp)

    # Decode the bitstring back into selected movies
    selected_flags = [solution[f"x{i}"] for i in range(len(items))]
    data = pd.DataFrame(
        {
            "item": items,
            "rating": small.values,
            "selected": selected_flags,
        }
    )

    # Keep only those chosen by QAOA, sort by rating, take top `keep`
    chosen = (
        data[data["selected"] == 1]
        .sort_values("rating", ascending=False)
        .head(keep)
    )

    # Return as {movie: score} so the frontend can render it easily
    return {row["item"]: float(row["rating"]) for _, row in chosen.iterrows()}


if __name__ == "__main__":
    # Tiny sanity check if you ever want to run this file directly
    top_classical = pd.Series(
        [4.5, 4.3, 3.8, 3.6, 3.2],
        index=["Movie A", "Movie B", "Movie C", "Movie D", "Movie E"],
    )

    sim = pd.DataFrame(
        [
            [1.0, 0.9, 0.2, 0.3, 0.1],
            [0.9, 1.0, 0.4, 0.2, 0.3],
            [0.2, 0.4, 1.0, 0.6, 0.5],
            [0.3, 0.2, 0.6, 1.0, 0.7],
            [0.1, 0.3, 0.5, 0.7, 1.0],
        ],
        index=top_classical.index,
        columns=top_classical.index,
    )

    out = recommended_quantum(top_classical, sim, keep=3, lam=0.3)
    print(out)
