from __future__ import annotations
import numpy as np
import pandas as pd
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import SPSA


def build_qubo(ratings: pd.Series, penalty: np.ndarray, lam: float = 0.2) -> QuadraticProgram:
    n = len(ratings)
    qp = QuadraticProgram("Quantum_Recommender")

    for i in range(n):
        qp.binary_var(name=f"x{i}")

    linear = {f"x{i}": float(ratings.iloc[i]) for i in range(n)}

    quadratic = {}
    for i in range(n):
        for j in range(i + 1, n):
            quadratic[(f"x{i}", f"x{j}")] = -lam * float(penalty[i, j])

    qp.maximize(linear=linear, quadratic=quadratic)
    return qp


def solve_qubo(qp: QuadraticProgram):
    sampler = StatevectorSampler()
    optimizer = SPSA(maxiter=20)
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=1)
    algo = MinimumEigenOptimizer(qaoa)
    result = algo.solve(qp)
    return {var.name: int(val) for var, val in zip(qp.variables, result.x)}


def recommended_quantum(classicalRecs: pd.Series, similarity: pd.DataFrame, lam: float = 0.2) -> pd.DataFrame:
    if classicalRecs.empty:
        raise ValueError("No classical recommendations provided")

    items = classicalRecs.index.tolist()
    penalty = similarity.loc[items, items].to_numpy()

    qp = build_qubo(classicalRecs, penalty, lam)
    solution = solve_qubo(qp)

    selected = [solution[f"x{i}"] for i in range(len(items))]
    data = pd.DataFrame({"item": items, "rating": classicalRecs.values, "selected": selected})
    return data[data["selected"] == 1].sort_values("rating", ascending=False)


if __name__ == "__main__":
    top_classical = pd.Series([4.5, 4.3, 3.8, 3.6, 3.2],
                              index=["Movie A", "Movie B", "Movie C", "Movie D", "Movie E"])

    sim = pd.DataFrame(
        [
            [1.0, 0.9, 0.2, 0.3, 0.1],
            [0.9, 1.0, 0.4, 0.2, 0.3],
            [0.2, 0.4, 1.0, 0.6, 0.5],
            [0.3, 0.2, 0.6, 1.0, 0.7],
            [0.1, 0.3, 0.5, 0.7, 1.0],
        ],
        index=top_classical.index,
        columns=top_classical.index
    )

    out = recommended_quantum(top_classical, sim, lam=0.3)
    print(out)
