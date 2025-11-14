from __future__ import annotations
import numpy as np
import pandas as pd
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.translators import from_docplex_mp
from qiskit.primitives import Sampler
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler


# Build QUBO 
def build_qubo(ratings:pd.Series, penalty: np.ndarray, lam: float = 0.2) -> QuadraticProgram:
    n = len(ratings)
    qp = QuadraticProgram("Quantum_Recommender")

    for i in range(n):
        qp.binary_var(name=f"x{i}")
    
    linear = {f"x{i}": float(ratings.iloc[i]) for i in range(n)}

    quadratic = {}
    for i in range(n):
        for j in range(i + 1, n):
            quadratic[(f"x{i}", f"x{j}")] = -lam * float(penalty[i,j])

    qp.maximize(linear=linear, quadratic=quadratic)
    return qp

def solve_qubo(qp: QuadraticProgram, seed: int = 42) -> dict:
    
    sampler = StatevectorSampler()  # modern replacement
    optimizer = COBYLA(maxiter=100)  # you can also try SPSA for noisy runs
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=2)
    algo = MinimumEigenOptimizer(qaoa)
    result = algo.solve(qp)

    solutionDict = {var.name: int(val) for var, val in zip(qp.variables, result.x)}
    return solutionDict


def recommended_quantum(
        classicalRecs: pd.Series,
        similarityMatrix: pd.DataFrame,
        lam: float = 0.2
) -> pd.DataFrame:
    if classicalRecs.empty:
        raise ValueError("No classical recommendations provided")
    
    # Convert to arrays
    ratings = classicalRecs
    items = ratings.index.tolist()
    penalty = similarityMatrix.loc[items, items].to_numpy()

    qp = build_qubo(ratings, penalty, lam=lam)
    solution = solve_qubo(qp)

    chosen = [items[i] for i, x in enumerate(solution.values()) if x == 1]
    results = pd.DataFrame({
        "item": items,
        "predicted_rating": ratings.values,
        "selected": [solution[f"x{i}"] for i in range(len(items))]
    })

    print("\n🔹 Quantum Optimization Result:")
    print(results.to_string(index=False))

    return results[results["selected"] == 1].sort_values("predicted_rating", ascending=False)

if __name__ == "__main__":
    # Fake top classical recommendations
    top_classical = pd.Series(
        [4.5, 4.3, 3.8, 3.6, 3.2],
        index=["Movie A", "Movie B", "Movie C", "Movie D", "Movie E"]
    )

    # Mock similarity matrix (smaller = more diverse)
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

    results = recommended_quantum(top_classical, sim, lam=0.3)
    print("\n✅ Final Quantum Recommendations:")
    print(results)