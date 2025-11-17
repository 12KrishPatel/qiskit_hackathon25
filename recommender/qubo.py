import numpy as np
import pandas as pd

def build_pen_matrix(similarities: pd.DataFrame) -> np.ndarray:
    S = similarities.to_numpy(copy = True)
    S = np.nan_to_num(S, nan = 0.0)

    # Normalize to [0, 1]
    minVal, maxVal = S.min(), S.max()
    if maxVal > minVal:
        S = (S - minVal) / (maxVal - minVal)
    np.fill_diagonal(S, 0.0)
    return S


def build_qubo_weights(
        ratings: pd.Series,
        penalty: np.ndarray,
        lam: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    linear = ratings.values.astype(float)
    n = len(linear)

    Q = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1, n):
            Q[i, j] = -lam * penalty[i, j]
    return linear, Q


def give_summary(linear: np.ndarray, Q: np.ndarray, items: list[str]) -> pd.DataFrame:
    n = len(items)
    data = []

    for i in range(n):
        data.append({"item": items[i], "linear weight":linear[i]})
    return pd.DataFrame(data)