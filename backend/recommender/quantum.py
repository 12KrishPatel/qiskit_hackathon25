import numpy as np
import pandas as pd
from recommender.qubo import build_pen_matrix, build_qubo_weights


def recommended_quantum(classicalRecs: pd.Series,
                        similarity: pd.DataFrame,
                        keep: int = 3,
                        lam: float = 0.2):

    if classicalRecs.empty:
        raise ValueError("No classical recommendations provided")

    items = classicalRecs.index.tolist()

    penalty = build_pen_matrix(similarity.loc[items, items])

    linear, Q = build_qubo_weights(classicalRecs, penalty, lam)

    scores = linear - lam * penalty.mean(axis=1)

    top_idx = np.argsort(scores)[::-1][:keep]

    return {
        items[i]: float(scores[i])
        for i in top_idx
    }
