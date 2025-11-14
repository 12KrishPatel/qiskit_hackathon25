import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from recommender.qubo import build_pen_matrix, build_qubo_weights, give_summary

def test_qubo_shapes():
    items = ["A", "B", "C"]

    sim = pd.DataFrame(
        [[1.0, 0.5, 0.2],
         [0.5, 1.0, 0.3],
         [0.2, 0.3, 1.0]],
         index=items, columns=items
    )

    pen = build_pen_matrix(sim)
    ratings = pd.Series([4.5, 4.2, 3.9], index=items)
    linear, Q = build_qubo_weights(ratings, pen, lam=0.3)

    assert linear.shape == (3,)
    assert Q.shape == (3, 3)
    assert np.allclose(np.diag(Q), 0.0)
    assert np.allclose(np.triu(Q), Q)
