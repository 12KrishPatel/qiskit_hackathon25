import pandas as pd
from recommender.classical import build_user_item, computeAdjustedSimilarity

def test_user_matrix_shape():
    df = pd.DataFrame({
        "user": [1, 1, 2, 2],
        "item": ["A", "B", "A", "C"],
        "rating": [4, 5, 3, 2]
    })
    userItem = build_user_item(df)
    assert userItem.shape == (2, 3)

def test_simialarity():
    df = pd.DataFrame({
        "user": [1, 1, 2, 2],
        "item": ["A", "B", "A", "C"],
        "rating": [4, 5, 3, 2]
    })
    userItem = build_user_item(df)
    sim = computeAdjustedSimilarity(userItem)
    assert all(sim[col][col] == 1.0 for col in sim.columns)