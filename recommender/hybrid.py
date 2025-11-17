import pandas as pd
import numpy as np

from recommender.classical import (
    build_user_item,
    computeAdjustedSimilarity,
    recommendItems,
)
from recommender.quantum import recommended_quantum


def hybrid_recommend(
    df: pd.DataFrame,
    external_ratings: dict,
    topK: int = 5,
):
    """
    external_ratings: dict like:
        {
            "Interstellar": 5,
            "Inception": 4,
            "Pulp Fiction": 2
        }
    """

    if not external_ratings:
        raise ValueError("No external ratings given.")

    ext_series = pd.Series(external_ratings)

    userItem = build_user_item(df)
    sim = computeAdjustedSimilarity(userItem)

    # We only need similarity values for the items user rated
    items = ext_series.index
    sim_subset = sim.loc[items, items]

    # Classical Top-N using similarity
    classical = ext_series.sort_values(ascending=False).head(topK)


    # Quantum optimization over the classical top items
    quantum = recommended_quantum(classical, sim_subset)

    return classical, quantum
