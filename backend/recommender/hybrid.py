import pandas as pd

from recommender.classical import (
    build_user_item,
    computeAdjustedSimilarity,
    recommendItems,
)
from recommender.quantum import recommended_quantum


def hybrid_recommend(
    df: pd.DataFrame,
    external_ratings: dict,
    top_classical: int = 20,
    top_quantum: int = 3,
):

    if not external_ratings:
        raise ValueError("No external ratings given.")

    userItem = build_user_item(df)
    itemSim = computeAdjustedSimilarity(userItem)

    new_user = userItem.index.max() + 1
    extended = userItem.copy()
    extended.loc[new_user] = float("nan")

    for movie, rating in external_ratings.items():
        if movie in extended.columns:
            extended.at[new_user, movie] = float(rating)

    classical_series = recommendItems(
        extended,
        itemSim,
        userId=new_user,
        topN=top_classical,  # get top 20
        k=30,
    )

    if classical_series.empty:
        return classical_series, {}

    items = classical_series.index
    sim_subset = itemSim.loc[items, items]

    quantum_dict = recommended_quantum(
        classical_series,
        sim_subset,
        keep=top_quantum,  # show up to 3 quantum picks
    )

    return classical_series, quantum_dict
