import pandas as pd
from pathlib import Path

from recommender.classical import (
    loadMovieLens, build_user_item, computeAdjustedSimilarity, recommendItems
)
from recommender.quantum import recommended_quantum

def hybrid_recommend(df: pd.DataFrame, userId: int, top_classical: int = 5):
    userItem = build_user_item(df)
    sim = computeAdjustedSimilarity(userItem)

    classical = recommendItems(userItem, sim, userId, topN=top_classical, k=30)
    classical = classical.head(min(len(classical), top_classical))

    items = classical.index.tolist()

    sim_subset = sim.loc[items, items]

    quantum = recommended_quantum(classical, sim_subset)
    return classical, quantum


if __name__ == "__main__":
    dataFolder = Path("data/ml-100k")
    df = loadMovieLens(dataFolder)

    userId = 1
    classical, quantum = hybrid_recommend(df, userId)

    print("\nClassical:")
    print(classical)

    print("\nQuantum:")
    print(quantum)
