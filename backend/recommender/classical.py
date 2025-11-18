import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from recommender.subset import topnmSubset


# load dataset
def loadMovieLens(dataFolder: Path) -> pd.DataFrame:
    dataPath = dataFolder / "u.data"
    itemPath = dataFolder / "u.item"
    
    ratings = pd.read_csv(
        dataPath, sep="\t",
        names=["user", "itemId", "rating", "timestamp"],
        usecols=[0, 1, 2]
    )
    movies = pd.read_csv(
        itemPath, sep="|", encoding="latin-1",
        usecols=[0, 1], names=["itemId", "item"]
    )
    df = ratings.merge(movies, on="itemId")[["user", "item", "rating"]]
    return df

def build_user_item(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(
        index="user",
        columns="item",
        values="rating",
        aggfunc="mean"
    )

def computeAdjustedSimilarity(userItem: pd.DataFrame, min_support: int = 1) -> pd.DataFrame:
    means = userItem.mean(axis=1)
    centered = userItem.sub(means, axis=0)

    filled = centered.fillna(0)

    sim = cosine_similarity(filled.T)
    sim_df = pd.DataFrame(sim, index=userItem.columns, columns=userItem.columns)

    exists = (~userItem.isna()).astype(int)
    co_occur = exists.T @ exists

    mask = co_occur < min_support
    sim_df = sim_df.where(~mask, 0)

    np.fill_diagonal(sim_df.values, 1.0)

    return sim_df

def recommendItems(
    userItem: pd.DataFrame,
    itemSim: pd.DataFrame,
    userId: int,
    topN: int = 5,
    k: int = 30,
    reg: float = 0.3,   
) -> pd.Series:

    if userId not in userItem.index:
        raise ValueError(f"User {userId} not found in matrix")

    # Center ratings around the user's mean
    userRatings = userItem.loc[userId]
    user_mean = userRatings.mean()

    centered_ratings = userRatings - user_mean
    rated_items = centered_ratings.dropna().index
    unrated_items = userItem.columns.difference(rated_items)

    predictions = {}
    global_mean = userItem.stack().mean()

    for item in unrated_items:
        sims = itemSim.loc[item, rated_items]
        sims = sims[sims > 0].sort_values(ascending=False).head(k)

        if len(sims) == 0:
            predictions[item] = global_mean
            continue

        numerator = (sims * centered_ratings[sims.index]).sum()
        denominator = sims.abs().sum() + reg 

        pred = user_mean + numerator / denominator

        predictions[item] = max(1, min(5, pred))

    return pd.Series(predictions).sort_values(ascending=False).head(topN)
