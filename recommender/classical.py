import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# Load MovieLens dataset
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

# Build user-item matrix
def build_user_item(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(
        index="user",
        columns="item",
        values="rating",
        aggfunc="mean"   
    )

# Compute adjusted cosine similarity between items
def computeAdjustedSimilarity(userItem: pd.DataFrame, min_support: int = 5) -> pd.DataFrame:
    # Center by user mean
    user_means = userItem.mean(axis=1)
    userItem_centered = userItem.sub(user_means, axis=0)
    
    # Compute cosine similarity
    centered_filled = userItem_centered.fillna(0)
    sim = cosine_similarity(centered_filled.T)
    sim_df = pd.DataFrame(sim, index=userItem.columns, columns=userItem.columns)
    
    # Apply minimum support filter
    rating_exists = (~userItem.isna()).astype(int)
    co_occurrence = rating_exists.T @ rating_exists
    mask = co_occurrence < min_support
    sim_df = sim_df.where(~mask, 0)
    np.fill_diagonal(sim_df.values, 1.0)
    
    return sim_df

# Recommend top N items for a user
def recommendItems(
    userItem: pd.DataFrame, 
    itemSim: pd.DataFrame, 
    userId: int,
    topN: int = 5,
    k: int = 30
) -> pd.Series:
    if userId not in userItem.index:
        raise KeyError(f"User {userId} not found in matrix")
    
    userRatings = userItem.loc[userId]
    rated_items = userRatings.dropna().index
    unrated_items = userItem.columns.difference(rated_items)
    
    predictions = {}
    
    for item in unrated_items:
        sims = itemSim.loc[item, rated_items]
        sims = sims[sims > 0].sort_values(ascending=False).head(k)
        
        if len(sims) == 0:
            continue
        
        # Weighted average
        ratings = userRatings[sims.index]
        numerator = (sims * ratings).sum()
        denominator = sims.abs().sum()
        
        if denominator > 0:
            predictions[item] = numerator / denominator
    
    pred_series = pd.Series(predictions)
    return pred_series.sort_values(ascending=False).head(topN)

# Main
if __name__ == "__main__":
    dataFolder = Path("data/ml-100k")
    df = loadMovieLens(dataFolder)
    print(f"✅ Loaded {len(df):,} ratings, {df['user'].nunique()} users, {df['item'].nunique()} movies")

    userItem = build_user_item(df)
    print(f"📊 User-Item matrix: {userItem.shape}")

    print("🧮 Computing item similarities...")
    itemSim = computeAdjustedSimilarity(userItem, min_support=5)
    print(f"✓ Similarity matrix: {itemSim.shape}")

    sampleUser = 1
    rated_count = userItem.loc[sampleUser].count()
    print(f"\n🎯 User {sampleUser} has rated {rated_count} movies")

    recs = recommendItems(userItem, itemSim, sampleUser, topN=5, k=30)

    # Display results
    if recs.empty:
        print(f"⚠️  No recommendations found for user {sampleUser}")
    else:
        print(f"\n🎬 Top 5 recommendations for user {sampleUser}:")
        for item, score in recs.items():
            print(f"  • {item:45s}  predicted rating: {score:.2f}")
    
    # Show user's actual high ratings
    print(f"\n🌟 Movies user {sampleUser} actually rated highly:")
    user_ratings = userItem.loc[sampleUser].dropna().sort_values(ascending=False).head(5)
    for item, rating in user_ratings.items():
        print(f"  • {item:45s}  rating: {rating:.1f}")