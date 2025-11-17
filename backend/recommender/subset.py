import pandas as pd
# Returns smaller DF of most active users and most popular items (Quantum Testing)
def topnmSubset(df: pd.DataFrame, topUsers: int = 50, topItems: int = 50) -> pd.DataFrame:
    top_items = df["item"].value_counts().head(topItems).index
    top_users = df["user"].value_counts().head(topUsers).index
    return df[df["item"].isin(top_items) & df["user"].isin(top_users)].copy()