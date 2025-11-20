#!/usr/bin/env python3
"""
QUBO Matrix Generator

This script reads data from the MovieLens workflow and generates a QUBO matrix
for the quantum movie recommendation system. It prints the linear and quadratic
terms of the QUBO to the console in a readable format.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from recommender.classical import (
    loadMovieLens,
    build_user_item,
    computeAdjustedSimilarity,
    recommendItems,
)
from recommender.qubo import build_pen_matrix, build_qubo_weights


def print_qubo_matrix(linear: np.ndarray, quadratic: np.ndarray, items: list[str]):
    """
    Print the QUBO matrix in a readable format.

    Args:
        linear: Linear coefficients (movie ratings)
        quadratic: Quadratic penalty matrix
        items: List of movie titles
    """
    n = len(items)

    print("\n" + "="*80)
    print("QUBO MATRIX FOR QUANTUM MOVIE RECOMMENDATION")
    print("="*80)

    # Print linear terms (reward for selecting highly-rated movies)
    print("\n📊 LINEAR TERMS (Movie Ratings - Higher is Better)")
    print("-" * 80)
    print(f"{'Movie Title':<50} {'Rating Score':>15}")
    print("-" * 80)
    for i, item in enumerate(items):
        print(f"{item[:48]:<50} {linear[i]:>15.4f}")

    # Print quadratic terms (penalties for selecting similar movies together)
    print("\n🔗 QUADRATIC TERMS (Similarity Penalties - Negative values)")
    print("-" * 80)
    print("These penalize selecting similar movies together to promote diversity")
    print("-" * 80)

    # Print as a matrix
    print(f"\n{'':>50}", end="")
    for i in range(min(n, 5)):  # Limit width for readability
        print(f"{f'x{i}':>10}", end="")
    print()
    print("-" * (50 + min(n, 5) * 10))

    for i in range(min(n, 10)):  # Limit rows for readability
        item_name = items[i][:48] if len(items[i]) > 48 else items[i]
        print(f"{item_name:>50}", end="")
        for j in range(min(n, 5)):
            if j > i:
                print(f"{quadratic[i, j]:>10.4f}", end="")
            else:
                print(f"{'':>10}", end="")
        print()

    if n > 10 or n > 5:
        print(f"\n(Matrix truncated for display - full size: {n}x{n})")

    # Print full QUBO matrix
    print("\n📐 FULL QUBO MATRIX (H = linear·x + x^T·Q·x)")
    print("-" * 80)
    full_matrix = quadratic.copy()
    np.fill_diagonal(full_matrix, linear)

    print("\nMatrix shape:", full_matrix.shape)
    print("\nFull matrix (diagonal = linear terms, upper triangle = quadratic penalties):")

    # Print condensed version
    max_display = min(n, 8)
    print(f"\n{'':>5}", end="")
    for i in range(max_display):
        print(f"{f'x{i}':>10}", end="")
    print()
    print("-" * (5 + max_display * 10))

    for i in range(max_display):
        print(f"{f'x{i}':>5}", end="")
        for j in range(max_display):
            if i == j:
                print(f"{full_matrix[i, j]:>10.4f}", end="")
            elif j > i:
                print(f"{full_matrix[i, j]:>10.4f}", end="")
            else:
                print(f"{'':>10}", end="")
        print()

    if n > max_display:
        print(f"\n(Displaying {max_display}x{max_display} subset of {n}x{n} matrix)")

    # Print statistics
    print("\n📈 QUBO STATISTICS")
    print("-" * 80)
    print(f"Number of movies (variables):     {n}")
    print(f"Linear term range:                 [{linear.min():.4f}, {linear.max():.4f}]")
    print(f"Linear term mean:                  {linear.mean():.4f}")
    print(f"Quadratic term range:              [{quadratic.min():.4f}, {quadratic.max():.4f}]")
    print(f"Quadratic term mean (non-zero):    {quadratic[quadratic != 0].mean():.4f}")
    print(f"Number of quadratic terms:         {np.count_nonzero(quadratic)}")
    print("="*80 + "\n")


def generate_qubo_from_workflow(
    sample_ratings: dict = None,
    top_n: int = 6,
    lam: float = 0.2
):
    """
    Generate a QUBO matrix from the MovieLens workflow.

    Args:
        sample_ratings: Dictionary of {movie_title: rating}. If None, uses example ratings.
        top_n: Number of top movies to include in QUBO
        lam: Lambda parameter for diversity penalty (default 0.2)
    """
    # Load MovieLens data
    print("Loading MovieLens data...")
    data_path = Path(__file__).parent / "data" / "ml-100k"
    df = loadMovieLens(data_path)
    print(f"✓ Loaded {len(df)} ratings")

    # Build user-item matrix and similarity
    print("\nBuilding user-item matrix and computing similarities...")
    userItem = build_user_item(df)
    itemSim = computeAdjustedSimilarity(userItem)
    print(f"✓ User-item matrix: {userItem.shape}")
    print(f"✓ Similarity matrix: {itemSim.shape}")

    # Use sample ratings or create example
    if sample_ratings is None:
        # Get some popular movies for the example
        available_movies = userItem.columns.tolist()
        sample_ratings = {
            available_movies[0]: 5,
            available_movies[1]: 4,
            available_movies[2]: 5,
            available_movies[3]: 3,
            available_movies[4]: 4,
        }
        print(f"\nUsing example ratings for demonstration:")
        for movie, rating in sample_ratings.items():
            print(f"  {movie}: {rating} stars")

    # Add new user with ratings
    print("\nAdding new user and generating recommendations...")
    new_user = userItem.index.max() + 1
    extended = userItem.copy()
    extended.loc[new_user] = float("nan")

    for movie, rating in sample_ratings.items():
        if movie in extended.columns:
            extended.at[new_user, movie] = float(rating)

    # Get classical recommendations
    classical_series = recommendItems(
        extended,
        itemSim,
        userId=new_user,
        topN=top_n,
        k=30,
    )

    print(f"✓ Generated {len(classical_series)} classical recommendations")

    # Get subset of similarity matrix for recommended movies
    items = classical_series.index.tolist()
    sim_subset = itemSim.loc[items, items]

    # Build QUBO components
    print(f"\nGenerating QUBO matrix (lambda={lam})...")
    penalty_matrix = build_pen_matrix(sim_subset)
    linear, quadratic = build_qubo_weights(classical_series, penalty_matrix, lam=lam)

    print(f"✓ QUBO matrix generated successfully")

    # Print the QUBO matrix
    print_qubo_matrix(linear, quadratic, items)

    return linear, quadratic, items


if __name__ == "__main__":
    print("\n🎬 QUBO Matrix Generator for Quantum Movie Recommendations")
    print("=" * 80)

    # Generate and print QUBO matrix
    # You can customize the sample_ratings parameter to use different user preferences
    linear, quadratic, movies = generate_qubo_from_workflow(
        sample_ratings=None,  # Will use example ratings
        top_n=6,              # Number of movies to include in QUBO
        lam=0.2              # Diversity penalty parameter
    )

    print("\n✅ QUBO matrix generation complete!")
    print("\nThe QUBO encodes the optimization problem:")
    print("  - Maximize: sum of movie ratings (linear terms)")
    print("  - Minimize: similarity penalties when selecting similar movies (quadratic terms)")
    print("  - Balance controlled by lambda parameter\n")
