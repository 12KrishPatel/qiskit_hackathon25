from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path

from recommender.hybrid import hybrid_recommend
from recommender.classical import loadMovieLens

app = Flask(__name__)
CORS(app)

# Load MovieLens data once
DATA_PATH = Path("data/ml-100k")
df = loadMovieLens(DATA_PATH)

# Store ratings from the frontend user
# Example structure:
# { 1: {"Toy Story (1995)": 5, "Heat (1995)": 4} }
user_ratings = {}


@app.post("/api/rate")
def rate_movie():
    data = request.json
    user_id = data.get("user_id")
    movie = data.get("movie_title")
    rating = data.get("rating")

    if user_id is None or movie is None or rating is None:
        return jsonify({"error": "Missing fields"}), 400

    user_id = int(user_id)

    if user_id not in user_ratings:
        user_ratings[user_id] = {}

    user_ratings[user_id][movie] = float(rating)

    return jsonify({"status": "saved"})


@app.post("/api/recommend")
def recommend():
    data = request.json
    user_id = data.get("userId")

    if user_id is None:
        return jsonify({"error": "Missing userId"}), 400

    user_id = int(user_id)

    # If this user hasn't rated anything yet:
    if user_id not in user_ratings or len(user_ratings[user_id]) == 0:
        return jsonify({"error": "No ratings found for user"}), 400

    # Pass ACTUAL USER RATINGS into hybrid model
    classical_df, quantum_df = hybrid_recommend(df, user_ratings[user_id])

    return jsonify({
        "classical": classical_df.to_dict(),
        "quantum": quantum_df.to_dict(),
        "used_ratings": user_ratings[user_id]
    })


@app.get("/api/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
