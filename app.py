from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import requests

# Import your recommender code (no changes needed in that file)
from recommender.classical import (
    build_recommender_for_api,
    recommend_for_user
)

# Initialize Flask
app = Flask(__name__)
CORS(app)

print("🔧 Initializing recommender model... (this may take a moment)")
userItem, itemSim = build_recommender_for_api()
print("✅ Recommender initialized successfully!")


# Fetch popular movies
load_dotenv()
TMDB_KEY = os.getenv("TMDB_API_KEY")

@app.route("/api/movies")
def get_movies():
    if TMDB_KEY is None:
        return jsonify({"error": "TMDB_API_KEY missing in .env"}), 500

    url = f"https://api.themoviedb.org/3/movie/popular?api_key={TMDB_KEY}"

    response = requests.get(url)

    if response.status_code != 200:
        return jsonify({"error": "Failed to fetch from TMDB"}), 500

    movies = response.json().get("results", [])
    return jsonify(movies)


# API Route — Get Recommendations
@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    try:
        data = request.get_json()

        if not data or "userId" not in data:
            return jsonify({"error": "Missing userId"}), 400

        user_id = int(data["userId"])

        recs = recommend_for_user(user_id, userItem, itemSim)

        return jsonify({
            "userId": user_id,
            "recommendations": recs
        })

    except KeyError as e:
        return jsonify({"error": str(e)}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Health Check Route
@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


# Run the server
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
