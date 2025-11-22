from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path

from recommender.classical import loadMovieLens
from recommender.hybrid import hybrid_recommend

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load MovieLens once at startup
df = loadMovieLens(Path("data/ml-100k"))


@app.get("/")
def health():
    return {"status": "ok", "message": "Backend running"}


@app.get("/movies")
def movies():
    movie_counts = df["item"].value_counts()
    titles = movie_counts.index.tolist()
    return jsonify(titles)


@app.post("/recommend")
def recommend():
    print("=== /recommend hit ===")
    data = request.get_json() or {}
    print("Incoming payload:", data)

    ratings = data.get("ratings", {})
    if not isinstance(ratings, dict):
        return jsonify({"error": "Invalid payload: 'ratings' must be a dict"}), 400

    try:
        classical, quantum = hybrid_recommend(df, ratings)

        response = {
            "classical": classical.to_dict(),
            "quantum": quantum,
        }

        print("Recommendation response ready.")
        return jsonify(response)

    except Exception as e:
        import traceback

        print("\n--- Error while computing recommendations ---")
        traceback.print_exc()
        print("--- End error ---\n")

        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
