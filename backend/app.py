from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path

from recommender.classical import loadMovieLens
from recommender.hybrid import hybrid_recommend

app = Flask(__name__)
CORS(app)

df = loadMovieLens(Path("data/ml-100k"))


@app.get("/")
def health():
    return {"status": "ok", "message": "Backend running"}


@app.post("/recommend")
def recommend():
    data = request.json

    # Validate incoming data
    if "ratings" not in data or not isinstance(data["ratings"], dict):
        return jsonify({"error": "Missing or invalid 'ratings' field"}), 400

    user_ratings = data["ratings"]  # dict: movie → rating

    try:
        classical, quantum = hybrid_recommend(df, user_ratings)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "classical": classical.to_dict(),
        "quantum": quantum.to_dict()
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
