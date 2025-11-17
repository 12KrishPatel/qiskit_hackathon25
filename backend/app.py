from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path

from recommender.classical import loadMovieLens
from recommender.hybrid import hybrid_recommend

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

df = loadMovieLens(Path("data/ml-100k"))


@app.get("/")
def health():
    return {"status": "ok", "message": "Backend running"}

@app.get("/movies")
def movies():
    titles = sorted(df['item'].unique())
    return jsonify(titles)



@app.post("/recommend")
def recommend():
    print("/recommend CALLED", flush=True)

    data = request.json
    print("Incoming data:", data, flush=True)

    try:
        classical, quantum = hybrid_recommend(
            df,
            data["ratings"]
        )

        print("Hybrid result OK", flush=True)

        return jsonify({
            "classical": classical.to_dict(),
            "quantum": quantum if isinstance(quantum, dict) else quantum.to_dict()
        })

    except Exception as e:
        import traceback
        print("\n\nHYBRID ERROR", flush=True)
        traceback.print_exc()
        print("END ERROR\n\n", flush=True)

        return jsonify({"error": str(e)}), 500




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
