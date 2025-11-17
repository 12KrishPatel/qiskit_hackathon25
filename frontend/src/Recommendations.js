import React, { useState } from "react";
import axios from "axios";

function Recommendations() {
  const [userId, setUserId] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [recs, setRecs] = useState([]);

  const fetchRecommendations = async () => {
    setLoading(true);
    setError("");
    setRecs([]);

    try {
      const response = await axios.post("http://127.0.0.1:5000/api/recommend", {
        userId: Number(userId),
      });

      const recommendations = response.data.recommendations || {};
      const formatted = Object.entries(recommendations);

      setRecs(formatted);
    } catch (err) {
      setError("Failed to fetch recommendations.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: "20px", fontFamily: "Arial" }}>
      <h1>🎬 Movie Recommendations</h1>

      {/* USER INPUT */}
      <div style={{ marginBottom: "15px" }}>
        <label>User ID:</label>
        <input
          type="number"
          value={userId}
          onChange={(e) => setUserId(e.target.value)}
          style={{ marginLeft: "10px", width: "80px" }}
        />

        <button
          onClick={fetchRecommendations}
          style={{ marginLeft: "15px", padding: "6px 12px" }}
        >
          Get Recommendations
        </button>
      </div>

      {/* LOADING */}
      {loading && <p>Loading...</p>}

      {/* ERROR */}
      {error && <p style={{ color: "red" }}>{error}</p>}

      {/* RESULTS */}
      {!loading && recs.length > 0 && (
        <div>
          <h2>Top Recommendations:</h2>
          <ul>
            {recs.map(([movie, score]) => (
              <li key={movie}>
                <strong>{movie}</strong> — predicted rating:{" "}
                {Number(score).toFixed(2)}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default Recommendations;
