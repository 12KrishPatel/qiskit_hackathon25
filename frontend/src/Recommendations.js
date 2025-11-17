import React, { useState } from "react";
import axios from "axios";

function Recommendations({ ratings }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [results, setResults] = useState(null);

  const fetchRecommendations = async () => {
    setLoading(true);
    setError("");
    setResults(null);

    try {
      const res = await axios.post("http://localhost:5001/recommend", {
        ratings: ratings
      });

      setResults(res.data);
    } catch (err) {
      console.error(err);
      setError("Failed to fetch recommendations.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 20 }}>
      <h2>Get Recommendations</h2>
      <button onClick={fetchRecommendations}>Get Recommendations</button>

      {loading && <p>Loading...</p>}
      {error && <p style={{ color: "red" }}>{error}</p>}

      {results && (
        <div style={{ marginTop: 20 }}>
          {/* Classical */}
          <h3>Classical Recommendations</h3>
          <ul>
            {Object.entries(results.classical).map(([movie, score]) => (
              <li key={movie}>
                <strong>{movie}</strong> — predicted rating:{" "}
                {Number(score).toFixed(2)}
              </li>
            ))}
          </ul>

          {/* Quantum */}
          <h3>Quantum Recommendations</h3>
          <ul>
            {Object.entries(results.quantum).map(([movie, score]) => (
              <li key={movie}>
                <strong>{movie}</strong> — quantum score:{" "}
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
