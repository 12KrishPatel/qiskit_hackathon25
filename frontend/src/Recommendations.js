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
        ratings: ratings,
      });

      console.log("BACKEND RESPONSE →", res.data);
      setResults(res.data);

    } catch (err) {
      console.error(err);
      setError("Failed to fetch recommendations.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        padding: 30,
        minHeight: "100vh",
        backgroundColor: "#0b1e39",
        color: "white",
      }}
    >
      <h2 style={{ fontSize: 32, marginBottom: 10 }}>Get Recommendations</h2>

      <button
        onClick={fetchRecommendations}
        style={{
          padding: "10px 16px",
          fontSize: 16,
          borderRadius: 6,
          cursor: "pointer",
          border: "2px solid white",
          background: "transparent",
          color: "white",
        }}
      >
        Get Recommendations
      </button>

      {loading && <p style={{ marginTop: 15 }}>Loading...</p>}
      {error && <p style={{ color: "#ff8080", marginTop: 15 }}>{error}</p>}

      {results && (
        <div style={{ marginTop: 35 }}>
          <div
            style={{
              display: "flex",
              gap: "40px",
              justifyContent: "space-between",
              alignItems: "flex-start",
              flexWrap: "wrap",
            }}
          >

            {/* CLASSICAL column */}
            <div
              style={{
                flex: 1,
                minWidth: "320px",
                background: "white",
                color: "black",
                padding: 20,
                borderRadius: 10,
              }}
            >
              <h3>Classical Recommendations</h3>
              <ul style={{ paddingLeft: 20 }}>
                {Object.entries(results.classical).map(([movie, classicalScore]) => (
                  <li key={movie} style={{ marginBottom: 6 }}>
                    <strong>{movie}</strong> — predicted rating:{" "}
                    {Number(classicalScore).toFixed(2)}
                  </li>
                ))}
              </ul>
            </div>

            {/* QUANTUM column */}
            <div
              style={{
                flex: 1,
                minWidth: "320px",
                background: "white",
                color: "black",
                padding: 20,
                borderRadius: 10,
              }}
            >
              <h3>Quantum Recommendations</h3>
              <ul style={{ paddingLeft: 20 }}>
                {Object.entries(results.quantum).map(([movie, quantumScore]) => (
                  <li key={movie} style={{ marginBottom: 6 }}>
                    <strong>{movie}</strong>
                  </li>
                ))}
              </ul>
            </div>

          </div>
        </div>
      )}
    </div>
  );
}

export default Recommendations;
