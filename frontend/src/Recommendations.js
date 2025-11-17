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
					<h3>Classical Recommendations</h3>
					<pre>{JSON.stringify(results.classical, null, 2)}</pre>

					<h3>Quantum Recommendations</h3>
					<pre>{JSON.stringify(results.quantum, null, 2)}</pre>
				</div>
			)}
		</div>
	);
}

export default Recommendations;
