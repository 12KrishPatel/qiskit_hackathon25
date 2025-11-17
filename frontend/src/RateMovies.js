import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import Recommendations from "./Recommendations";

const TMDB_API_KEY = process.env.REACT_APP_TMDB_KEY;

function cleanTitle(title) {
	if (!title) return "";
	return title
		.replace(/\(\d{4}\)/g, "")
		.replace(/, The$/i, "")
		.replace(/, A$/i, "")
		.trim();
}

function RateMovies({ movies = [] }) {
	const [posters, setPosters] = useState({});
	const [ratings, setRatings] = useState({});
	const scrollRef = useRef(null);

	const fetchPoster = async (title) => {
		const cleaned = cleanTitle(title);
		if (!TMDB_API_KEY) return null;

		try {
			const res = await axios.get("https://api.themoviedb.org/3/search/movie", {
				params: { api_key: TMDB_API_KEY, query: cleaned },
			});

			if (res.data.results.length > 0) {
				return (
					"https://image.tmdb.org/t/p/w500" +
					res.data.results[0].poster_path
				);
			}
		} catch (err) {
			console.error("Poster fetch error:", err);
		}

		return null;
	};

	useEffect(() => {
		if (!movies.length) return;
		const load = async () => {
			const p = {};
			for (let m of movies) {
				p[m.title] = await fetchPoster(m.title);
			}
			setPosters(p);
		};
		load();
	}, [movies]);

	const scrollLeft = () =>
		scrollRef.current?.scrollBy({ left: -500, behavior: "smooth" });

	const scrollRight = () =>
		scrollRef.current?.scrollBy({ left: 500, behavior: "smooth" });

	const handleRate = (title, stars) =>
		setRatings((r) => ({ ...r, [title]: stars }));

	return (
		<div style={{ padding: 20 }}>
			<h1>⭐ Rate Movies</h1>

			<div style={{ display: "flex", alignItems: "center" }}>
				<button onClick={scrollLeft}>⬅</button>

				<div
					ref={scrollRef}
					style={{
						display: "flex",
						overflowX: "auto",
						gap: "20px",
						padding: "20px 0",
						width: "100%",
					}}
				>
					{movies.map((movie) => (
						<div key={movie.title} style={{ minWidth: "180px" }}>
							<div
								style={{
									width: "150px",
									height: "225px",
									borderRadius: "10px",
									overflow: "hidden",
									background: "#ddd",
								}}
							>
								{posters[movie.title] ? (
									<img
										src={posters[movie.title]}
										alt={movie.title}
										style={{
											width: "100%",
											height: "100%",
											objectFit: "cover",
										}}
									/>
								) : (
									<div>No Image</div>
								)}
							</div>

							<h3 style={{ margin: 10 }}>{movie.title}</h3>

							<div>
								{[1, 2, 3, 4, 5].map((s) => (
									<span
										key={s}
										onClick={() => handleRate(movie.title, s)}
										style={{
											cursor: "pointer",
											color:
												ratings[movie.title] >= s
													? "gold"
													: "#ccc",
											fontSize: 20,
										}}
									>
										★
									</span>
								))}
							</div>
						</div>
					))}
				</div>

				<button onClick={scrollRight}>➡</button>
			</div>

			<Recommendations ratings={ratings} />
		</div>
	);
}

export default RateMovies;
