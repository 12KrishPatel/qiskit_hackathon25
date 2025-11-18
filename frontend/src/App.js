import React, { useEffect, useState } from "react";
import RateMovies from "./RateMovies";

function App() {
	const [movieList, setMovieList] = useState([]);

	useEffect(() => {
    fetch("http://localhost:5001/movies")
      .then((res) => res.json())
      .then((data) => {
        console.log("Loaded movies:", data.length);
  
        const formatted = data.slice(0, 100).map((t) => ({ title: t }));
        setMovieList(formatted);
      })
      .catch((err) => console.error("Failed to load movie list:", err));
  }, []);

	return <RateMovies movies={movieList} />;
}

export default App;
