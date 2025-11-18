Quantum Recommender

A hybrid classical–quantum movie recommendation system built with React, Flask, and Qiskit.
Users rate movies in the frontend, and the backend generates recommendations using classical similarity scoring combined with a QAOA-optimized QUBO solution.

Features

- Interactive React interface for rating movies

- Classical cosine-similarity scoring

- QUBO formulation that balances movie relevance and diversity

- QAOA optimization using Qiskit

- Flask backend serving classical, quantum, and hybrid recommendations

- TMDB poster fetching

- Clean hybrid output combining quantum bitstrings with classical predictions

Quickstart (For Judges)

The following instructions run the full app locally in under a minute.

1. Clone the Repository
```
git clone https://github.com/12KrishPatel/qiskit_hackathon25
cd qiskit_hackathon25
```

2. Create and activate a virtual environment
```
cd backend
python3 -m venv venv
source venv/bin/activate
```

3. Install backend dependencies
```
pip install -r requirements.txt
```

4. Start the backend
```
python app.py
```

Backend runs at:
```
http://localhost:5001
```
Leave this terminal open.

Frontend Setup (React)

Open a second terminal window.

1. Install frontend dependencies
```
cd frontend
npm install
```

2. Start the frontend
```
npm start
```

Frontend runs at:
```
http://localhost:3000
```

Using the App

Open http://localhost:3000

Rate several movies (1–5)

Click Get Recommendations

The backend computes classical scores, runs QAOA, and returns a curated movie list

The UI will show posters and recommended movie titles.
