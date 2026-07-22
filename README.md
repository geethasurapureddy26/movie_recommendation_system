# OTT Movie Recommendation System

A Flask-based web app that recommends movies/series based on user
preferences (type, genre, country, release year, rating, duration),
and shows which streaming platform each recommendation is on, with a
"Watch Now" button that redirects to that platform.

## Features

- Takes 6 user inputs: **Type** (Movie/Series), **Genre**, **Country**,
  **Release Year**, **Rating**, **Duration (minutes)**
- Returns the **top 5 recommended titles** that match the requested
  genre and country exactly, ranked by closeness to the requested
  rating/duration/year/type
- Displays each result's **platform** (Netflix, Prime Video, or Hotstar)
- Provides a **Watch Now** button that opens that platform's search
  results for the title in a new tab
- Falls back gracefully (with an on-screen note) if no exact
  genre+country match exists, without ever mixing in the wrong country

## Tech Stack

- **Backend:** Python, Flask
- **ML:** scikit-learn (`NearestNeighbors`, `MinMaxScaler`, `LabelEncoder`)
- **Data handling:** pandas
- **Model persistence:** joblib
- **Frontend:** HTML + CSS (Jinja2 templates)

## Project Structure

```
movierecommendation/
├── app.py                      # Flask app — routes and prediction logic
├── model.pkl                   # Trained NearestNeighbors model
├── scaler.pkl                  # Fitted MinMaxScaler
├── type_encoder.pkl            # LabelEncoder for movie/series type
├── genre_encoder.pkl           # LabelEncoder for genre
├── country_encoder.pkl         # LabelEncoder for country
├── ott_movies_clean_unique.csv # Source dataset (2,500 titles)
├── templates/
│   └── index.html              # Form + results page
└── static/
    └── style.css                # (optional) page styling
```

## Dataset

`ott_movies_clean_unique.csv` — 2,500 rows, columns:

| Column | Description |
|---|---|
| `title` | Movie/series title |
| `type` | Movie or Series |
| `genre` | Action, Comedy, Crime, Drama, Fantasy, Sci-Fi, Thriller |
| `platform` | Netflix, Prime Video, or Hotstar |
| `country` | Germany, India, South Korea, Spain, UK, or USA |
| `release_year` | Year released |
| `duration_minutes` | Runtime |
| `rating` | 0.0–10.0 |

> **Note:** this dataset contains generated/placeholder titles for
> demonstration purposes, not real movies. See **Limitations** below.

## How It Works

1. User submits the form (type, genre, country, year, rating, duration).
2. Inputs are label-encoded and scaled using the same encoders/scaler
   used during training.
3. The dataset is **hard-filtered** to rows matching the requested
   **country AND genre** exactly — this guarantees every result is
   relevant, not just "close" in some averaged sense.
4. Within that matched pool (drawn from the entire dataset, not a
   fixed subset), a `NearestNeighbors` model (cosine similarity) ranks
   rows by closeness to the requested year/rating/duration/type.
5. The top 5 are returned, each with a platform badge and a
   **Watch Now** link built from the platform's search URL + the
   movie title.

## Setup & Installation

```bash
# 1. Clone/copy the project folder, then install dependencies
pip install flask pandas scikit-learn joblib

# 2. Run the app
python app.py

# 3. Open in your browser
http://127.0.0.1:5000
```

All `.pkl` files and the CSV must sit in the same folder as `app.py`;
`index.html` must be inside a `templates/` subfolder (Flask requirement).

## Retraining the Model

If you change the dataset or feature set, retrain and re-save all 5
files together from the same script/notebook run — never regenerate
just one file on its own, or the encoders/scaler/model will fall out
of sync and predictions will error or behave incorrectly.

## Limitations & Possible Improvements

- **Synthetic data:** titles in the current dataset aren't real
  movies, so "Watch Now" can only open a **platform search page**
  for the title, not the exact real movie page. Swapping in a real
  dataset (e.g. via the TMDB API) would allow real per-title watch
  links.
- **Platform data freshness:** if switched to TMDB/JustWatch-sourced
  platform data, note that availability info updates roughly once a
  day, not instantly.
- **Possible next steps:** real movie data, poster images, user
  accounts/watch history, multi-country platform support, deployment
  to a public host (Render/Railway/PythonAnywhere).
