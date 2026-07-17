from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load recommendation model
model = joblib.load("model.pkl")

# Load scaler
scaler = joblib.load("scaler.pkl")

# Load encoders
type_encoder = joblib.load("type_encoder.pkl")
genre_encoder = joblib.load("genre_encoder.pkl")
country_encoder = joblib.load("country_encoder.pkl")

# Load original dataset
df = pd.read_csv("ott_movies_clean_unique.csv")

FEATURE_COLS = ["type", "genre", "release_year", "rating", "duration_minutes", "votes", "country"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1. Gather raw form text data from the user
        movie_type = request.form["type"].strip()
        genre = request.form["genre"].strip()
        country = request.form["country"].strip()
        release_year = int(request.form["release_year"])
        rating = float(request.form["rating"])
        duration = int(request.form["duration_minutes"])
        votes = int(request.form["votes"])

        # 2. Encode the single row of user inputs
        movie_type_encoded = type_encoder.transform([movie_type])[0]
        genre_encoded = genre_encoder.transform([genre])[0]
        country_encoded = country_encoder.transform([country])[0]

        user_data = pd.DataFrame(
            [[
                movie_type_encoded,
                genre_encoded,
                release_year,
                rating,
                duration,
                votes,
                country_encoded
            ]],
            columns=FEATURE_COLS
        )
        user_scaled = scaler.transform(user_data)

        # 3. HARD FILTER on the entire dataset: only rows that match the
        #    requested genre AND country exactly. (Country/genre were
        #    previously just "soft" distance features, so a Thriller
        #    request could come back with Sci-Fi results — this fixes that.)
        matched_df = df[
            (df["country"] == country) &
            (df["genre"] == genre)
        ].copy()

        # Progressive relaxation ONLY if the exact genre+country combo has
        # no rows at all — still never relaxes country.
        relaxation_note = None
        if matched_df.empty:
            matched_df = df[df["country"] == country].copy()
            relaxation_note = (
                f"No {genre} titles found for {country} — "
                f"showing other genres from {country} instead."
            )
        if matched_df.empty:
            return render_template(
                "index.html",
                error=f"No movies found for country: {country}"
            )

        # 4. Drop duplicate titles so the same movie can't appear twice,
        #    and so re-querying similar inputs doesn't keep surfacing an
        #    identical short list drawn from a tiny candidate pool.
        matched_df = matched_df.drop_duplicates(subset="title").reset_index(drop=True)

        # 5. Within that matched set (which can be hundreds/thousands of
        #    rows — the WHOLE dataset, not a top-40 slice), rank by how
        #    close each row is to the requested rating, votes, duration,
        #    and release year, using the same scaler for a fair comparison.
        subset = pd.DataFrame({
            "type": type_encoder.transform(matched_df["type"]),
            "genre": genre_encoder.transform(matched_df["genre"]),
            "release_year": matched_df["release_year"],
            "rating": matched_df["rating"],
            "duration_minutes": matched_df["duration_minutes"],
            "votes": matched_df["votes"],
            "country": country_encoder.transform(matched_df["country"]),
        })[FEATURE_COLS]

        subset_scaled = scaler.transform(subset)

        k = min(5, len(matched_df))
        local_model = NearestNeighbors(n_neighbors=k, metric="cosine")
        local_model.fit(subset_scaled)

        distances, indices = local_model.kneighbors(user_scaled)

        # 6. These indices are positions within matched_df, so every
        #    result is guaranteed to match the requested country (and
        #    genre, when available).
        final_recommendations = matched_df.iloc[indices[0]][
            ["title", "genre", "rating", "country"]
        ]

        movies = final_recommendations.to_dict(orient="records")
        return render_template("index.html", movies=movies, note=relaxation_note)

    except Exception as e:
        return render_template("index.html", error=str(e))


if __name__ == "__main__":
    app.run(debug=True)
