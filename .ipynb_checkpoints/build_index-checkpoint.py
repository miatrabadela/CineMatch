"""
1_build_index.py — Build the enriched movie embedding index
============================================================
Unchanged pipeline structure, but we now extract and store much richer
metadata per movie so the filter layer in 3_recommender.py has real
signal to work with.

New fields extracted per movie:
  - cast           (top 5 actors, parsed from credits JSON)
  - director       (from crew JSON, job == "Director")
  - runtime        (minutes → short / medium / long classification)
  - keywords       (TMDB plot tags → used for uniqueness heuristic)
  - genres         (list, not a raw string)
  - is_musical     (derived from genres)
  - is_best_picture (title-matched against Oscar winners list)
  - imax_likely    (heuristic: high-budget action/sci-fi)
  - rating_tier    (family / teen / adult — inferred from genres)
  - uniqueness_score (0–1 heuristic from keyword rarity)

TMDB Dataset:
  Kaggle: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
  Download BOTH files:
    tmdb_5000_movies.csv
    tmdb_5000_credits.csv
  Place both in data/
"""

import ast
import os
import pickle
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

MOVIES_PATH   = "data/merged_movies.csv"
CREDITS_PATH  = "data/merged_credits.csv"
INDEX_PATH    = "embeddings/movies.faiss"
METADATA_PATH = "embeddings/metadata.pkl"
MODEL_NAME    = "all-MiniLM-L6-v2"
BATCH_SIZE    = 64

# ── Oscar Best Picture winners ────────────────────────────────────────────────
# TMDB free CSV has no award data, so we maintain this manually.
# Matching is case-insensitive on movie title.

BEST_PICTURE_WINNERS = {
    "wings", "broadway melody", "all quiet on the western front",
    "cimarron", "grand hotel", "cavalcade", "it happened one night",
    "mutiny on the bounty", "the great ziegfeld", "the life of emile zola",
    "you can't take it with you", "gone with the wind", "rebecca",
    "how green was my valley", "mrs. miniver", "casablanca",
    "going my way", "the lost weekend", "the best years of our lives",
    "gentleman's agreement", "hamlet", "all the king's men",
    "all about eve", "an american in paris", "the greatest show on earth",
    "from here to eternity", "on the waterfront", "marty",
    "around the world in 80 days", "the bridge on the river kwai",
    "gigi", "ben-hur", "the apartment", "west side story",
    "lawrence of arabia", "tom jones", "my fair lady", "the sound of music",
    "a man for all seasons", "in the heat of the night", "oliver!",
    "midnight cowboy", "patton", "the french connection", "the godfather",
    "the sting", "the godfather part ii", "one flew over the cuckoo's nest",
    "rocky", "annie hall", "the deer hunter", "kramer vs. kramer",
    "ordinary people", "chariots of fire", "gandhi", "terms of endearment",
    "amadeus", "out of africa", "platoon", "the last emperor",
    "rain man", "driving miss daisy", "dances with wolves",
    "the silence of the lambs", "unforgiven", "schindler's list",
    "forrest gump", "braveheart", "the english patient", "titanic",
    "shakespeare in love", "american beauty", "gladiator", "a beautiful mind",
    "chicago", "the lord of the rings: the return of the king",
    "million dollar baby", "crash", "the departed", "no country for old men",
    "slumdog millionaire", "the hurt locker", "the king's speech",
    "the artist", "argo", "12 years a slave", "birdman",
    "spotlight", "moonlight", "the shape of water", "green book",
    "parasite", "nomadland", "coda", "everything everywhere all at once",
    "oppenheimer",
}

# Keywords that appear in hundreds of films → signal a more conventional plot
COMMON_KEYWORDS = {
    "love", "friendship", "death", "family", "revenge", "betrayal",
    "survival", "good versus evil", "redemption", "based on novel",
    "police", "murder", "father son relationship", "mother daughter relationship",
    "high school", "wedding", "kidnapping", "drug", "romance",
    "coming of age", "superhero", "sequel", "based on comic book",
    "based on true story", "marriage",
}


# ── JSON column parsers ───────────────────────────────────────────────────────

def safe_parse(val) -> list:
    """
    TMDB stores lists as JSON strings inside CSV cells, e.g.:
      '[{"id": 28, "name": "Action"}, ...]'
    ast.literal_eval converts that string to an actual Python list.
    """
    if pd.isna(val):
        return []
    try:
        return ast.literal_eval(val)
    except Exception:
        return []


def extract_genres(genres_str) -> list[str]:
    return [g["name"] for g in safe_parse(genres_str)]


def extract_cast(cast_str, top_n: int = 5) -> list[str]:
    """Top-N billed actors from the cast JSON column."""
    return [c["name"] for c in safe_parse(cast_str)[:top_n]]


def extract_director(crew_str) -> str:
    """First crew member with job == 'Director'."""
    for member in safe_parse(crew_str):
        if member.get("job") == "Director":
            return member.get("name", "")
    return ""


def extract_keywords(keywords_str, top_n: int = 12) -> list[str]:
    """
    TMDB plot keywords (e.g. 'time travel', 'dystopia', 'unreliable narrator').
    We use these to compute a uniqueness score — movies with rare keywords
    tend to have more distinctive plots than movies with common ones.
    """
    return [k["name"] for k in safe_parse(keywords_str)[:top_n]]


# ── Derived feature functions ─────────────────────────────────────────────────

def compute_uniqueness_score(keywords: list[str], genres: list[str]) -> float:
    """
    Returns 0.0–1.0 where higher = more unique/unconventional.

    Algorithm:
      1. Compute what fraction of the movie's keywords are NOT in our
         "common keywords" set (i.e. rare, distinctive keywords)
      2. Apply a small penalty if the genre set is a subset of genres that
         historically skew toward formulaic plots

    This is a data-derived heuristic — imperfect but directionally useful,
    and a great talking point in interviews about feature engineering.
    """
    if not keywords:
        return 0.5  # no keyword data → neutral

    kw_lower     = {k.lower() for k in keywords}
    common_found = kw_lower & COMMON_KEYWORDS
    unique_ratio = 1.0 - (len(common_found) / max(len(kw_lower), 1))

    # Slight penalty for pure-genre films that tend toward formulaic plots
    formulaic_genres = {"Romance", "Action", "Comedy"}
    if set(genres) and set(genres).issubset(formulaic_genres):
        unique_ratio *= 0.85

    return round(min(max(unique_ratio, 0.0), 1.0), 3)


def classify_runtime(runtime_mins) -> str:
    """
    Classify runtime into short / medium / long.

    Thresholds from project spec:
      Short  : <= 90 min   (1.5 hours)
      Medium : 91–105 min  (between 1.5 and 1.75 hours)
      Long   : > 105 min   (> 1.75 hours)
    """
    try:
        mins = float(runtime_mins)
    except (TypeError, ValueError):
        return "unknown"

    if mins <= 90:
        return "short"
    elif mins <= 105:
        return "medium"
    else:
        return "long"


def infer_rating_tier(genres: list[str]) -> str:
    """
    Infer an appropriateness tier from genre signals.

    Returns: "family" | "teen" | "adult" | "unknown"

    Production improvement: use the OMDB API (omdbapi.com, free tier)
    to pull real MPAA ratings (G/PG/PG-13/R/NC-17) per film.
    """
    genre_set = set(genres)
    if "Animation" in genre_set or "Family" in genre_set:
        return "family"
    if "Horror" in genre_set or "Thriller" in genre_set:
        return "adult"
    if "Action" in genre_set or "Crime" in genre_set:
        return "teen"
    if "Drama" in genre_set and "Romance" in genre_set:
        return "teen"
    return "unknown"


# ── Document builder ──────────────────────────────────────────────────────────

def build_document(row: pd.Series, director: str, cast: list[str],
                   genres: list[str], keywords: list[str]) -> str:
    """
    Combine structured fields into one rich text string for embedding.

    Including director and cast names means the embedding model learns
    their stylistic associations — queries mentioning "Kubrick" or
    "Tilda Swinton" will naturally pull toward their filmographies.

    Example output:
      "Annihilation | Genres: Science Fiction, Horror, Mystery |
       Director: Alex Garland | Cast: Natalie Portman, Jennifer Jason Leigh |
       Themes: female protagonist, nature gone wrong, expedition |
       A biologist signs up for a dangerous, secret expedition..."
    """
    parts = [row.get("title", "")]

    if genres:
        parts.append(f"Genres: {', '.join(genres)}")
    if director:
        parts.append(f"Director: {director}")
    if cast:
        parts.append(f"Cast: {', '.join(cast[:3])}")
    if keywords:
        parts.append(f"Themes: {', '.join(keywords[:6])}")

    parts.append(row.get("overview", ""))
    return " | ".join(parts)


# ── Classic status helper ─────────────────────────────────────────────────────

def _is_classic(release_date: str, vote_average: float, vote_count: int) -> bool:
    """
    Determine whether a film qualifies as a "classic."

    Two paths to classic status:

    Path A — Pre-1980 (cinematic canon):
      Any film released before 1980 that has survived this long in the dataset
      is old enough to be considered foundational cinema. The vote_count filter
      in load_and_merge (>= 10) already removes truly obscure films.

    Path B — Pre-2000 with lasting critical standing:
      Films from 1980–1999 need to prove they've stood the test of time.
      We require vote_average >= 7.5 (well above the ~6.5 dataset average)
      and vote_count >= 500 (enough people still care about it to vote).
      This captures Pulp Fiction, Schindler's List, Goodfellas, etc. while
      filtering out forgotten films that are merely old.

    Not included in "classic":
      Films from 2000 onward — even beloved ones like There Will Be Blood or
      Mulholland Drive. "Classic" implies a proven track record across decades,
      and recent films haven't had enough time to demonstrate that yet. A future
      improvement could add a "modern classic" tier for post-2000 films with
      exceptional vote metrics.
    """
    try:
        year = int(str(release_date)[:4])
    except (ValueError, TypeError):
        return False

    if year < 1980:
        return True

    if year < 2000 and vote_average >= 7.5 and vote_count >= 500:
        return True

    return False




def build_metadata(row: pd.Series, director: str, cast: list[str],
                   genres: list[str], keywords: list[str]) -> dict:
    """
    Build the metadata dict saved parallel to the FAISS index.
    metadata[i] is the ground truth for vector i in the index.

    Every field here is available to the filter system in 3_recommender.py.
    """
    title        = row.get("title", "Unknown")
    runtime_mins = row.get("runtime", None)
    budget       = float(row.get("budget", 0) or 0)

    return {
        # Identity
        "title":        title,
        "overview":     row.get("overview", ""),
        "release_date": str(row.get("release_date", "")),

        # People
        "director": director,
        "cast":     cast,               # list[str], up to 5 names

        # Genre & content
        "genres":    genres,            # list[str]
        "keywords":  keywords,          # list[str]
        "is_musical": (
            "Music" in genres or "Musical" in genres
        ),

        # Ratings & popularity
        "vote_average": float(row.get("vote_average", 0.0)),
        "vote_count":   int(row.get("vote_count", 0)),
        "popularity":   float(row.get("popularity", 0.0)),

        # Runtime
        "runtime_mins":  int(runtime_mins) if pd.notna(runtime_mins) else None,
        "runtime_class": classify_runtime(runtime_mins),  # "short"/"medium"/"long"

        # Appropriateness tier (inferred)
        "rating_tier": infer_rating_tier(genres),         # "family"/"teen"/"adult"

        # Derived signals
        "uniqueness_score": compute_uniqueness_score(keywords, genres),  # 0.0–1.0
        "is_best_picture":  title.lower() in BEST_PICTURE_WINNERS,

        # IMAX heuristic
        # Real IMAX data: TMDB /movies/{id}/release_dates endpoint (requires API key)
        "imax_likely": (
            budget > 80_000_000
            and bool({"Action", "Adventure", "Science Fiction"} & set(genres))
        ),

        # Classic status
        # A film is considered a "classic" if it meets one of two criteria:
        #   A) Released before 1980 — old enough to be considered part of
        #      cinema's foundational canon regardless of rating
        #   B) Released before 2000 AND still highly rated (7.5+) AND
        #      has enough votes (500+) to prove lasting cultural relevance
        # This captures films like Chinatown, Apocalypse Now, Schindler's List,
        # and Pulp Fiction while excluding forgotten films that are merely old.
        "is_classic": _is_classic(
            release_date=str(row.get("release_date", "")),
            vote_average=float(row.get("vote_average", 0) or 0),
            vote_count=int(row.get("vote_count", 0) or 0),
        ),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)

    # Load & merge
    print(f"\nLoading {MOVIES_PATH}...")
    movies = pd.read_csv(MOVIES_PATH)

    if os.path.exists(CREDITS_PATH):
        print(f"Loading {CREDITS_PATH}...")
        credits = pd.read_csv(CREDITS_PATH)
        id_col  = "movie_id" if "movie_id" in credits.columns else "id"
        credits = credits.rename(columns={id_col: "id"})
        df = movies.merge(credits[["id", "cast", "crew"]], on="id", how="left")
    else:
        print("Credits CSV not found — cast/director fields will be empty.")
        df = movies
        df["cast"] = "[]"
        df["crew"] = "[]"

    df = df.dropna(subset=["overview", "title"])
    df = df[df["overview"].str.len() >= 20]
    if "vote_count" in df.columns:
        df = df[df["vote_count"] >= 10]
    df = df.reset_index(drop=True)
    print(f"Clean rows: {len(df)}")

    # Build documents + metadata
    print("\nProcessing movies...")
    documents = []
    metadata  = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        genres   = extract_genres(row.get("genres", "[]"))
        keywords = extract_keywords(row.get("keywords", "[]"))
        cast     = extract_cast(row.get("cast", "[]"))
        director = extract_director(row.get("crew", "[]"))

        documents.append(build_document(row, director, cast, genres, keywords))
        metadata.append(build_metadata(row, director, cast, genres, keywords))

    # Embed
    print(f"\nEmbedding {len(documents)} movies in batches of {BATCH_SIZE}...")
    all_vecs = []
    for i in tqdm(range(0, len(documents), BATCH_SIZE), desc="Embedding"):
        batch = documents[i : i + BATCH_SIZE]
        vecs  = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_vecs.append(vecs)

    embeddings = np.vstack(all_vecs).astype("float32")
    print(f"Matrix shape: {embeddings.shape}")

    # FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    print(f"FAISS vectors: {index.ntotal}")

    # Save
   # Save
    os.makedirs("embeddings", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
    np.save("embeddings/vectors.npy", embeddings)

    print(f"\n✓ {INDEX_PATH}")
    print(f"✓ {METADATA_PATH}")
    print(f"✓ embeddings/vectors.npy")

    # Quick sanity check — print one entry
    import json
    print("\nSample metadata entry:")
    print(json.dumps(metadata[0], indent=2, default=str))


if __name__ == "__main__":
    main()
