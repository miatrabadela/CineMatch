"""
merge_datasets.py — Combine multiple movie datasets into one clean CSV
======================================================================
Run this BEFORE build_index.py whenever you add a new dataset.
Output: data/merged_movies.csv and data/merged_credits.csv

Supported datasets:
  - TMDB 5000 (kaggle.com/datasets/tmdb/tmdb-movie-metadata)
  - MovieLens + TMDB joined (kaggle.com/datasets/rounakbanik/the-movies-dataset)
    This one has 45,000 movies and is the best upgrade from TMDB 5000.

Add more datasets by adding another block in the merge_all() function.
"""

import pandas as pd
import os

OUTPUT_MOVIES  = "data/merged_movies.csv"
OUTPUT_CREDITS = "data/merged_credits.csv"


def load_keywords() -> pd.DataFrame:
    """
    keywords.csv from The Movies Dataset.
    Contains plot keywords per movie id — feeds the uniqueness score.
    """
    path = "data/keywords.csv"
    if not os.path.exists(path):
        print("  keywords.csv not found, skipping.")
        return pd.DataFrame()

    kw = pd.read_csv(path)
    kw["id"] = pd.to_numeric(kw["id"], errors="coerce")
    kw = kw.dropna(subset=["id"])
    kw["id"] = kw["id"].astype(int)
    print(f"  Keywords: {len(kw)} entries")
    return kw

def load_tmdb_5000() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Original TMDB 5000 dataset."""
    movies_path  = "data/tmdb_5000_movies.csv"
    credits_path = "data/tmdb_5000_credits.csv"

    if not os.path.exists(movies_path):
        print("  TMDB 5000 not found, skipping.")
        return pd.DataFrame(), pd.DataFrame()

    movies  = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    # Normalize id column name
    if "movie_id" in credits.columns:
        credits = credits.rename(columns={"movie_id": "id"})

    print(f"  TMDB 5000: {len(movies)} movies")
    return movies, credits


def load_movies_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    The larger 'The Movies Dataset' from Kaggle — 45,000 films.
    kaggle.com/datasets/rounakbanik/the-movies-dataset
    Download and place these files in data/:
      movies_metadata.csv
      credits.csv
    """
    movies_path  = "data/movies_metadata.csv"
    credits_path = "data/credits.csv"

    if not os.path.exists(movies_path):
        print("  Movies Dataset not found, skipping.")
        return pd.DataFrame(), pd.DataFrame()

    movies = pd.read_csv(movies_path, low_memory=False)

    # This dataset uses 'id' as a string — normalize to int for merging
    movies["id"] = pd.to_numeric(movies["id"], errors="coerce")
    movies = movies.dropna(subset=["id"])
    movies["id"] = movies["id"].astype(int)

    # Rename columns to match TMDB 5000 format
    if "original_title" in movies.columns and "title" not in movies.columns:
        movies = movies.rename(columns={"original_title": "title"})

    credits = pd.read_csv(credits_path) if os.path.exists(credits_path) else pd.DataFrame()
    if not credits.empty:
        credits["id"] = pd.to_numeric(credits["id"], errors="coerce")
        credits = credits.dropna(subset=["id"])
        credits["id"] = credits["id"].astype(int)

    print(f"  Movies Dataset: {len(movies)} movies")
    return movies, credits


def merge_all():
    """
    Load all available datasets, combine, deduplicate, and save.

    Deduplication strategy:
      1. Combine all movies into one DataFrame
      2. Normalize titles to lowercase for comparison
      3. Keep the row with the higher vote_count when duplicates exist
         (more votes = richer data)
      4. Reset index and save
    """
    print("Loading datasets...")

    all_movies  = []
    all_credits = []

    # Load each dataset
    m1, c1 = load_tmdb_5000()
    if not m1.empty:
        all_movies.append(m1)
        all_credits.append(c1)

    m2, c2 = load_movies_dataset()
    if not m2.empty:
        all_movies.append(m2)
        all_credits.append(c2)

    # Add more datasets here following the same pattern:
    # m3, c3 = load_your_dataset()
    # if not m3.empty:
    #     all_movies.append(m3)
    #     all_credits.append(c3)

    if not all_movies:
        print("No datasets found. Place CSV files in data/ and try again.")
        return

    # Combine movies
    print("\nMerging movies...")
    combined = pd.concat(all_movies, ignore_index=True)
    print(f"  Combined rows before dedup: {len(combined)}")

    # Deduplicate on title (case-insensitive)
    # Keep the entry with the highest vote_count — it has the most data
    combined["title_lower"] = combined["title"].str.lower().str.strip()
    combined["vote_count"]  = pd.to_numeric(
        combined.get("vote_count", 0), errors="coerce"
    ).fillna(0)

    combined = (
        combined
        .sort_values("vote_count", ascending=False)
        .drop_duplicates(subset="title_lower", keep="first")
        .drop(columns=["title_lower"])
        .reset_index(drop=True)
    )
    print(f"  Rows after dedup: {len(combined)}")

    # Combine credits
    if all_credits:
        combined_credits = pd.concat(
            [c for c in all_credits if not c.empty], ignore_index=True
        )
        combined_credits["id"] = pd.to_numeric(
            combined_credits["id"], errors="coerce"
        )
        combined_credits = combined_credits.dropna(subset=["id"])
        combined_credits["id"] = combined_credits["id"].astype(int)
        # Deduplicate credits by movie id
        combined_credits = combined_credits.drop_duplicates(
            subset="id", keep="first"
        ).reset_index(drop=True)
        print(f"  Credits rows: {len(combined_credits)}")
    else:
        combined_credits = pd.DataFrame()

    # Save
    os.makedirs("data", exist_ok=True)
    combined.to_csv(OUTPUT_MOVIES, index=False)
    print(f"\n✓ Saved {OUTPUT_MOVIES} ({len(combined)} movies)")

    if not combined_credits.empty:
        combined_credits.to_csv(OUTPUT_CREDITS, index=False)
        print(f"✓ Saved {OUTPUT_CREDITS} ({len(combined_credits)} credits)")
    # Merge keywords into movies if available
    keywords_df = load_keywords()
    if not keywords_df.empty and "id" in combined.columns:
        combined["id"] = pd.to_numeric(combined["id"], errors="coerce")
        combined = combined.merge(
            keywords_df.rename(columns={"keywords": "keywords_raw"}),
            on="id",
            how="left"
        )
        # If the dataset already has a keywords column, fill gaps with the new one
        if "keywords" in combined.columns:
            combined["keywords"] = combined["keywords"].fillna(
                combined.get("keywords_raw", "")
            )
        else:
            combined["keywords"] = combined.get("keywords_raw", "")
        combined = combined.drop(
            columns=["keywords_raw"], errors="ignore"
        )
        combined.to_csv(OUTPUT_MOVIES, index=False)
        print(f"✓ Keywords merged into {OUTPUT_MOVIES}")

if __name__ == "__main__":
    merge_all()