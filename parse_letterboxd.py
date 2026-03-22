"""
2_parse_letterboxd.py — Parse your Letterboxd export
=====================================================
Letterboxd lets you export your entire diary as a CSV.
Go to: letterboxd.com → Settings → Import & Export → Export Your Data.
You'll get a ZIP with several CSVs. We want "ratings.csv".

What this file does:
  1. Reads the Letterboxd ratings.csv
  2. Filters to only your highly-rated films (4+ stars by default)
  3. Returns a list of film titles to use as taste seeds

Why this matters for CS:
  Real-world data ingestion. The CSV has weird column names, mixed date formats,
  rating scales that differ from TMDB's. This is the "Extract + Transform" part
  of an ETL pipeline — normalizing external data into your system's format.

Usage:
  python 2_parse_letterboxd.py --csv data/ratings.csv --min_stars 4.0
  
  Or import the function directly in 3_recommender.py:
    from parse_letterboxd import get_favorite_titles
"""

import argparse
import pandas as pd


# ── Core parsing function ─────────────────────────────────────────────────────

def get_favorite_titles(csv_path: str, min_stars: float = 4.0) -> list[str]:
    """
    Parse a Letterboxd ratings.csv and return a list of highly-rated movie titles.

    Letterboxd CSV format (columns we care about):
      Name        — movie title (string)
      Rating      — your star rating, 0.5–5.0 in 0.5 increments
      Watched Date — "YYYY-MM-DD"

    Args:
        csv_path:  path to ratings.csv from your Letterboxd export
        min_stars: minimum star rating to include (default 4.0)

    Returns:
        list of title strings, sorted by rating descending
    """
    print(f"Reading Letterboxd export: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not find {csv_path}. Export your data from letterboxd.com "
            "→ Settings → Import & Export → Export Your Data."
        )

    print(f"  Total logged films: {len(df)}")
    print(f"  Columns found: {list(df.columns)}")

    # ── Normalize column names ──────────────────────────────────────────────
    # Letterboxd column names have capital letters; normalize to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # ── Handle the rating column ────────────────────────────────────────────
    # Some exports use "rating", some use "rating (out of 5)"
    rating_col = next(
        (c for c in df.columns if "rating" in c.lower()),
        None
    )
    if rating_col is None:
        raise ValueError("Could not find a rating column in the CSV.")

    # Drop rows with no rating (films you logged but didn't rate)
    df = df.dropna(subset=[rating_col, "name"])

    # Cast rating to float for comparison
    df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")
    df = df.dropna(subset=[rating_col])

    # ── Filter to favorites ─────────────────────────────────────────────────
    favorites = df[df[rating_col] >= min_stars].copy()
    favorites = favorites.sort_values(rating_col, ascending=False)

    titles = favorites["name"].tolist()
    print(f"  Films rated {min_stars}+ stars: {len(titles)}")

    if len(titles) == 0:
        print(f"  Warning: no films found at {min_stars}+ stars. "
              "Try lowering --min_stars.")
    else:
        print(f"  Top 5 favorites: {titles[:5]}")

    return titles


# ── Summary stats (bonus) ─────────────────────────────────────────────────────

def print_taste_summary(csv_path: str):
    """
    Print a quick summary of your Letterboxd taste profile.
    Good for debugging or showing in the UI.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    rating_col = next((c for c in df.columns if "rating" in c), None)
    if not rating_col:
        print("No rating column found.")
        return

    df = df.dropna(subset=[rating_col])
    df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")
    df = df.dropna(subset=[rating_col])

    print("\n── Your Letterboxd stats ──────────────────────────")
    print(f"  Total rated films : {len(df)}")
    print(f"  Average rating    : {df[rating_col].mean():.2f} / 5.0")
    print(f"  5-star films      : {(df[rating_col] == 5.0).sum()}")
    print(f"  4.5-star films    : {(df[rating_col] == 4.5).sum()}")
    print(f"  4-star films      : {(df[rating_col] == 4.0).sum()}")
    print("────────────────────────────────────────────────────\n")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parse your Letterboxd ratings.csv to extract favorite films."
    )
    parser.add_argument(
        "--csv",
        default="data/ratings.csv",
        help="Path to your Letterboxd ratings.csv (default: data/ratings.csv)"
    )
    parser.add_argument(
        "--min_stars",
        type=float,
        default=4.0,
        help="Minimum star rating to include (default: 4.0)"
    )
    args = parser.parse_args()

    print_taste_summary(args.csv)
    titles = get_favorite_titles(args.csv, args.min_stars)

    print("\nFavorite titles extracted:")
    for i, t in enumerate(titles[:20], 1):
        print(f"  {i:2d}. {t}")
    if len(titles) > 20:
        print(f"  ... and {len(titles) - 20} more")


if __name__ == "__main__":
    main()
