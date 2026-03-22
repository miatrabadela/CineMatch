"""
3_recommender.py — Core recommendation engine (v2)
===================================================
Now supports structured filtering across all movie attributes and a
"Surprise Me" mode that deliberately avoids the obvious.

Two recommendation modes:
  recommend()    — Normal mode: taste vector + filters + Claude explanations
  surprise_me()  — Surprise mode: purposely searches AWAY from taste centroid
                   within filtered constraints

Filter system (FilterConfig dataclass):
  Every filter is optional — only applies if explicitly set.
  Post-filter happens AFTER FAISS retrieval, which means we over-fetch
  (k * 5) candidates from FAISS and then apply structured filters to the
  result set. This is called "retrieve then filter" and is the standard
  pattern for hybrid vector + structured search.

  Why not filter BEFORE the vector search?
    FAISS doesn't support structured predicates natively (unlike pgvector
    or Pinecone which support metadata filters at query time). For a dataset
    of this size, over-fetching + post-filtering is fast enough.
    At millions of records you'd use a vector DB with native filter support.
"""

import json
import os
import pickle
import random
import numpy as np
import faiss
from dataclasses import dataclass, field
from typing import Optional
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

INDEX_PATH    = "embeddings/movies.faiss"
METADATA_PATH = "embeddings/metadata.pkl"
MODEL_NAME    = "all-MiniLM-L6-v2"

# ── Filter configuration dataclass ───────────────────────────────────────────

@dataclass
class FilterConfig:
    """
    All filter fields are Optional — None means "no filter on this field".

    Using a dataclass here instead of a raw dict is a good CS practice:
    it gives you type hints, IDE autocomplete, and a clear contract for
    what filters are available. In a real system you'd validate these
    with Pydantic.

    Fields:
      genres_include    : movie must have AT LEAST ONE of these genres
      genres_exclude    : movie must have NONE of these genres
      directors         : movie must be directed by one of these names (fuzzy match)
      actors            : movie must star at least one of these actors (fuzzy match)
      runtime_classes   : list of allowed runtime classes ("short","medium","long")
      min_rating        : minimum TMDB vote_average (0–10 scale)
      max_rating        : maximum TMDB vote_average (useful for "cult" picks)
      rating_tiers      : list of allowed appropriateness tiers ("family","teen","adult")
      only_musicals     : True = only musicals, False = exclude musicals, None = either
      only_imax         : if True, only return IMAX-likely films
      only_best_picture : if True, only return Oscar Best Picture winners
      min_uniqueness    : 0.0–1.0 floor on uniqueness score (0.7+ = very unconventional)
      max_uniqueness    : 0.0–1.0 ceiling (use 0.4 to get more conventional/cliché films)
      era_decades       : list of decade strings to allow, e.g. ["1980s", "1990s"].
                          Empty list = no era filter (all years allowed).
                          Derived from release_date year at filter time — no extra
                          metadata field needed since release_date is already stored.
      only_classic      : if True, only return films marked is_classic = True.
                          A "classic" is defined as: released before 1980 OR
                          (released before 2000 AND vote_average >= 7.5 AND
                          vote_count >= 500). This captures the intuitive meaning —
                          older films that have stood the test of time and are still
                          widely regarded as essential viewing.
    """
    genres_include:    list[str]      = field(default_factory=list)
    genres_exclude:    list[str]      = field(default_factory=list)
    directors:         list[str]      = field(default_factory=list)
    actors:            list[str]      = field(default_factory=list)
    runtime_classes:   list[str]      = field(default_factory=list)   # "short","medium","long"
    min_rating:        Optional[float] = None   # TMDB score, 0–10
    max_rating:        Optional[float] = None
    rating_tiers:      list[str]      = field(default_factory=list)   # "family","teen","adult"
    only_musicals:     Optional[bool]  = None
    only_imax:         bool            = False
    only_best_picture: bool            = False
    min_uniqueness:    Optional[float] = None   # 0.0–1.0
    max_uniqueness:    Optional[float] = None
    era_decades:       list[str]      = field(default_factory=list)   # e.g. ["1980s","1990s"]
    only_classic:      bool            = False


# ── Load index & metadata ─────────────────────────────────────────────────────

def load_index_and_metadata():
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError("Run build_index.py first to build the index.")
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    # Load raw vectors for direct numpy lookup (avoids index.reconstruct() compatibility issues)
    vectors = np.load("embeddings/vectors.npy") if os.path.exists("embeddings/vectors.npy") else None
    print(f"Loaded index: {index.ntotal} movies.")
    return index, metadata, vectors


# ── Filtering ─────────────────────────────────────────────────────────────────

def _fuzzy_match(target: str, candidates: list[str]) -> bool:
    """
    Case-insensitive substring match.
    e.g. "Nolan" matches "Christopher Nolan"
    Used for director and actor filters where users rarely type full names.
    """
    target_lower = target.lower()
    return any(target_lower in c.lower() for c in candidates)


def passes_filters(movie: dict, cfg: FilterConfig) -> bool:
    """
    Return True if a movie passes all active filters in cfg.

    Each filter only activates if its value is non-empty / non-None.
    This is the "retrieve then filter" step that runs after FAISS search.

    CS note: this is O(n) over the candidate set, which is small (~50-200
    movies). It's also easy to extend — add a new field to FilterConfig
    and a new check here without touching anything else.
    """
    genres = set(movie.get("genres", []))

    # Genre inclusion — must contain at least one requested genre
    if cfg.genres_include:
        if not genres & {g.strip() for g in cfg.genres_include}:
            return False

    # Genre exclusion — must contain none of the excluded genres
    if cfg.genres_exclude:
        if genres & {g.strip() for g in cfg.genres_exclude}:
            return False

    # Director filter (fuzzy)
    if cfg.directors:
        director = movie.get("director", "")
        if not any(_fuzzy_match(d, [director]) for d in cfg.directors):
            return False

    # Actor filter (fuzzy — passes if ANY requested actor appears in cast)
    if cfg.actors:
        cast = movie.get("cast", [])
        if not any(_fuzzy_match(a, cast) for a in cfg.actors):
            return False

    # Runtime class filter
    if cfg.runtime_classes:
        if movie.get("runtime_class", "unknown") not in cfg.runtime_classes:
            return False

    # TMDB vote_average range
    rating = movie.get("vote_average", 0.0)
    if cfg.min_rating is not None and rating < cfg.min_rating:
        return False
    if cfg.max_rating is not None and rating > cfg.max_rating:
        return False

    # Appropriateness tier
    if cfg.rating_tiers:
        if movie.get("rating_tier", "unknown") not in cfg.rating_tiers:
            return False

    # Musical filter
    if cfg.only_musicals is True and not movie.get("is_musical", False):
        return False
    if cfg.only_musicals is False and movie.get("is_musical", False):
        return False

    # IMAX
    if cfg.only_imax and not movie.get("imax_likely", False):
        return False

    # Best Picture
    if cfg.only_best_picture and not movie.get("is_best_picture", False):
        return False

    # Uniqueness score range
    u = movie.get("uniqueness_score", 0.5)
    if cfg.min_uniqueness is not None and u < cfg.min_uniqueness:
        return False
    if cfg.max_uniqueness is not None and u > cfg.max_uniqueness:
        return False

    # Era / decade filter
    # We derive the decade from the stored release_date string ("YYYY-MM-DD").
    # e.g. "1994-09-23" → year 1994 → decade "1990s"
    # Films released before 1950 are all bucketed as "pre-1950s" since very few
    # users will want to distinguish 1930s from 1940s separately.
    # If era_decades is empty, all years pass.
    if cfg.era_decades:
        release = str(movie.get("release_date", ""))
        try:
            year = int(release[:4])
            if year < 1950:
                decade = "pre-1950s"
            else:
                decade = f"{(year // 10) * 10}s"   # 1994 → "1990s"
        except (ValueError, TypeError):
            decade = "unknown"
        if decade not in cfg.era_decades:
            return False

    # Classic filter
    # is_classic is pre-computed in 1_build_index.py and stored in metadata.
    # We check it here rather than re-deriving it so the logic lives in one place.
    if cfg.only_classic and not movie.get("is_classic", False):
        return False

    return True


# ── Taste vector builder ──────────────────────────────────────────────────────

def build_taste_vector(
    favorite_titles: list[str],
    preference_text: str,
    model: SentenceTransformer,
    index: faiss.IndexFlatL2,
    vectors: np.ndarray = None,
    title_weight: float = 0.6,
    preference_weight: float = 0.4,
) -> np.ndarray:
    """
    Build a single float32 vector representing the user's taste.

    Strategy:
      1. For each favorite title, embed it and find its nearest match in the
         index via FAISS. Retrieve the stored vector for that match.
      2. Average all retrieved vectors → taste centroid.
      3. Embed the free-text preference_text → preference vector.
      4. Weighted blend: centroid * title_weight + pref * preference_weight.
      5. L2-normalize the result so distances are cosine-equivalent.

    Returns shape (1, 384) — FAISS .search() expects a 2D array.
    """
    vecs = []
    for title in favorite_titles:
        q   = model.encode([title], convert_to_numpy=True).astype("float32")
        _, idxs = index.search(q, k=1)
        idx = idxs[0][0]
        if idx >= 0:
            if vectors is not None:
                vecs.append(vectors[idx])
            else:
                vecs.append(index.reconstruct(int(idx)))

    if not vecs:
        # Fallback: embed the preference text only
        pref_vec = model.encode([preference_text], convert_to_numpy=True)[0]
        return pref_vec.astype("float32").reshape(1, -1)

    centroid = np.mean(vecs, axis=0).astype("float32")
    pref_vec = model.encode([preference_text], convert_to_numpy=True)[0].astype("float32")

    blended = centroid * title_weight + pref_vec * preference_weight
    norm    = np.linalg.norm(blended)
    if norm > 0:
        blended /= norm

    return blended.reshape(1, -1)


# ── FAISS search + filter ─────────────────────────────────────────────────────

def search_and_filter(
    taste_vector: np.ndarray,
    index: faiss.IndexFlatL2,
    metadata: list[dict],
    cfg: FilterConfig,
    k_final: int = 10,
    exclude_titles: set[str] = None,
    reverse: bool = False,
) -> list[dict]:
    """
    FAISS search followed by structured post-filtering.

    Args:
      taste_vector  : shape (1, 384) query vector
      cfg           : FilterConfig — which structured filters to apply
      k_final       : how many results to return after filtering
      exclude_titles: titles to skip (e.g. user's favorites)
      reverse       : if True, sort by DESCENDING distance (for Surprise Me)

    The over-fetch multiplier (k_final * 8) ensures we have enough
    candidates after filtering. If filters are very restrictive, we may
    still return fewer than k_final results — that's handled gracefully.
    """
    exclude_titles = {t.lower() for t in (exclude_titles or [])}
    fetch_k        = min(k_final * 8, index.ntotal)

    distances, indices = index.search(taste_vector, k=fetch_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue

        movie = metadata[idx].copy()

        if movie["title"].lower() in exclude_titles:
            continue

        if not passes_filters(movie, cfg):
            continue

        # Convert L2 distance to a 0–100 similarity score
        movie["similarity_score"] = round(max(0, 100 - dist * 20), 1)
        movie["faiss_distance"]   = round(float(dist), 4)
        results.append(movie)

    # For Surprise Me: flip the order so we show least-similar first
    if reverse:
        results = sorted(results, key=lambda m: m["faiss_distance"], reverse=True)
    else:
        results = sorted(results, key=lambda m: m["faiss_distance"])
    # If actor filter returned nothing, retry without it and warn the user
    if not results and cfg.actors:
        cfg_relaxed = FilterConfig(
            **{k: v for k, v in cfg.__dict__.items() if k != "actors"}
        )
        cfg_relaxed.actors = []
        results = search_and_filter(
            taste_vector, index, metadata, cfg_relaxed,
            k_final=k_final,
            exclude_titles=exclude_titles,
            reverse=reverse,
        )
        for r in results:
            r["actor_filter_relaxed"] = True

    return results[:k_final]


# ── Claude explanation generator ──────────────────────────────────────────────

def generate_explanations(
    candidates: list[dict],
    favorite_titles: list[str],
    preference_text: str,
    filters_summary: str = "",
    surprise_mode: bool = False,
) -> list[dict]:
    """
    Call Claude to write personalized explanations for each candidate.

    The prompt changes depending on whether we're in normal or surprise mode:
      Normal  → explain WHY it matches the user's taste
      Surprise → explain what's UNEXPECTED or fresh about this pick

    We pass the filter summary so Claude knows what constraints were active —
    e.g. "user wanted only short films rated 8+" — so explanations can
    reference those choices.
    """
    client = Anthropic()

    movie_list = "\n".join(
        f"{i+1}. {m['title']} ({str(m.get('release_date',''))[:4]})"
        f" | Genres: {', '.join(m.get('genres', []))}"
        f" | Director: {m.get('director','?')}"
        f" | Rating: {m.get('vote_average','?')}/10"
        f" | Runtime: {m.get('runtime_mins','?')} min ({m.get('runtime_class','?')})"
        f" | Uniqueness: {m.get('uniqueness_score','?')}"
        f" | Best Picture: {'Yes' if m.get('is_best_picture') else 'No'}"
        f"\n   Plot: {m.get('overview','')[:220]}..."
        for i, m in enumerate(candidates)
    )

    if surprise_mode:
        mode_instruction = (
            "These are SURPRISE picks — intentionally outside the user's comfort zone "
            "but matched to their active filters. Explain what makes each film UNEXPECTED, "
            "FRESH, or UNLIKE what they usually watch. Be enthusiastic about the discovery."
        )
    else:
        mode_instruction = (
            "Explain WHY each film matches this user's specific taste — "
            "reference their favorites and stated preference. Be specific, like a "
            "knowledgeable friend who knows their taste deeply."
        )

    prompt = f"""You are a film expert and personal movie recommender.

User's favorite films: {', '.join(favorite_titles[:10])}
Their stated preference: "{preference_text}"
Active filters: {filters_summary or 'none'}

{mode_instruction}

Candidate films:
{movie_list}

Write a SHORT (2-3 sentence) personalized explanation for each film.
Respond with valid JSON only — no markdown fences, no preamble.
Format exactly:
[
  {{"title": "Film Title", "year": "YYYY", "explanation": "Your explanation here."}},
  ...
]"""

    message = Anthropic().messages.create(
        model="claude-opus-4-5",
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    explanations   = json.loads(raw.strip())
    expl_map       = {e["title"]: e["explanation"] for e in explanations}

    for movie in candidates:
        movie["explanation"] = expl_map.get(movie["title"], movie.get("overview", "")[:280])

    return candidates


# ── Public API ────────────────────────────────────────────────────────────────

def recommend(
    favorite_titles: list[str],
    preference_text: str,
    filters: FilterConfig = None,
    top_n: int = 5,
    use_claude: bool = True,
) -> list[dict]:
    """
    Normal recommendation mode.

    Returns top_n movies closest to the user's taste vector that also
    pass all active structured filters. Claude explains each match.
    """
    filters = filters or FilterConfig()
    model   = SentenceTransformer(MODEL_NAME)
    index, metadata, vectors = load_index_and_metadata()

    taste_vec = build_taste_vector(
        favorite_titles, preference_text, model, index
    )

    candidates = search_and_filter(
        taste_vec, index, metadata, filters,
        k_final=top_n,
        exclude_titles=set(favorite_titles),
        reverse=False,
    )

    filters_summary = _summarize_filters(filters)

    if use_claude and os.getenv("ANTHROPIC_API_KEY") and candidates:
        candidates = generate_explanations(
            candidates, favorite_titles, preference_text,
            filters_summary=filters_summary,
            surprise_mode=False,
        )
    else:
        for c in candidates:
            c["explanation"] = c.get("overview", "")[:300]

    return candidates


def surprise_me(
    favorite_titles: list[str],
    preference_text: str,
    filters: FilterConfig = None,
    top_n: int = 5,
    use_claude: bool = True,
) -> list[dict]:
    """
    Surprise Me mode — intentionally picks films UNLIKE the usual taste.

    How it works:
      1. Build the normal taste vector (centroid of favorites).
      2. Invert it by negating the vector — this points AWAY from everything
         the user usually watches.
      3. Search using the inverted vector but still apply their filters
         (so they still get the runtime/genre/rating constraints they care about).
      4. Optionally shuffle the results slightly so it feels unpredictable.

    The inverted vector trick is a simple but effective way to get
    "anti-recommendations." In embedding space, negating a vector moves
    you to the opposite end of the semantic space — furthest from your taste.

    For extra variety we inject a small random perturbation so consecutive
    "Surprise Me" clicks don't always return the same films.
    """
    filters = filters or FilterConfig()
    model   = SentenceTransformer(MODEL_NAME)
    index, metadata, vectors = load_index_and_metadata()

    taste_vec = build_taste_vector(
    favorite_titles, preference_text, model, index, vectors=vectors
    )

    # Invert + add random noise for variety
    inverted = -taste_vec
    noise    = np.random.normal(0, 0.15, inverted.shape).astype("float32")
    inverted = inverted + noise

    # Re-normalize
    norm = np.linalg.norm(inverted)
    if norm > 0:
        inverted /= norm

    # Fetch more candidates (reverse=False here because we already inverted the query)
    candidates = search_and_filter(
        inverted, index, metadata, filters,
        k_final=top_n * 2,
        exclude_titles=set(favorite_titles),
        reverse=False,
    )

    # Shuffle and trim — adds unpredictability
    random.shuffle(candidates)
    candidates = candidates[:top_n]

    # Mark them as surprise picks
    for c in candidates:
        c["is_surprise"] = True

    filters_summary = _summarize_filters(filters)

    if use_claude and os.getenv("ANTHROPIC_API_KEY") and candidates:
        candidates = generate_explanations(
            candidates, favorite_titles, preference_text,
            filters_summary=filters_summary,
            surprise_mode=True,
        )
    else:
        for c in candidates:
            c["explanation"] = c.get("overview", "")[:300]

    return candidates


def _summarize_filters(cfg: FilterConfig) -> str:
    """Build a human-readable summary of active filters for the Claude prompt."""
    parts = []
    if cfg.genres_include:
        parts.append(f"genres include: {', '.join(cfg.genres_include)}")
    if cfg.genres_exclude:
        parts.append(f"genres exclude: {', '.join(cfg.genres_exclude)}")
    if cfg.directors:
        parts.append(f"director: {', '.join(cfg.directors)}")
    if cfg.actors:
        parts.append(f"starring: {', '.join(cfg.actors)}")
    if cfg.runtime_classes:
        parts.append(f"runtime: {', '.join(cfg.runtime_classes)}")
    if cfg.min_rating:
        parts.append(f"min TMDB rating: {cfg.min_rating}")
    if cfg.rating_tiers:
        parts.append(f"audience: {', '.join(cfg.rating_tiers)}")
    if cfg.only_musicals is True:
        parts.append("musicals only")
    if cfg.only_musicals is False:
        parts.append("no musicals")
    if cfg.only_imax:
        parts.append("IMAX films only")
    if cfg.only_best_picture:
        parts.append("Best Picture winners only")
    if cfg.min_uniqueness:
        parts.append(f"uniqueness >= {cfg.min_uniqueness}")
    if cfg.max_uniqueness:
        parts.append(f"uniqueness <= {cfg.max_uniqueness} (more conventional)")
    if cfg.era_decades:
        parts.append(f"era: {', '.join(cfg.era_decades)}")
    if cfg.only_classic:
        parts.append("classics only")
    return "; ".join(parts) if parts else "none"


# ── CLI demo ──────────────────────────────────────────────────────────────────

def main():
    print("── CineMatch v2 ───────────────────────────────────────────────────\n")

    favorites  = ["Annihilation", "Hereditary", "Arrival", "Midsommar"]
    preference = "Atmospheric slow-burn with a strong sense of dread. Not a sequel."

    # Example: only long films, highly rated, no horror, very unique plots
    filters = FilterConfig(
        genres_exclude=["Horror"],
        runtime_classes=["long"],
        min_rating=7.0,
        min_uniqueness=0.6,
    )

    print("=== Normal recommendations ===")
    results = recommend(favorites, preference, filters=filters, top_n=3)
    for i, m in enumerate(results, 1):
        print(f"\n{i}. {m['title']} ({str(m.get('release_date',''))[:4]})")
        print(f"   Genres: {', '.join(m.get('genres',[]))}")
        print(f"   Rating: {m.get('vote_average')}/10  |  "
              f"Runtime: {m.get('runtime_class')}  |  "
              f"Uniqueness: {m.get('uniqueness_score')}")
        print(f"   {m['explanation']}")

    print("\n\n=== Surprise Me! ===")
    surprises = surprise_me(favorites, preference, filters=FilterConfig(min_rating=6.5), top_n=3)
    for i, m in enumerate(surprises, 1):
        print(f"\n{i}. {m['title']} ({str(m.get('release_date',''))[:4]}) 🎲")
        print(f"   {m['explanation']}")


if __name__ == "__main__":
    main()
