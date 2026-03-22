"""
4_app.py — Streamlit Web UI (v3)
=================================
Now includes a first-run onboarding flow that:
  1. Asks if the user wants AI-powered explanations (Claude)
  2. If yes, walks them through getting an Anthropic API key
  3. Validates the key before storing it in session state
  4. Falls back gracefully to explanation-free mode if they decline

All state is stored in st.session_state — Streamlit's way of persisting
data across re-runs without a database.

Run with:  streamlit run 4_app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.card {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
    border: 1px solid #2a2a4a;
    border-left: 4px solid #e94560;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.card.surprise { border-left-color: #f5a623; }
.card-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.25rem;
    color: #ffffff;
    margin: 0 0 2px 0;
}
.card-meta  { color: #888; font-size: 0.82rem; margin-bottom: 0.6rem; }
.badge {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 20px;
    font-size: 0.76rem;
    font-weight: 500;
    margin-right: 4px;
    margin-bottom: 4px;
}
.badge-match    { background: #e94560; color: #fff; }
.badge-surprise { background: #f5a623; color: #111; }
.badge-genre    { background: #2a2a4a; color: #aab; }
.badge-oscar    { background: #bfa040; color: #111; }
.badge-imax     { background: #1e4d6b; color: #9df; }
.badge-musical  { background: #4b2260; color: #daf; }
.badge-classic  { background: #2d4a1e; color: #aed681; }
.card-expl  { color: #ccc; line-height: 1.65; margin-top: 0.5rem; }
.card-plot  { color: #777; font-size: 0.82rem; margin-top: 0.4rem; }
.stat-row   { display: flex; gap: 1.4rem; font-size: 0.82rem;
              color: #999; margin-top: 0.5rem; }

/* Onboarding card */
.onboard-box {
    background: #0f0f1a;
    border: 1px solid #2a2a4a;
    border-radius: 16px;
    padding: 2.5rem 2.8rem;
    max-width: 640px;
    margin: 3rem auto;
}
.onboard-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #fff;
    margin-bottom: 0.3rem;
}
.onboard-sub { color: #888; margin-bottom: 1.6rem; line-height: 1.6; }
.step-num {
    display: inline-block;
    background: #e94560;
    color: #fff;
    width: 22px; height: 22px;
    border-radius: 50%;
    text-align: center;
    line-height: 22px;
    font-size: 0.75rem;
    font-weight: 700;
    margin-right: 8px;
}
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ────────────────────────────────────────────────────
# st.session_state persists across Streamlit re-runs for the same browser session.
# We use it to track where the user is in the onboarding flow and store their key.

if "onboarding_complete" not in st.session_state:
    st.session_state.onboarding_complete = False

if "use_claude" not in st.session_state:
    # If a key is already in the environment (from .env), skip asking
    st.session_state.use_claude = bool(os.getenv("ANTHROPIC_API_KEY"))

if "anthropic_key" not in st.session_state:
    st.session_state.anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")

if "onboarding_step" not in st.session_state:
    # Steps: "ask" → "get_key" → "done"
    st.session_state.onboarding_step = "ask"

# If a .env key was found, skip onboarding entirely
if st.session_state.anthropic_key:
    st.session_state.onboarding_complete = True


# ── API key validator ─────────────────────────────────────────────────────────

def validate_anthropic_key(key: str) -> tuple[bool, str]:
    """
    Make a minimal API call to verify the key works.
    Returns (is_valid, error_message).

    We use a very small max_tokens (10) so this costs essentially nothing —
    just enough to confirm authentication succeeds.
    """
    if not key.startswith("sk-ant-"):
        return False, "Key should start with 'sk-ant-'. Double-check what you copied."

    try:
        from anthropic import Anthropic, AuthenticationError, APIError
        client = Anthropic(api_key=key)
        client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": "hi"}],
        )
        return True, ""
    except AuthenticationError:
        return False, "That key wasn't accepted. Make sure you copied the full key."
    except APIError as e:
        return False, f"API error: {e}"
    except Exception as e:
        return False, f"Unexpected error: {e}"


# ── Onboarding flow ───────────────────────────────────────────────────────────

def show_onboarding():
    """
    Multi-step onboarding rendered in the main content area.
    Replaces the normal UI until the user completes or skips setup.
    """

    # ── Step 1: Ask if they want Claude ──────────────────────────────────────
    if st.session_state.onboarding_step == "ask":

        st.markdown("""
<div class="onboard-box">
  <div class="onboard-title">🎬 Welcome to CineMatch</div>
  <div class="onboard-sub">
    Your personal movie recommendation engine — built around your taste,
    not an algorithm's guess.<br><br>
    Before we start, one quick question:
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("### Would you like AI-powered explanations?")
        st.markdown(
            "CineMatch works great on its own — it uses vector similarity search "
            "to find films that genuinely match your taste. But if you add a free "
            "**Claude API key**, it will also write a personalized explanation of "
            "*why* each film fits you specifically, referencing your favorites and "
            "stated preferences.\n\n"
            "You control this key. It lives only in your session and is never stored anywhere."
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✨ Yes, I want AI explanations", type="primary",
                         use_container_width=True):
                st.session_state.onboarding_step = "get_key"
                st.rerun()
        with col2:
            if st.button("Skip — use without AI", use_container_width=True):
                st.session_state.use_claude         = False
                st.session_state.onboarding_complete = True
                st.rerun()

    # ── Step 2: Walk them through getting a key ───────────────────────────────
    elif st.session_state.onboarding_step == "get_key":

        st.markdown("### Getting your Anthropic API key")

        st.markdown("""
Here's exactly how to get one — it takes about 2 minutes and the free credits
Anthropic gives you are more than enough to explore this app:
""")

        # Step-by-step instructions
        st.markdown("""
<div style="line-height:2.2; font-size:0.97rem;">
  <div><span class="step-num">1</span>
    Go to <a href="https://console.anthropic.com" target="_blank"
    style="color:#e94560;">console.anthropic.com</a> and create a free account.
  </div>
  <div><span class="step-num">2</span>
    Once logged in, click <strong>API Keys</strong> in the left sidebar.
  </div>
  <div><span class="step-num">3</span>
    Click <strong>Create Key</strong>, give it any name (e.g. "CineMatch"),
    and click <strong>Create</strong>.
  </div>
  <div><span class="step-num">4</span>
    Copy the key — it starts with <code>sk-ant-</code>. 
    You won't be able to see it again after closing the dialog, so copy it now.
  </div>
  <div><span class="step-num">5</span>
    Paste it in the field below and click <strong>Verify & Continue</strong>.
  </div>
</div>
""", unsafe_allow_html=True)

        st.info(
            "**What does it cost?** Anthropic gives every new account free credits "
            "that cover hundreds of recommendation sessions. After that, each session "
            "costs a fraction of a cent. You can set a spending cap in your Anthropic "
            "console so you're never charged more than you expect.",
            icon="💡"
        )

        st.markdown("---")

        # Key input — type="password" masks the key as they type
        key_input = st.text_input(
            "Paste your API key here",
            type="password",
            placeholder="sk-ant-api03-...",
            help="Your key is only stored in this browser session and never sent anywhere except directly to Anthropic's servers.",
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            verify_btn = st.button("Verify & Continue →", type="primary",
                                   use_container_width=True)
        with col2:
            if st.button("← Back", use_container_width=True):
                st.session_state.onboarding_step = "ask"
                st.rerun()

        if verify_btn:
            if not key_input.strip():
                st.error("Please paste your API key first.")
            else:
                with st.spinner("Verifying your key with Anthropic..."):
                    valid, err = validate_anthropic_key(key_input.strip())

                if valid:
                    st.session_state.anthropic_key   = key_input.strip()
                    st.session_state.use_claude       = True
                    st.session_state.onboarding_complete = True
                    st.success("Key verified! AI explanations are enabled.")
                    import time; time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Couldn't verify that key. {err}")

        # Let them skip even from this screen
        st.markdown("---")
        if st.button("Skip — I'll add a key later"):
            st.session_state.use_claude          = False
            st.session_state.onboarding_complete = True
            st.rerun()


# ── Backend loader ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading movie index...")
def load_backend():
    import faiss, pickle
    from sentence_transformers import SentenceTransformer

    if not os.path.exists("embeddings/movies.faiss"):
        return None, None, None

    index = faiss.read_index("embeddings/movies.faiss")
    with open("embeddings/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, metadata, model


# ── Movie card renderer ───────────────────────────────────────────────────────

def render_movie_card(movie: dict, rank: int):
    is_surprise = movie.get("is_surprise", False)
    title       = movie.get("title", "Unknown")
    year        = str(movie.get("release_date", ""))[:4]
    director    = movie.get("director", "")
    genres      = movie.get("genres", [])
    rating      = movie.get("vote_average", 0)
    runtime     = movie.get("runtime_mins")
    runtime_cls = movie.get("runtime_class", "")
    score       = movie.get("similarity_score", 0)
    explanation = movie.get("explanation", "")
    overview    = movie.get("overview", "")[:240]
    uniqueness  = movie.get("uniqueness_score", 0)
    is_oscar    = movie.get("is_best_picture", False)
    is_imax     = movie.get("imax_likely", False)
    is_musical  = movie.get("is_musical", False)

    card_class  = "card surprise" if is_surprise else "card"

    if is_surprise:
        match_badge = '<span class="badge badge-surprise">🎲 Surprise pick</span>'
    else:
        match_badge = f'<span class="badge badge-match">{score}% match</span>'

    genre_badges   = "".join(f'<span class="badge badge-genre">{g}</span>'
                             for g in genres[:4])
    special_badges = ""
    if is_oscar:
        special_badges += '<span class="badge badge-oscar">🏆 Best Picture</span>'
    if is_imax:
        special_badges += '<span class="badge badge-imax">📽 IMAX</span>'
    if is_musical:
        special_badges += '<span class="badge badge-musical">🎵 Musical</span>'
    if movie.get("is_classic", False):
        special_badges += '<span class="badge badge-classic">🎞 Classic</span>'

    runtime_str = f"{runtime} min ({runtime_cls})" if runtime else runtime_cls

    if uniqueness >= 0.75:
        u_label = "🌀 Very unique plot"
    elif uniqueness >= 0.5:
        u_label = "✨ Distinctive"
    elif uniqueness >= 0.3:
        u_label = "📖 Familiar formula"
    else:
        u_label = "🔁 Classic tropes"

    st.markdown(f"""
<div class="{card_class}">
  <div class="card-title">{rank}. {title}</div>
  <div class="card-meta">{year}{f' · {director}' if director else ''}</div>
  <div>{match_badge}{genre_badges}{special_badges}</div>
  <div class="card-expl">{explanation}</div>
  <div class="card-plot">Plot: {overview}...</div>
  <div class="stat-row">
    <span>⭐ {rating}/10</span>
    <span>🕐 {runtime_str}</span>
    <span>{u_label}</span>
  </div>
</div>
""", unsafe_allow_html=True)

MOVIES_PATH   = "data/merged_movies.csv"
CREDITS_PATH  = "data/merged_credits.csv"

# ── Main app (shown after onboarding) ────────────────────────────────────────

def show_main_app():

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("# 🎬 CineMatch")

        # Show Claude status and let them change it
        if st.session_state.use_claude:
            st.success("✨ AI explanations on", icon="🤖")
        else:
            st.warning("AI explanations off", icon="💤")

        if st.button("Change AI settings", use_container_width=True):
            st.session_state.onboarding_complete = False
            st.session_state.onboarding_step     = "ask"
            st.rerun()

        st.divider()

        # Favorites
        st.subheader("Your Favorite Films")
        st.caption("One title per line")
        favorites_text = st.text_area(
            label="Favorites",
            value="Annihilation\nHereditary\nArrival\nMidsommar",
            height=130,
            label_visibility="collapsed",
        )

        uploaded_file = st.file_uploader(
            "Or upload Letterboxd ratings.csv",
            type=["csv"],
        )

        favorite_titles = [t.strip() for t in favorites_text.splitlines()
                           if t.strip()]

        if uploaded_file:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            try:
                from parse_letterboxd import get_favorite_titles
                lb = get_favorite_titles(tmp_path, min_stars=4.0)
                favorite_titles = lb
                st.success(f"Loaded {len(lb)} films (4★+) from Letterboxd")
            except Exception as e:
                st.error(f"Parse error: {e}")

        st.divider()

        st.subheader("What are you after?")
        preference_text = st.text_area(
            label="Preference",
            placeholder="e.g. Slow-burn psychological thriller with a female lead. No sequels.",
            height=90,
            label_visibility="collapsed",
        )

        st.divider()

        # Filters
        st.subheader("Filters")

        ALL_GENRES = [
            "Action", "Adventure", "Animation", "Comedy", "Crime",
            "Documentary", "Drama", "Family", "Fantasy", "History",
            "Horror", "Music", "Mystery", "Romance", "Science Fiction",
            "Thriller", "War", "Western",
        ]

        genres_include = st.multiselect("Must include genre(s)", ALL_GENRES,
                                        placeholder="Any genre")
        genres_exclude = st.multiselect("Exclude genre(s)", ALL_GENRES,
                                        placeholder="Nothing excluded")

        runtime_options = {
            "Any length":         [],
            "Short (≤ 90 min)":   ["short"],
            "Medium (91–105 min)":["medium"],
            "Long (> 105 min)":   ["long"],
            "Short or Medium":    ["short", "medium"],
        }
        runtime_classes = runtime_options[
            st.selectbox("Runtime", list(runtime_options.keys()))
        ]

        col1, col2 = st.columns(2)
        with col1:
            min_rating = st.number_input("Min score", 0.0, 10.0, 0.0, 0.5,
                                         format="%.1f")
        with col2:
            max_rating = st.number_input("Max score", 0.0, 10.0, 10.0, 0.5,
                                         format="%.1f")
        if min_rating == 0.0: min_rating = None
        if max_rating == 10.0: max_rating = None

        tier_options = {
            "Any audience":  [],
            "Family friendly": ["family"],
            "Teen & up":     ["teen", "adult"],
            "Adult only":    ["adult"],
        }
        rating_tiers = tier_options[
            st.selectbox("Audience", list(tier_options.keys()))
        ]

        st.caption("Plot originality (0 = clichéd, 1 = very unique)")
        u_range = st.slider("Uniqueness", 0.0, 1.0, (0.0, 1.0), 0.05)
        min_u   = u_range[0] if u_range[0] > 0.0 else None
        max_u   = u_range[1] if u_range[1] < 1.0 else None

        director_filter = st.text_input("Director contains",
                                        placeholder="e.g. Kubrick")
        actor_filter    = st.text_input("Actor contains",
                                        placeholder="e.g. Cate Blanchett")

        col3, col4 = st.columns(2)
        with col3:
            only_best_picture = st.toggle("🏆 Best Picture", value=False)
            only_imax         = st.toggle("📽 IMAX",         value=False)
            only_classic      = st.toggle("🎞 Classics only", value=False)
        with col4:
            musicals_choice = st.radio("Musicals", ["Either","Only","Exclude"])
        only_musicals = {"Either": None, "Only": True, "Exclude": False}[
            musicals_choice
        ]

        # Era / decade filter
        st.caption("Time period")
        ALL_DECADES = [
            "pre-1950s", "1950s", "1960s", "1970s",
            "1980s", "1990s", "2000s", "2010s", "2020s",
        ]
        era_decades = st.multiselect(
            "Decade(s)",
            options=ALL_DECADES,
            placeholder="Any era",
            label_visibility="collapsed",
        )

        st.divider()
        top_n = st.slider("Results", 3, 10, 5)

        col5, col6 = st.columns(2)
        with col5:
            recommend_btn = st.button("Find Films →", type="primary",
                                      use_container_width=True)
        with col6:
            surprise_btn  = st.button("🎲 Surprise Me",
                                      use_container_width=True)

    # ── Main content ──────────────────────────────────────────────────────────
    st.title("Your Recommendations")

    index, metadata, embed_model = load_backend()

    if index is None:
        st.error("Movie index not found. Run `python 1_build_index.py` first.",
                 icon="⚠️")
        st.stop()

    def build_filter_config():
        from recommender import FilterConfig
        return FilterConfig(
            genres_include    = genres_include,
            genres_exclude    = genres_exclude,
            directors         = [director_filter] if director_filter.strip() else [],
            actors            = [actor_filter]    if actor_filter.strip()    else [],
            runtime_classes   = runtime_classes,
            min_rating        = min_rating,
            max_rating        = max_rating,
            rating_tiers      = rating_tiers,
            only_musicals     = only_musicals,
            only_imax         = only_imax,
            only_best_picture = only_best_picture,
            min_uniqueness    = min_u,
            max_uniqueness    = max_u,
            era_decades       = era_decades,
            only_classic      = only_classic,
        )

    def run_recommendation(surprise: bool):
        if not favorite_titles:
            st.warning("Add at least one favorite film in the sidebar.")
            return

        pref = preference_text.strip() or "a great film I would love"
        cfg  = build_filter_config()

        # Pass the user's key directly into the environment for this call
        if st.session_state.use_claude and st.session_state.anthropic_key:
            os.environ["ANTHROPIC_API_KEY"] = st.session_state.anthropic_key

        use_claude = st.session_state.use_claude and bool(st.session_state.anthropic_key)

        from recommender import recommend, surprise_me

        if surprise:
            with st.spinner("🎲 Finding something unexpected..."):
                results = surprise_me(favorite_titles, pref,
                                      filters=cfg, top_n=top_n,
                                      use_claude=use_claude)
            st.subheader("🎲 Surprise Picks")
            st.caption("Deliberately outside your usual taste — still within your filters.")
        else:
            with st.spinner("Searching thousands of films..."):
                results = recommend(favorite_titles, pref,
                                    filters=cfg, top_n=top_n,
                                    use_claude=use_claude)
            st.subheader("Your Recommendations")

        if not results:
            st.warning(
                "No films matched all your filters. "
                "Try relaxing some constraints — widen the runtime, "
                "lower the minimum rating, or remove genre filters."
            )
            return

        st.caption(
            f"Based on: {', '.join(favorite_titles[:5])}"
            + (f" +{len(favorite_titles)-5} more" if len(favorite_titles) > 5 else "")
        )

        if not use_claude:
            st.info(
                "Showing plot summaries instead of personalized explanations. "
                "Click **Change AI settings** in the sidebar to add a Claude key.",
                icon="💡"
            )

        if any(m.get("actor_filter_relaxed") for m in results):
            st.warning(
                f"No films featuring {', '.join(cfg.actors)} were found in the dataset. "
                "Showing results based on your other preferences instead.",
                icon="⚠️"
        )

        st.divider()
        for i, movie in enumerate(results, 1):
            render_movie_card(movie, i)

    if recommend_btn:
        run_recommendation(surprise=False)
    elif surprise_btn:
        run_recommendation(surprise=True)
    else:
        st.info(
            "Configure your taste profile and filters in the sidebar, "
            "then click **Find Films →** or **🎲 Surprise Me**.",
            icon="👈"
        )
        with st.expander("How each filter works"):
            st.markdown("""
**Genre filters** — "Must include" means at least one selected genre must be present.
"Exclude" removes any film containing those genres.

**Runtime** — Short = ≤ 90 min, Medium = 91–105 min, Long = > 105 min.

**TMDB score** — film's rating out of 10. Use 7.0+ for acclaimed, 8.0+ for all-time greats.

**Audience tier** — inferred from genre signals (Horror → adult, Animation → family).

**Plot originality** — 0–1 score from how unusual the film's TMDB plot keywords are.

**IMAX** — heuristic: high-budget (>$80M) action/adventure/sci-fi.

**Best Picture** — matched against the full Academy Awards Best Picture winners list.

**🎞 Classics only** — filters to films released before 1980 (cinematic canon),
or films from 1980–1999 with a TMDB score of 7.5+ and at least 500 votes —
meaning they've genuinely stood the test of time, not just survived it.

**Time period / decade** — filters by the decade the film was released.
Select multiple decades to combine eras (e.g. 1970s + 1980s for New Hollywood
and its immediate aftermath). "pre-1950s" covers everything before 1950.

**🎲 Surprise Me** — inverts your taste vector mathematically to find films
in the opposite direction from your usual picks, while respecting your filters.
            """)


# ── Entry point — onboarding gate ────────────────────────────────────────────

if not st.session_state.onboarding_complete:
    show_onboarding()
else:
    show_main_app()
