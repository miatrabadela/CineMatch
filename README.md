# CineMatch
The Python program that matches movie watchers with their next best film. This is an AI-powered movie recommender that builds a personalized taste profile from your viewing history and uses vector similarity search to surface the exact right film for any mood, with full control over genre, runtime, tone, cast, and a dozen other preferences.
 ---
# What is it?
CineMatch is a personal movie recommendation engine built for people who actually care about what they watch next. Rather than black-box suggestions, it gives users complete control over their next film, filtering by genre, runtime, plot originality, director, cast, MPAA-equivalent audience tier, musical or not, IMAX, Oscar pedigree, and more. Users can type their favorite films directly or upload their Letterboxd export CSV, and the system analyzes that viewing history to build a mathematical "taste profile." A sentence-transformer embedding model converts every movie's plot, genre, director, and cast into a 384-dimensional vector, and Facebook's FAISS library performs millisecond similarity search across thousands of films to find what's genuinely closest to your taste. Claude then reads the top candidates and writes a personalized explanation of why each film fits you specifically, not just a generic plot summary. A "Surprise Me" mode mathematically inverts the taste vector to deliberately surface films outside your comfort zone, while still respecting your filters. This was built because great films deserve to be found, and generic recommendation algorithms rarely dig deep enough to find them.
---
# Skills Used:
- Vector embeddings and semantic search (the technical core.) Unlike using keyword matching or collaborative filtering, like Netflix's early system, I'm representing meaning as geometry, as similar films literally live close together in 384-dimensional space. This is the same architecture used in production at Spotify, Pinterest, and Airbnb for recommendations at a large scale.
- ETL data pipeline. I do this by taking two messy CSVs from Kaggle (movies + credits), parsing nested JSON columns inside those CSVs, deriving new features (uniqueness score, runtime class, rating tier, IMAX heuristic), joining the two datasets on a shared ID, cleaning bad rows, and producing a clean indexed dataset.
- Retrieve-then-filter architecture (a pattern used in every production search and recommendation system.) FAISS handles the fast approximate retrieval, and structured Python filters handle the business logic.
- Feature engineering (the uniqueness score) I derived a meaningful signal (how unconventional a film's plot is) from raw TMDB keyword data, using a domain-informed heuristic.
- LLM integration as a reasoning layer (Claude isn't doing the recommendation, FAISS is.) Claude is doing what LLMs are actually good at: taking a small structured candidate set and generating nuanced natural-language reasoning about it.
- Vector space manipulation. The Surprise Me mode negates the taste vector to search in the opposite semantic direction.
---
# Why I Built it:<br>
Since I was 10, I've been a big film person, and I kept running into the same frustration: recommendation algorithms are built for engagement, not for the experience of actually watching a great movie. They optimize for what you'll click or what's paying the most, not for what you'll love. I wanted to build something that treats the viewer as someone with a genuine, specific taste. The filter system exists. Sometimes you need a short film because you only have 90 minutes, or you want a story that's entirely enveloping for hours, or you want to know if it's okay to watch with your family. Those constraints are real, and existing platforms mostly ignore them. The Surprise Me mode came from the same instinct, because sometimes you want to be pushed somewhere you wouldn't go on your own, but you still don't want something completely random. Inverting the taste vector gives you that mathematically.
---
# How to Run:
What you need before starting:<br>
- Python 3.10 or newer — download at python.org
- VSCode — download at code.visualstudio.com
- Git — download at git-scm.com
- A free Kaggle account — needed to download the movie dataset

Step 1 — Get the code<br>
Open a terminal in VSCode and run:<br>
- bashgit clone https://github.com/yourusername/CineMatch.git<br>
- cd CineMatch<br>


Step 2 — Set up your Python environment<br>
- bashpython -m venv venv<br>
Then activate it:<br>
Windows: venv\Scripts\activate<br>
Mac/Linux: source venv/bin/activate<br>
You'll know it worked when you see (venv) at the start of your terminal line.<br>
Then install all dependencies:<br>
- bashpip install -r requirements.txt<br>


Step 3 — Download the movie dataset<br>
Go to kaggle.com/datasets/tmdb/tmdb-movie-metadata<br>
Click Download — you'll get a ZIP file<br>
Unzip it and place both files inside the data/ folder in your project:<br>
- tmdb_5000_movies.csv
- tmdb_5000_credits.csv


Step 4 — Build the movie index<br>
This is a one-time setup step that reads the dataset, converts every movie into a searchable vector, and saves it to disk. It takes around 5–10 minutes the first time.
- bashpython 1_build_index.py<br>
You'll see a progress bar as it processes movies. When it finishes, you'll see two new files appear in the embeddings/ folder. You never need to run this again unless you swap out the dataset.<br>


Step 5 — Launch the app<br>
- bashstreamlit run 4_app.py
Your browser will open automatically at http://localhost:8501. If it doesn't, just open that address manually.<br>
The app will walk you through the rest — including whether you want to add a Claude API key for AI-powered explanations. That part is optional and the app works fully without it.<br>

Optional — Letterboxd export<br>
If you have a Letterboxd account and want the app to automatically build your taste profile from your real watch history:
- Log in at letterboxd.com
- Go to Settings → Import & Export → Export Your Data
- Download and unzip the file
- In the app, upload ratings.csv using the file uploader in the sidebar
