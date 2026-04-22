# CineMatch
The Python program that matches movie watchers with their next best film. This is an AI-powered movie recommender that builds a personalized taste profile from your viewing history and uses vector similarity search to surface the exact right film for any mood, with full control over genre, runtime, tone, cast, and a dozen other preferences.
 ---
# What is it?
CineMatch is a movie recommendation app built for people who actually care about what they watch next. Instead of vague suggestions with no explanation, it lets you tell it exactly what you're looking for. Filter by genre, runtime, cast, director, audience rating, and more. You can type in some of your favorite films or upload your Letterboxd history, and CineMatch builds a personal taste profile from what you love. It then searches through thousands of movies to find what genuinely matches your taste. It also gives you the option to use AI to explain exactly why each recommendation fits you specifically, not just a generic plot summary. There's also a Surprise Me mode that deliberately finds something outside your usual picks while still respecting your preferences. Built because great films deserve to be found, and most recommendation algorithms don't dig deep enough to find them.
---
# Skills Used:
- Vector embeddings and semantic search (the technical core.) Unlike using keyword matching or collaborative filtering, like Netflix's early system, I'm representing meaning as geometry, as similar films literally live close together in 384-dimensional space. This is the same architecture used in production at Spotify, Pinterest, and Airbnb for recommendations at a large scale.
- ETL data pipeline. I do this by taking over nine messy CSVs from Kaggle (movies + credits), parsing nested JSON columns inside those CSVs, deriving new features (uniqueness score, runtime class, rating tier, IMAX heuristic), joining the nine datasets on a shared ID, cleaning bad rows, and producing a clean indexed dataset.
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
- Optional: Your Anthropic API Key — AI explanations are optional; the recommender works without them

<p><strong>Step 1 — Get the code</strong></p>

<p>Open a terminal in VSCode and run:</p>

<pre><code>git clone https://github.com/yourusername/CineMatch.git

cd CineMatch</code></pre>

<p>In bash terminal:</p>

<pre><code><strong>python -m venv venv</strong></code></pre>

<p>Then activate it:</p>

<pre><code><strong>source venv/bin/activate</strong>  # Mac/Linux/JupyterHub

<strong>venv\Scripts\activate</strong>  # Windows Command Prompt

<strong>.\venv\Scripts\Activate.ps1</strong>  # Windows PowerShell</code></pre>

<p>You’ll know it worked when you see <strong>(venv)</strong> at the start of your terminal line.</p>

<p>Then install all dependencies in the bash terminal:</p>

<pre><code><strong>pip install -r requirements.txt</strong></code></pre>

Step 2 — Add your Anthropic API key (optional)<br>
Create a .env file in the CineMatch folder:<br>
- **ANTHROPIC_API_KEY=sk-ant-your-key-here**<br>
If you skip this, the app will ask for your key on first launch.<br>

Step 3 — Download the movie dataset<br>
Dataset Setup:<br>
Download the following files and place them in the **data/** folder. None of these are committed to Git due to file size — you must download them manually.<br>
TMDB 5000 (required)<br>
Download from: kaggle.com/datasets/tmdb/tmdb-movie-metadata<br>
Place these two files in **data/**:<br>

- **tmdb_5000_movies.csv**
- **tmdb_5000_credits.csv**

The Movies Dataset (recommended — expands to 45,000 films)
Download from: kaggle.com/datasets/rounakbanik/the-movies-dataset
Place these files in data/:

- **movies_metadata.csv**
- **credits.csv**
- **keywords.csv**
- **links.csv**

MovieLens Tag Genome (recommended — improves search relevance dramatically)
Download from: grouplens.org/datasets/movielens (MovieLens 25M)
Place these files in data/:

- **genome-scores.csv**
- **genome-tags.csv**

Generated files (created automatically by the scripts — do not download)
These files are created when you run the pipeline scripts below:

- **merged_movies.csv** — created by merge_datasets.py
- **merged_credits.csv** — created by merge_datasets.py
- **enriched_movies.csv**  — created by enrich_tags.py


Step 4 — Build the movie index<br>
This is a one-time setup step that reads the dataset, converts every movie into a searchable vector, and saves it to disk. It takes around 5–10 minutes the first time.
In bash terminal:
- **python merge_datasets.py**<br>
- **python enrich_tags.py**<br>
- **python build_index.py**<br>
You'll see a progress bar as it processes movies. When it finishes, you'll see two new files appear in the embeddings/ folder. Note: build_index.py is a one-time step. After it finishes, you never run it again unless you change the dataset.<br>


Step 5 — Launch the app<br>
In bash terminal:
- **streamlit run 4_app.py**<br>
Open the URL printed in the terminal. If it doesn't, just open that address manually.<br>
The app will walk you through the rest — including whether you want to add a Claude API key for AI-powered explanations. That part is optional, and the app works fully without it.<br>

How to Run (returning users)<br>
Every time you come back to the project, just run:<br>
- **bashsource venv/bin/activate**
- **streamlit run 4_app.py**
That's it. The index loads from disk in seconds.

Optional — Letterboxd export<br>
If you have a Letterboxd account and want the app to build your taste profile from your real watch history automatically:
- Log in at letterboxd.com
- Go to Settings → Data → Export Your Data
- Download and unzip the file
- In the app, upload ratings.csv using the file uploader in the sidebar

Troubleshooting<br>
- **ModuleNotFoundError: No module named 'faiss'** → Make sure your virtual environment is active **(source venv/bin/activate)** then run **pip install faiss-cpu**
- **FAISS index not found** → Run the full data pipeline in order: **merge_datasets.py** → **enrich_tags.py** → **build_index.py**
- **EmptyDataError: No columns to parse from file** → Your CSV file is empty or corrupted. Re-download it from Kaggle and make sure it's fully unzipped before placing it in **data/**
- **Disk quota exceeded** → Clear your pip cache with **pip cache purge** and delete any raw dataset files you no longer need (the originals can be deleted once merged)
- **App opens but recommendations feel off** → Be descriptive in the preference box — full sentences work better than single words. The embedding model responds to context, not keywords.
