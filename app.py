import streamlit as st
import pandas as pd
import re
import os

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Movie Recommendation System")
st.write("Content-based filtering using TF-IDF + Cosine Similarity")

# -----------------------------
# Load Dataset (cached)
# -----------------------------
@st.cache_data
def load_data():
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_HUB_TOKEN")
    dataset = load_dataset("jquigl/imdb-genres", token=token) if token else load_dataset("jquigl/imdb-genres")
    df = pd.DataFrame(dataset["train"])

    # Map source schema to app schema.
    df = df.rename(
        columns={
            "movie title - year": "title",
            "expanded-genres": "genres",
        }
    )
    df = df[["title", "description", "genres", "rating"]]
    df = df.dropna(subset=["title", "description"])
    df.reset_index(drop=True, inplace=True)
    return df

df = load_data()

# -----------------------------
# Preprocessing
# -----------------------------
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
stop_words = set(ENGLISH_STOP_WORDS)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

@st.cache_data
def preprocess(df):
    df = df.copy()
    df["clean_description"] = df["description"].apply(clean_text)
    return df

df = preprocess(df)

st.success(f"Loaded {len(df):,} movies.")
with st.expander("Preview dataset", expanded=False):
    st.dataframe(df[["title", "genres", "rating"]].head(10), use_container_width=True)

# -----------------------------
# Vectorization
# -----------------------------
@st.cache_resource
def vectorize(text_data):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(text_data)
    return vectorizer, X

vectorizer, X = vectorize(df['clean_description'])

# -----------------------------
# Recommendation Functions
# -----------------------------
def recommend_by_title(title, top_n=5):
    if title not in df['title'].values:
        return None
    
    idx = df[df['title'] == title].index[0]
    sim_scores = cosine_similarity(X[idx], X)[0]
    
    similar_indices = sim_scores.argsort()[::-1][1:top_n+1]
    return df.iloc[similar_indices][['title', 'genres', 'rating']]

def recommend_by_description(desc, top_n=5):
    desc_clean = clean_text(desc)
    vec = vectorizer.transform([desc_clean])
    sim_scores = cosine_similarity(vec, X)[0]
    
    similar_indices = sim_scores.argsort()[::-1][:top_n]
    return df.iloc[similar_indices][['title', 'genres', 'rating']]

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("⚙️ Options")

mode = st.sidebar.radio(
    "Choose Recommendation Type:",
    ["By Movie Title", "By Description"]
)

top_n = st.sidebar.slider("Number of Recommendations", 1, 10, 5)

# -----------------------------
# Main UI
# -----------------------------
if mode == "By Movie Title":
    st.subheader("🎥 Select a Movie")
    
    movie_list = df['title'].sort_values().unique()
    selected_movie = st.selectbox("Choose a movie:", movie_list)
    
    if st.button("Recommend"):
        results = recommend_by_title(selected_movie, top_n)
        
        if results is None:
            st.error("Movie not found!")
        else:
            st.success("Top Recommendations:")
            st.dataframe(results, use_container_width=True)

else:
    st.subheader("📝 Describe a Movie")
    
    user_input = st.text_area(
        "Enter a description (e.g., space adventure, romance, war...):"
    )
    
    if st.button("Recommend"):
        if user_input.strip() == "":
            st.warning("Please enter a description.")
        else:
            results = recommend_by_description(user_input, top_n)
            st.success("Top Recommendations:")
            st.dataframe(results, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Built with Streamlit | TF-IDF | Cosine Similarity")