import re

import os
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

STOP_WORDS = set(ENGLISH_STOP_WORDS)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [word for word in words if word not in STOP_WORDS]
    return " ".join(words)


def build_dataset() -> pd.DataFrame:
    # Use HF auth when available to avoid unauthenticated rate-limit warnings.
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_HUB_TOKEN")
    if token:
        dataset = load_dataset("jquigl/imdb-genres", token=token)
    else:
        dataset = load_dataset("jquigl/imdb-genres")
    df = pd.DataFrame(dataset["train"])
    df = df.rename(
        columns={
            "movie title - year": "title",
            "expanded-genres": "genres",
        }
    )
    df = df[["title", "description", "genres", "rating"]]
    df = df.dropna(subset=["description", "title"]).reset_index(drop=True)
    df["clean_description"] = df["description"].apply(clean_text)
    return df


def train_model(df: pd.DataFrame):
    train_df, _ = train_test_split(df, test_size=0.2, random_state=42)
    train_df = train_df.reset_index(drop=True)

    vectorizer = TfidfVectorizer(max_features=5000)
    x_train = vectorizer.fit_transform(train_df["clean_description"])
    indices = (
        train_df.reset_index()
        .drop_duplicates(subset=["title"])
        .set_index("title")["index"]
        .astype(int)
    )

    return train_df, vectorizer, x_train, indices


def recommend_movies(title, train_df, x_train, indices, top_n=5):
    if title not in indices:
        return "Movie not found."

    idx = int(indices[title])
    sim_values = cosine_similarity(x_train[idx], x_train)[0]
    sim_scores = list(enumerate(sim_values))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1 : top_n + 1]
    movie_indices = [i[0] for i in sim_scores]
    return train_df.iloc[movie_indices][["title", "genres", "rating"]]


def recommend_from_description(desc, train_df, vectorizer, x_train, top_n=5):
    desc_clean = clean_text(desc)
    desc_vec = vectorizer.transform([desc_clean])
    sim_scores = cosine_similarity(desc_vec, x_train)
    sim_scores = list(enumerate(sim_scores[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:top_n]
    movie_indices = [i[0] for i in sim_scores]
    return train_df.iloc[movie_indices][["title", "genres", "rating"]]


def main():
    print("Loading data and training recommender...")
    df = build_dataset()
    train_df, vectorizer, x_train, indices = train_model(df)

    print("\nRecommendations by title:")
    print(recommend_movies("Avatar - 2009", train_df, x_train, indices))

    print("\nRecommendations by description:")
    print(
        recommend_from_description(
            "space adventure with aliens and futuristic battles",
            train_df,
            vectorizer,
            x_train,
        )
    )


if __name__ == "__main__":
    main()
