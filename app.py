import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load movie data
movies_dict = pickle.load(open("movies_dict.pkl", "rb"))
movies = pd.DataFrame(movies_dict)


# Create movie feature vectors
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(movies["tags"].fillna(""))


def recommend(movie):
    movie_index = movies[movies["title"] == movie].index[0]

    # Calculate similarity only for the selected movie
    similarity_scores = cosine_similarity(
        vectors[movie_index],
        vectors
    ).flatten()

    # Get the 5 most similar movies
    movie_indices = sorted(
        list(enumerate(similarity_scores)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []

    for i in movie_indices:
        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies


# Streamlit UI
st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox(
    "Select a movie you like:",
    movies["title"].values
)


if st.button("Recommend"):

    recommendations = recommend(selected_movie)

    st.subheader("Recommended Movies")

    for movie in recommendations:
        st.write("🎥", movie)