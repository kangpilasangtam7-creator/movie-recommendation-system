import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


# --------------------------------------------------
# MOVIE IMAGE
# --------------------------------------------------

MOVIE_IMAGE = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSlwWhBylGRogOQiDGU1RfF_CMkenZI1GkWkb4NDREu9Q&s=10"


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #0f0f0f;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: white;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #bdbdbd;
}

.recommendation {
    background-color: #1c1c1c;
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 12px;
    border: 1px solid #333333;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MOVIE DATA
# --------------------------------------------------

@st.cache_data
def load_movies():

    with open("movies_dict.pkl", "rb") as file:
        movies_dict = pickle.load(file)

    return pd.DataFrame(movies_dict)


movies = load_movies()


# --------------------------------------------------
# CREATE FEATURE VECTORS
# --------------------------------------------------

@st.cache_resource
def create_vectors(movie_tags):

    cv = CountVectorizer(
        max_features=5000,
        stop_words="english"
    )

    vectors = cv.fit_transform(movie_tags)

    return vectors


vectors = create_vectors(
    movies["tags"].fillna("")
)


# --------------------------------------------------
# RECOMMENDATION FUNCTION
# --------------------------------------------------

def recommend(movie):

    movie_index = movies[movies["title"] == movie].index[0]

    similarity_scores = cosine_similarity(
        vectors[movie_index],
        vectors
    ).flatten()

    movie_indices = sorted(
        list(enumerate(similarity_scores)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []

    for i in movie_indices:
        recommended_movies.append(
            movies.iloc[i[0]]["title"]
        )

    return recommended_movies


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.image(
    MOVIE_IMAGE,
    use_container_width=True
)

st.markdown(
    '<p class="main-title">🎬 Movie Recommendation System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Discover movies similar to the ones you already love</p>',
    unsafe_allow_html=True
)

st.divider()


# --------------------------------------------------
# MOVIE SELECTION
# --------------------------------------------------

st.subheader("🎥 Choose a movie")

selected_movie = st.selectbox(
    "Select a movie you like:",
    movies["title"].values
)


# --------------------------------------------------
# RECOMMEND BUTTON
# --------------------------------------------------

if st.button(
    "🎬 Recommend Movies",
    use_container_width=True
):

    recommendations = recommend(selected_movie)

    st.divider()

    st.subheader(
        f"✨ Movies similar to {selected_movie}"
    )

    for number, movie in enumerate(
        recommendations,
        start=1
    ):

        st.markdown(
            f"""
            <div class="recommendation">
                <h3>#{number} &nbsp; 🎬 {movie}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )


# --------------------------------------------------
# HOW IT WORKS
# --------------------------------------------------

st.divider()

st.subheader("⚙️ How it works")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 01")
    st.write(
        "Movie information is converted into feature vectors."
    )

with col2:
    st.markdown("###  02")
    st.write(
        "Cosine similarity compares movies based on their features."
    )

with col3:
    st.markdown("###  03")
    st.write(
        "The five most similar movies are recommended."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Built with Python • Streamlit • Scikit-learn"
)