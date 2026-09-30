# 🎬 Movie Recommendation System

A content-based movie recommendation system built using Python, Pandas, Scikit-learn, and Streamlit.

The system recommends movies similar to a movie selected by the user using **cosine similarity** based on movie features.

## 🚀 Live Demo

👉 [Try the Movie Recommendation System](https://7cy2skiahheyejagm5kpkc.streamlit.app)

## 📌 Features

- Select a movie from the available movie dataset
- Find similar movies using content-based filtering
- Uses cosine similarity to calculate movie similarity
- Displays the top 5 recommended movies
- Interactive Streamlit interface

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Jupyter Notebook

## 📂 Project Structure

```text
movie-recommendation-system/
│
├── app.py
├── movies_dict.pkl
├── requirements.txt
│
├── data/
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
│
└── notebooks/
    └── 01_data_loading.ipynb
