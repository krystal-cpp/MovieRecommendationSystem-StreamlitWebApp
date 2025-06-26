import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Movie Recommendation System")

@st.cache_resource
def load_data():
    movies = pickle.load(open('movies.pkl','rb'))
    movies = movies[movies['tags'].notna()].reset_index(drop=True)
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies['tags'])
    return movies, tfidf_matrix

@st.cache_data
def compute_similarity(_tfidf_matrix):
    return cosine_similarity(_tfidf_matrix)

if "data_loaded" not in st.session_state:
    try:
        movies, tfidf_matrix = load_data()
        st.toast("✅ Data loaded successfully.")
        similarity = compute_similarity(tfidf_matrix)
        st.session_state["data_loaded"]=True
        st.toast("✅ Similarity computed successfully.")
    except Exception as e:
        st.toast(f"❌ Failed to load: {e}")
        st.stop()
else:
    movies, tfidf_matrix = load_data()
    similarity = compute_similarity(tfidf_matrix)

def recommend(movie):
    movie = movie.lower()
    if movie not in movies['original_title'].str.lower().values:
        return ['Movie not found.']

    index = movies[movies['original_title'].str.lower() == movie].index[0]
    distances = list(enumerate(similarity[index]))
    movies_list = sorted(distances, key=lambda x: x[1], reverse=True)[1:20]
    recommendations = movies.iloc[[i[0] for i in movies_list]].copy()
    recommendations = recommendations.sort_values(by='vote_count', ascending=False)
    return recommendations['original_title'].head(5).tolist()

selected_movie = st.text_input('Enter a movie title.', '')

if(st.button('Recommend')):
    if selected_movie.strip() == '':
        st.warning("Please enter a movie title.")
    else:
        recommended_movies = recommend(selected_movie)
        for movie in recommended_movies:
            st.write("🎬", movie)