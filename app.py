import streamlit as st
import pickle
import gdown
import os
from dotenv import load_dotenv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Movie Recommendation System")

load_dotenv()

@st.cache_resource
def load_movies():
    return pickle.load(open('movies.pkl','rb'))

@st.cache_resource
def load_similarity():
    url = os.getenv("SIMILARITY_URL")
    if not os.path.exists("similarity.pkl"):
        gdown.download(url, "similarity.pkl", quiet=False)
    return pickle.load(open('similarity.pkl','rb'))

if "data_loaded" not in st.session_state:
    try:
        movies = load_movies()
        similarity = load_similarity()
        st.session_state["data_loaded"]=True
        st.toast("✅ Data loaded successfully")
    except Exception as e:
        st.toast(f"❌ Failed to load: {e}")
        st.stop()
else:
    movies = load_movies()
    similarity = load_similarity()

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