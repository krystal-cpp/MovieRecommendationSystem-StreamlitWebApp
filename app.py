import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pickle.load(open('movies.pkl','rb'))
tfidf = TfidfVectorizer(max_features=10000, stop_words='english')
vectors = tfidf.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)

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

st.title('Movie Recommendation System')
selected_movie = st.text_input('Enter a movie title.', '')

if(st.button('Recommend')):
    if selected_movie.strip() == '':
        st.warning("Please enter a movie title.")
    else:
        recommended_movies = recommend(selected_movie)
        for movie in recommended_movies:
            st.write("🎬", movie)