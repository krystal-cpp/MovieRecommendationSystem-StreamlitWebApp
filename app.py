import streamlit as st
import pickle
import pandas as pd

movies = pickle.load(open('movies.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))

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