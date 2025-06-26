# 🎬 Movie Recommendation System (Content-Based, TF-IDF + Cosine Similarity)
A content-based movie recommender system built using **Natural Language Processing (NLP)** techniques.  
It suggests similar movies based on the description, genres, keywords, and taglines using **TF-IDF vectorization** and **cosine similarity**.

## 🔗 Live Demo

🔗 [Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/krystallio/movie-recommender-tfidf)

## 🚀 Overview
This **Data Science project** allows users to input a movie title and get 5 similar movie recommendations.  
It processes textual movie metadata and computes similarity scores between all movies using vectorized features.

The project is ideal for demonstrating practical NLP, vectorization, and recommendation system skills.

## 📌 Features
- Real-time recommendations using TF-IDF vectorization.
- Cosine similarity calculated at runtime (no `.pkl` files needed).
- Works with over 20,000+ movies.
- Clean and interactive Streamlit interface.
- Deployed on Hugging Face Spaces — free, fast, and easy to share.

## 🧠 How It Works
1. **Data Preprocessing**:
   - Extracts and combines `overview`, `genres`, `keywords`, and `tagline` into a single textual feature.
   - Removes NaNs and performs basic cleaning.

2. **TF-IDF Vectorization**:
   - Converts text features into numerical vectors using `TfidfVectorizer`.

3. **Cosine Similarity**:
   - Computes cosine similarity between the TF-IDF vectors of all movies.

4. **Recommendation Engine**:
   - Based on the user's movie input, it retrieves the top 5 most similar movies using the similarity matrix.

## 💻 Tech Stack
- Python   
- pandas
- scikit-learn
- TfidfVectorizer
- cosine similarity
- Streamlit  
- Hugging Face Spaces

## 📷 Example
**Input:** `"The Dark Knight"`  
**Output:**
- The Dark Knight Rises  
- Batman Begins  
- Batman  
- Batman Returns  
- Batman Forever

## 📁 Dataset
- Used [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

## 🧪 Getting Started Locally
```bash
git clone https://github.com/yourusername/movie-recommender-tfidf.git
cd movie-recommender-tfidf
pip install -r requirements.txt
streamlit run app.py
```
## 📦 Deployment
This app is deployed on Hugging Face Spaces using Streamlit.
You can try the live version without installing anything:
👉 [Live](https://huggingface.co/spaces/krystallio/movie-recommender-tfidf)

## 🔗 Connect with Me
[LinkedIn](https://www.linkedin.com/in/vlad-kuzmenok-11aabb32a/)
