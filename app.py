import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Merge
movies = movies.merge(credits, on='title')

# Extract genres
def extract_names(text):
    try:
        return [i['name'] for i in ast.literal_eval(text)]
    except:
        return []

movies['genres_list'] = movies['genres'].apply(extract_names)
movies['keywords_list'] = movies['keywords'].apply(extract_names)
movies['cast_list'] = movies['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)][:3] if pd.notna(x) else [])

# Combine into one tag
movies['tags'] = movies['genres_list'] + movies['keywords_list'] + movies['cast_list']
movies['tags'] = movies['tags'].apply(lambda x: ' '.join(x))
movies['tags'] = movies['tags'] + ' ' + movies['overview'].fillna('')

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
matrix = tfidf.fit_transform(movies['tags'])
similarity = cosine_similarity(matrix)

# Recommend function
def recommend(movie):
    if movie not in movies['title'].values:
        return []
    idx = movies[movies['title'] == movie].index[0]
    distances = list(enumerate(similarity[idx]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
    return [movies.iloc[i[0]]['title'] for i in distances]

# Streamlit UI
st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title("🎬 Movie Recommendation System")
st.markdown("Find movies similar to your favourites!")

movie_list = movies['title'].values
selected_movie = st.selectbox("Type or select a movie", movie_list)

if st.button("Recommend"):
    results = recommend(selected_movie)
    if results:
        st.subheader("Top 5 Recommendations:")
        for i, movie in enumerate(results, 1):
            st.write(f"{i}. {movie}")
    else:
        st.error("Movie not found!")