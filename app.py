import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(
    page_title="CineFans - Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0f0f0f; }
    .stApp { background-color: #141414; }
    h1 { color: #E50914 !important; font-size: 3rem !important; }
    h3 { color: #ffffff !important; }
    .movie-card {
        background-color: #1f1f1f;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #E50914;
        color: white;
    }
    .subtitle { color: #aaaaaa; font-size: 1.1rem; }
    .stButton>button {
        background-color: #E50914;
        color: white;
        border: none;
        padding: 10px 30px;
        font-size: 1rem;
        border-radius: 5px;
        width: 100%;
    }
    .stButton>button:hover { background-color: #b20710; }
    .stSelectbox label { color: #aaaaaa !important; }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')
    movies = movies.merge(credits, on='title')

    def extract_names(text):
        try:
            return [i['name'] for i in ast.literal_eval(text)]
        except:
            return []

    movies['genres_list'] = movies['genres'].apply(extract_names)
    movies['keywords_list'] = movies['keywords'].apply(extract_names)
    movies['cast_list'] = movies['cast'].apply(
        lambda x: [i['name'] for i in ast.literal_eval(x)][:3] if pd.notna(x) else []
    )
    movies['tags'] = movies['genres_list'] + movies['keywords_list'] + movies['cast_list']
    movies['tags'] = movies['tags'].apply(lambda x: ' '.join(x))
    movies['tags'] = movies['tags'] + ' ' + movies['overview'].fillna('')
    return movies

movies = load_data()

# Build similarity matrix
@st.cache_data
def build_similarity():
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    matrix = tfidf.fit_transform(movies['tags'])
    return cosine_similarity(matrix)

similarity = build_similarity()

# Recommend function
def recommend(movie):
    idx = movies[movies['title'] == movie].index[0]
    distances = list(enumerate(similarity[idx]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
    return [movies.iloc[i[0]][['title', 'overview', 'vote_average']].to_dict() for i in distances]

# UI
st.markdown("<h1>🎬 CineFans</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Discover movies you'll love </p>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns([3, 1])
with col1:
    selected_movie = st.selectbox("🔍 Search for a movie", movies['title'].values)
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    recommend_btn = st.button("Get Recommendations")

if recommend_btn:
    st.markdown("<h3>Top 5 Recommendations for: " + selected_movie + "</h3>", unsafe_allow_html=True)
    results = recommend(selected_movie)
    for i, movie in enumerate(results, 1):
        rating = movie['vote_average']
        stars = "⭐" * round(rating / 2)
        overview = movie['overview'][:200] + "..." if len(str(movie['overview'])) > 200 else movie['overview']
        st.markdown(f"""
        <div class='movie-card'>
            <h4 style='color:#E50914; margin:0'>#{i} {movie['title']}</h4>
            <p style='color:#f5c518; margin:5px 0'>{stars} {rating}/10</p>
            <p style='color:#cccccc; font-size:0.9rem'>{overview}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
#st.markdown("<p style='color:#555; text-align:center'>Built with Python • Scikit-learn • Streamlit | Dataset: TMDB 5000 Movies</p>", unsafe_allow_html=True)