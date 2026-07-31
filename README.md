# 🎬 CineFans - Movie Recommendation System

A content-based movie recommendation web app built with Python, Scikit-learn, and Streamlit.

🔗 **[Live Demo](https://cinefans-movie-recommender.streamlit.app/)**

---

## 📌 About The Project

This app recommends 5 similar movies based on your selection using Natural Language Processing and Machine Learning techniques applied on the TMDB 5000 Movies dataset.

---

## ⚙️ How It Works

- Combines movie **genres**, **cast**, and **keywords** into unified tags
- Applies **TF-IDF Vectorization** to convert text into numerical vectors
- Uses **Cosine Similarity** to find the most similar movies
- Displays top 5 recommendations instantly via a **Streamlit** web interface

---

## 🛠️ Built With

- Python
- Pandas & NumPy
- Scikit-learn (TF-IDF, Cosine Similarity)
- Streamlit
- TMDB 5000 Movies Dataset

---

## 🚀 Run Locally

```bash
git clone https://github.com/SamisthaKesarwani/Movie-Recommender.git
cd Movie-Recommender
pip install -r requirements.txt
streamlit run app.py
```

---

## 📊 Dataset

[TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) — 5000+ movies with genres, cast, crew and keywords.

---

## 👩‍💻 Author

**Samistha Kesarwani**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/samistha-kesarwani-55393625b/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/SamisthaKesarwani)
