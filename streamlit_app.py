import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie & Book Assistant", page_icon="🎬", layout="wide")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_data():
    movies = pd.read_csv('data/movies.csv')
    books = pd.read_csv('data/books.csv')
    return movies, books

movies_df, books_df = load_model()  # Fix: Use proper function calls
model = load_model()

# Create embeddings
movie_texts = movies_df.apply(lambda x: f"{x['title']} {x['genres']} {x.get('description', '')}", axis=1).tolist()
book_texts = books_df.apply(lambda x: f"{x['title']} by {x['author']} {x['genres']} {x.get('description', '')}", axis=1).tolist()

movie_embeddings = model.encode(movie_texts)
book_embeddings = model.encode(book_texts)

st.title("🎬📚 Movie & Book Assistant")

query = st.text_input("🔍 What are you looking for?")
if query:
    query_embedding = model.encode([query])
    movie_scores = cosine_similarity(query_embedding, movie_embeddings)[0]
    book_scores = cosine_similarity(query_embedding, book_embeddings)[0]
    
    top_movies = np.argsort(movie_scores)[::-1][:5]
    top_books = np.argsort(book_scores)[::-1][:5]
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎬 Movies")
        for i in top_movies:
            st.write(f"**{movies_df.iloc[i]['title']}** - {movie_scores[i]*100:.1f}%")
    with col2:
        st.subheader("📚 Books")
        for i in top_books:
            st.write(f"**{books_df.iloc[i]['title']}** by {books_df.iloc[i]['author']} - {book_scores[i]*100:.1f}%")