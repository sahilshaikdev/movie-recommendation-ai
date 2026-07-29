"""
🎬📚 MOVIE & BOOK RECOMMENDER – LIGHTWEIGHT VERSION
Works immediately on Streamlit Cloud (no heavy ML packages)
"""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Movie & Book Assistant", page_icon="🎬", layout="wide")

# Load data
@st.cache_data
def load_data():
    movies = pd.read_csv('data/movies.csv')
    books = pd.read_csv('data/books.csv')
    return movies, books

movies_df, books_df = load_data()

# Custom CSS for nicer UI
st.markdown("""
<style>
    .main-title { font-size: 3rem; font-weight: 800; text-align: center; 
                  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                  -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .movie-card { background: #f0f4ff; border-radius: 12px; padding: 1rem; margin: 0.5rem 0; border-left: 5px solid #667eea; }
    .book-card { background: #fff5f5; border-radius: 12px; padding: 1rem; margin: 0.5rem 0; border-left: 5px solid #e53e3e; }
    .badge { display: inline-block; padding: 0.2rem 0.8rem; border-radius: 20px; font-size: 0.75rem; color: white; }
    .movie-badge { background: #667eea; }
    .book-badge { background: #e53e3e; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎬📚 Movie & Book Assistant</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666;">Your AI-powered recommendation assistant for movies and books</p>', unsafe_allow_html=True)

# Search
query = st.text_input("🔍 What are you looking for?", placeholder="e.g., romantic comedy, sci-fi adventure, thriller")

if query:
    st.subheader(f"📊 Results for: '{query}'")
    
    # Search Movies
    movie_results = movies_df[
        movies_df['title'].str.contains(query, case=False) |
        movies_df['genres'].str.contains(query, case=False) |
        movies_df['description'].str.contains(query, case=False, na=False)
    ]
    
    # Search Books
    book_results = books_df[
        books_df['title'].str.contains(query, case=False) |
        books_df['genres'].str.contains(query, case=False) |
        books_df['description'].str.contains(query, case=False, na=False)
    ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎬 Movies")
        if len(movie_results) > 0:
            for _, row in movie_results.head(5).iterrows():
                st.markdown(f"""
                <div class="movie-card">
                    <h4>🎥 {row['title']}</h4>
                    <span class="badge movie-badge">Movie</span>
                    <br><br>
                    <strong>Genres:</strong> {row['genres']}<br>
                    <strong>Description:</strong> {row.get('description', 'No description available.')}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No movies found.")
    
    with col2:
        st.markdown("### 📚 Books")
        if len(book_results) > 0:
            for _, row in book_results.head(5).iterrows():
                st.markdown(f"""
                <div class="book-card">
                    <h4>📖 {row['title']}</h4>
                    <p><em>by {row.get('author', 'Unknown')}</em></p>
                    <span class="badge book-badge">Book</span>
                    <br><br>
                    <strong>Genres:</strong> {row['genres']}<br>
                    <strong>Description:</strong> {row.get('description', 'No description available.')}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No books found.")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app helps you discover movies and books using natural language.
    
    **Try searches like:**
    - `romantic comedy`
    - `sci-fi adventure`
    - `thrilling mystery`
    - `sad drama`
    
    **Dataset:**
    - 100+ Movies
    - 100+ Books
    
    Made with ❤️ using Python & Streamlit
    """)

# Footer
st.divider()
st.caption("🔍 Built with Python & Streamlit • Data: 100+ movies & 100+ books")