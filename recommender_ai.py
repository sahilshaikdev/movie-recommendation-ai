"""
GEN AI RECOMMENDATION ENGINE FOR MOVIES & BOOKS
Uses TF-IDF + Cosine Similarity (lightweight, works on free hosting)
Replaced: sentence_transformers -> sklearn TfidfVectorizer
"""

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class GenAIRecommender:
    def __init__(self):
        self.movies_df = None
        self.books_df = None
        self.movie_texts = None
        self.book_texts = None
        self.movie_tfidf_matrix = None
        self.book_tfidf_matrix = None
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))

        self.load_data()
        self.create_embeddings()

    def load_data(self):
        """Load movies and books from CSV"""
        try:
            self.movies_df = pd.read_csv(
                os.path.join(os.path.dirname(__file__), 'data/movies.csv'),
                engine='python', quotechar='"'
            )
            print(f"✅ Loaded {len(self.movies_df)} movies")
        except Exception as e:
            print(f"❌ Error loading movies: {e}")
            self.movies_df = pd.DataFrame()

        try:
            self.books_df = pd.read_csv(
                os.path.join(os.path.dirname(__file__), 'data/books.csv'),
                engine='python', quotechar='"'
            )
            print(f"✅ Loaded {len(self.books_df)} books")
        except Exception as e:
            print(f"❌ Error loading books: {e}")
            self.books_df = pd.DataFrame()

        # Build rich text for TF-IDF
        if not self.movies_df.empty:
            if 'description' in self.movies_df.columns:
                self.movie_texts = self.movies_df.apply(
                    lambda row: f"{row['title']} {row['genres']} {row.get('description', '')}",
                    axis=1
                ).tolist()
            else:
                self.movie_texts = self.movies_df.apply(
                    lambda row: f"{row['title']} {row['genres']}",
                    axis=1
                ).tolist()
        else:
            self.movie_texts = []

        if not self.books_df.empty:
            if 'description' in self.books_df.columns:
                self.book_texts = self.books_df.apply(
                    lambda row: f"{row['title']} {row['author']} {row['genres']} {row.get('description', '')}",
                    axis=1
                ).tolist()
            else:
                self.book_texts = self.books_df.apply(
                    lambda row: f"{row['title']} {row['author']} {row['genres']}",
                    axis=1
                ).tolist()
        else:
            self.book_texts = []

    def create_embeddings(self):
        """Build TF-IDF matrix from all texts"""
        all_texts = self.movie_texts + self.book_texts

        if not all_texts:
            print("⚠️ No texts found to build TF-IDF matrix")
            return

        print("🔄 Building TF-IDF matrix...")
        self.vectorizer.fit(all_texts)

        if self.movie_texts:
            self.movie_tfidf_matrix = self.vectorizer.transform(self.movie_texts)
            print(f"✅ Movie TF-IDF matrix: {self.movie_tfidf_matrix.shape}")

        if self.book_texts:
            self.book_tfidf_matrix = self.vectorizer.transform(self.book_texts)
            print(f"✅ Book TF-IDF matrix: {self.book_tfidf_matrix.shape}")

    def search(self, query, top_n=5):
        """
        Search movies and books using natural language query
        Returns: dict with 'movies' and 'books' lists
        """
        results = {'movies': [], 'books': []}

        if not query.strip():
            return results

        # Transform query using same vectorizer
        query_vec = self.vectorizer.transform([query])

        # Search movies
        if self.movie_tfidf_matrix is not None and not self.movies_df.empty:
            movie_sims = cosine_similarity(query_vec, self.movie_tfidf_matrix)[0]
            top_indices = np.argsort(movie_sims)[::-1][:top_n]

            for idx in top_indices:
                if movie_sims[idx] == 0:
                    continue
                movie = self.movies_df.iloc[idx]
                image_url = movie.get('image_url', '') if 'image_url' in self.movies_df.columns else ''
                results['movies'].append({
                    'title': str(movie['title']),
                    'genres': str(movie['genres']),
                    'description': str(movie.get('description', 'No description available.')),
                    'image_url': str(image_url),
                    'similarity_score': round(float(movie_sims[idx]) * 100, 2),
                    'type': 'movie'
                })

        # Search books
        if self.book_tfidf_matrix is not None and not self.books_df.empty:
            book_sims = cosine_similarity(query_vec, self.book_tfidf_matrix)[0]
            top_indices = np.argsort(book_sims)[::-1][:top_n]

            for idx in top_indices:
                if book_sims[idx] == 0:
                    continue
                book = self.books_df.iloc[idx]
                image_url = book.get('image_url', '') if 'image_url' in self.books_df.columns else ''
                results['books'].append({
                    'title': str(book['title']),
                    'author': str(book['author']),
                    'genres': str(book['genres']),
                    'description': str(book.get('description', 'No description available.')),
                    'image_url': str(image_url),
                    'similarity_score': round(float(book_sims[idx]) * 100, 2),
                    'type': 'book'
                })

        return results


if __name__ == "__main__":
    rec = GenAIRecommender()
    print("\n✅ Recommender ready!")
    results = rec.search("romantic comedy", top_n=3)
    print("Movies:", [m['title'] for m in results['movies']])
    print("Books:", [b['title'] for b in results['books']])