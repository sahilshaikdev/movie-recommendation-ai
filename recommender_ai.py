"""
GEN AI RECOMMENDATION ENGINE FOR MOVIES & BOOKS
Uses Sentence Transformers for semantic understanding
"""

import pandas as pd
import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class GenAIRecommender:
    def __init__(self):
        self.movies_df = None
        self.books_df = None
        self.embedding_model = None
        self.movie_embeddings = None
        self.book_embeddings = None
        self.movie_texts = None
        self.book_texts = None
        
        self.load_data()
        self.init_model()
        self.create_embeddings()
    
    def load_data(self):
        """Load movies and books from CSV with robust parsing"""
        try:
            # Use Python engine to handle messy commas in descriptions
            self.movies_df = pd.read_csv('data/movies.csv', engine='python', quotechar='"')
            print(f"✅ Loaded {len(self.movies_df)} movies")
        except Exception as e:
            print(f"❌ Error loading movies: {e}")
            self.movies_df = pd.DataFrame()
        
        try:
            self.books_df = pd.read_csv('data/books.csv', engine='python', quotechar='"')
            print(f"✅ Loaded {len(self.books_df)} books")
        except Exception as e:
            print(f"❌ Error loading books: {e}")
            self.books_df = pd.DataFrame()
        
        # Create rich text descriptions for embeddings
        if not self.movies_df.empty:
            if 'description' in self.movies_df.columns:
                self.movie_texts = self.movies_df.apply(
                    lambda row: f"Title: {row['title']}. Genres: {row['genres']}. Description: {row['description']}",
                    axis=1
                ).tolist()
            else:
                self.movie_texts = self.movies_df.apply(
                    lambda row: f"Title: {row['title']}. Genres: {row['genres']}.",
                    axis=1
                ).tolist()
        else:
            self.movie_texts = []
        
        if not self.books_df.empty:
            if 'description' in self.books_df.columns:
                self.book_texts = self.books_df.apply(
                    lambda row: f"Title: {row['title']} by {row['author']}. Genres: {row['genres']}. Description: {row['description']}",
                    axis=1
                ).tolist()
            else:
                self.book_texts = self.books_df.apply(
                    lambda row: f"Title: {row['title']} by {row['author']}. Genres: {row['genres']}.",
                    axis=1
                ).tolist()
        else:
            self.book_texts = []
    
    def init_model(self):
        """Load the embedding model"""
        print("🔄 Loading embedding model... (this may take a minute)")
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Model loaded")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.embedding_model = None
    
    def create_embeddings(self):
        """Generate or load cached embeddings"""
        os.makedirs('models', exist_ok=True)
        
        movie_cache = 'models/movie_embeddings.pkl'
        book_cache = 'models/book_embeddings.pkl'
        
        if os.path.exists(movie_cache) and os.path.exists(book_cache):
            print("🔄 Loading cached embeddings...")
            with open(movie_cache, 'rb') as f:
                self.movie_embeddings = pickle.load(f)
            with open(book_cache, 'rb') as f:
                self.book_embeddings = pickle.load(f)
            print("✅ Embeddings loaded from cache")
            return
        
        if self.embedding_model is None:
            print("⚠️ No model available, skipping embeddings")
            return
        
        print("🔄 Generating embeddings... (this may take a few minutes)")
        if self.movie_texts:
            self.movie_embeddings = self.embedding_model.encode(
                self.movie_texts, show_progress_bar=True
            )
            with open(movie_cache, 'wb') as f:
                pickle.dump(self.movie_embeddings, f)
            print("✅ Movie embeddings saved")
        
        if self.book_texts:
            self.book_embeddings = self.embedding_model.encode(
                self.book_texts, show_progress_bar=True
            )
            with open(book_cache, 'wb') as f:
                pickle.dump(self.book_embeddings, f)
            print("✅ Book embeddings saved")
    
    def search(self, query, top_n=5):
        """
        Search movies and books using natural language query
        Returns: dict with 'movies' and 'books' lists
        """
        if self.embedding_model is None:
            return {'movies': [], 'books': []}
        
        query_embedding = self.embedding_model.encode([query])
        results = {'movies': [], 'books': []}
        
        # Search movies (with image_url fallback)
        if self.movie_embeddings is not None and len(self.movie_embeddings) > 0:
            movie_sims = cosine_similarity(query_embedding, self.movie_embeddings)[0]
            top_indices = np.argsort(movie_sims)[::-1][:top_n]
            for idx in top_indices:
                movie = self.movies_df.iloc[idx]
                # Safely get image_url or empty string
                image_url = movie.get('image_url', '') if 'image_url' in self.movies_df.columns else ''
                results['movies'].append({
                    'title': movie['title'],
                    'genres': movie['genres'],
                    'description': movie.get('description', 'No description available.'),
                    'image_url': image_url,
                    'similarity_score': round(float(movie_sims[idx]) * 100, 2),
                    'type': 'movie'
                })
        
        # Search books (with image_url fallback)
        if self.book_embeddings is not None and len(self.book_embeddings) > 0:
            book_sims = cosine_similarity(query_embedding, self.book_embeddings)[0]
            top_indices = np.argsort(book_sims)[::-1][:top_n]
            for idx in top_indices:
                book = self.books_df.iloc[idx]
                image_url = book.get('image_url', '') if 'image_url' in self.books_df.columns else ''
                results['books'].append({
                    'title': book['title'],
                    'author': book['author'],
                    'genres': book['genres'],
                    'description': book.get('description', 'No description available.'),
                    'image_url': image_url,
                    'similarity_score': round(float(book_sims[idx]) * 100, 2),
                    'type': 'book'
                })
        
        return results
    
    def chat_interface(self):
        """Command-line test interface"""
        print("\n" + "="*60)
        print("📚🎬 MOVIE & BOOK GEN AI RECOMMENDER (CLI)")
        print("="*60)
        print("\n💡 Type a natural language query like:")
        print("  'I want a thrilling mystery book'")
        print("  'Recommend a funny romantic movie'")
        print("  'Sci-fi adventures'")
        print("  'sad drama movie'")
        print("\n❌ Type 'exit' to quit")
        print("="*60)
        
        while True:
            query = input("\n🔍 You: ").strip()
            if query.lower() in ['exit', 'quit']:
                print("👋 Goodbye!")
                break
            if not query:
                continue
            
            print("\n🤖 Searching...")
            results = self.search(query, top_n=3)
            
            if results['movies']:
                print("\n🎬 MOVIES:")
                for i, m in enumerate(results['movies'], 1):
                    print(f"  {i}. {m['title']} (Match: {m['similarity_score']}%)")
                    print(f"     Genres: {m['genres']}")
                    print(f"     Description: {m['description'][:100]}...")
            if results['books']:
                print("\n📚 BOOKS:")
                for i, b in enumerate(results['books'], 1):
                    print(f"  {i}. {b['title']} by {b['author']} (Match: {b['similarity_score']}%)")
                    print(f"     Genres: {b['genres']}")
                    print(f"     Description: {b['description'][:100]}...")
            if not results['movies'] and not results['books']:
                print("😅 No results. Try different keywords.")

if __name__ == "__main__":
    rec = GenAIRecommender()
    rec.chat_interface()