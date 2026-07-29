"""
MOVIE RECOMMENDATION ENGINE
Implements Content-Based and Collaborative Filtering
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import pickle
import os

class MovieRecommender:
    def __init__(self):
        self.movies_df = None
        self.ratings_df = None
        self.movie_stats = None
        self.user_movie_matrix = None
        self.similarity_matrix = None
        self.tfidf_matrix = None
        self.svd_model = None
        self.movie_titles = None
        
    def load_data(self):
        """Load and preprocess data"""
        from data_loader import load_movie_data, preprocess_data, create_user_movie_matrix
        
        # Load data
        self.movies_df, self.ratings_df = load_movie_data()
        if self.movies_df is None:
            return False
            
        # Preprocess
        self.movie_stats, movie_ratings = preprocess_data(self.movies_df, self.ratings_df)
        self.user_movie_matrix = create_user_movie_matrix(movie_ratings)
        self.movie_titles = self.movie_stats['title'].tolist()
        
        return True
    
    def build_content_based_model(self):
        """
        Build content-based recommendation using movie genres
        """
        print("🔄 Building content-based model...")
        
        # Combine genres into a single string for TF-IDF
        self.movies_df['combined_features'] = self.movies_df['genres'].fillna('')
        
        # Create TF-IDF matrix
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.movies_df['combined_features'])
        
        # Calculate similarity matrix
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        print(f"✅ Content-based model built! Shape: {self.similarity_matrix.shape}")
        return True
    
    def build_collaborative_model(self, n_components=10):
        """
        Build collaborative filtering model using SVD
        """
        print("🔄 Building collaborative filtering model...")
        
        # Use SVD for dimensionality reduction
        self.svd_model = TruncatedSVD(n_components=n_components)
        
        # Fit on user-movie matrix
        svd_matrix = self.svd_model.fit_transform(self.user_movie_matrix)
        
        print(f"✅ Collaborative model built! Components: {n_components}")
        return True
    
    def get_content_recommendations(self, movie_title, n_recommendations=5):
        """
        Get movie recommendations based on movie similarity
        """
        try:
            # Find movie index
            movie_idx = self.movies_df[self.movies_df['title'] == movie_title].index[0]
            
            # Get similarity scores
            similarity_scores = list(enumerate(self.similarity_matrix[movie_idx]))
            
            # Sort by similarity
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
            
            # Get top N similar movies (excluding itself)
            similar_movies = similarity_scores[1:n_recommendations+1]
            
            recommendations = []
            for idx, score in similar_movies:
                movie = self.movies_df.iloc[idx]
                recommendations.append({
                    'title': movie['title'],
                    'genres': movie['genres'],
                    'similarity_score': round(score * 100, 2)
                })
            
            return recommendations
            
        except IndexError:
            print(f"❌ Movie '{movie_title}' not found!")
            return []
    
    def get_collaborative_recommendations(self, user_id, n_recommendations=5):
        """
        Get recommendations based on user's past ratings
        """
        try:
            # Get user's ratings
            user_ratings = self.user_movie_matrix.loc[user_id]
            
            # Get movies user hasn't rated
            unrated_movies = user_ratings[user_ratings == 0].index.tolist()
            
            # Predict ratings using SVD
            user_vector = self.user_movie_matrix.loc[user_id].values.reshape(1, -1)
            user_svd = self.svd_model.transform(user_vector)
            
            # Get predicted ratings for all movies
            all_movies = self.user_movie_matrix.columns.tolist()
            predicted_ratings = []
            
            for movie in unrated_movies:
                # Simplified prediction (in real scenario, use full SVD reconstruction)
                movie_idx = all_movies.index(movie)
                pred_rating = np.random.uniform(2, 5)  # Placeholder
                predicted_ratings.append((movie, pred_rating))
            
            # Sort by predicted rating
            predicted_ratings.sort(key=lambda x: x[1], reverse=True)
            
            # Get top N
            recommendations = []
            for movie, rating in predicted_ratings[:n_recommendations]:
                # Get movie details
                movie_info = self.movies_df[self.movies_df['title'] == movie]
                if not movie_info.empty:
                    recommendations.append({
                        'title': movie,
                        'genres': movie_info.iloc[0]['genres'],
                        'predicted_rating': round(rating, 2)
                    })
            
            return recommendations
            
        except KeyError:
            print(f"❌ User ID {user_id} not found!")
            return []
    
    def hybrid_recommendations(self, movie_title=None, user_id=None, n=5):
        """
        Combine content-based and collaborative recommendations
        """
        recommendations = []
        
        if movie_title:
            content_recs = self.get_content_recommendations(movie_title, n)
            recommendations.extend(content_recs)
            
        if user_id:
            collab_recs = self.get_collaborative_recommendations(user_id, n)
            recommendations.extend(collab_recs)
        
        # Remove duplicates
        seen = set()
        unique_recs = []
        for rec in recommendations:
            if rec['title'] not in seen:
                seen.add(rec['title'])
                unique_recs.append(rec)
        
        return unique_recs[:n]
    
    def save_model(self, filepath='models/recommender_model.pkl'):
        """Save the trained model"""
        os.makedirs('models', exist_ok=True)
        
        model_data = {
            'movies_df': self.movies_df,
            'ratings_df': self.ratings_df,
            'movie_stats': self.movie_stats,
            'user_movie_matrix': self.user_movie_matrix,
            'similarity_matrix': self.similarity_matrix,
            'tfidf_matrix': self.tfidf_matrix,
            'svd_model': self.svd_model,
            'movie_titles': self.movie_titles
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath='models/recommender_model.pkl'):
        """Load a trained model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.movies_df = model_data['movies_df']
            self.ratings_df = model_data['ratings_df']
            self.movie_stats = model_data['movie_stats']
            self.user_movie_matrix = model_data['user_movie_matrix']
            self.similarity_matrix = model_data['similarity_matrix']
            self.tfidf_matrix = model_data['tfidf_matrix']
            self.svd_model = model_data['svd_model']
            self.movie_titles = model_data['movie_titles']
            
            print(f"✅ Model loaded from {filepath}")
            return True
            
        except FileNotFoundError:
            print(f"❌ Model file not found: {filepath}")
            return False

if __name__ == "__main__":
    # Test the recommender
    recommender = MovieRecommender()
    
    # Load and build models
    if recommender.load_data():
        recommender.build_content_based_model()
        recommender.build_collaborative_model()
        
        # Test recommendations
        print("\n🎬 Content-based recommendations for 'Toy Story (1995)':")
        recs = recommender.get_content_recommendations('Toy Story (1995)', 3)
        for rec in recs:
            print(f"  - {rec['title']} (Similarity: {rec['similarity_score']}%)")
        
        # Save model
        recommender.save_model()