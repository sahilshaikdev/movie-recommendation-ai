"""
DATA LOADER MODULE
Handles loading and preprocessing of movie data
"""

import pandas as pd
import os

def load_movie_data():
    """
    Load movie and rating data from CSV files
    Returns: movies_df, ratings_df
    """
    try:
        # Load movies
        movies_df = pd.read_csv('data/movies.csv')
        print(f"✅ Loaded {len(movies_df)} movies")
        
        # Load ratings
        ratings_df = pd.read_csv('data/ratings.csv')
        print(f"✅ Loaded {len(ratings_df)} ratings")
        
        return movies_df, ratings_df
    
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please make sure data files exist in 'data/' folder")
        return None, None

def preprocess_data(movies_df, ratings_df):
    """
    Clean and prepare data for recommendation
    """
    # Merge movies with ratings
    movie_ratings = pd.merge(ratings_df, movies_df, on='movieId')
    
    # Calculate average rating per movie
    avg_ratings = movie_ratings.groupby('title')['rating'].mean().reset_index()
    avg_ratings.columns = ['title', 'avg_rating']
    
    # Count number of ratings per movie
    rating_counts = movie_ratings.groupby('title')['rating'].count().reset_index()
    rating_counts.columns = ['title', 'rating_count']
    
    # Merge both
    movie_stats = pd.merge(avg_ratings, rating_counts, on='title')
    
    print(f"✅ Preprocessed {len(movie_stats)} movies with ratings")
    return movie_stats, movie_ratings

def create_user_movie_matrix(movie_ratings):
    """
    Create a user-movie rating matrix for collaborative filtering
    """
    # Pivot table: users as rows, movies as columns
    user_movie_matrix = movie_ratings.pivot_table(
        index='userId', 
        columns='title', 
        values='rating'
    )
    
    # Fill NaN with 0
    user_movie_matrix.fillna(0, inplace=True)
    
    print(f"✅ Created user-movie matrix: {user_movie_matrix.shape}")
    return user_movie_matrix

if __name__ == "__main__":
    # Test the data loader
    movies, ratings = load_movie_data()
    if movies is not None and ratings is not None:
        stats, merged = preprocess_data(movies, ratings)
        matrix = create_user_movie_matrix(merged)
        print("\n📊 Data loaded successfully!")