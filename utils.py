"""
UTILITY FUNCTIONS
Helper functions for the recommendation system
"""

import pandas as pd
import numpy as np
from datetime import datetime

def display_movie_info(movie):
    """Display movie information in a nice format"""
    print(f"\n🎬 {movie['title']}")
    print(f"   Genres: {movie['genres']}")
    if 'avg_rating' in movie:
        print(f"   Rating: {movie['avg_rating']:.2f}/5.0")
    if 'rating_count' in movie:
        print(f"   Votes: {movie['rating_count']}")
    if 'similarity_score' in movie:
        print(f"   Match: {movie['similarity_score']}%")
    if 'predicted_rating' in movie:
        print(f"   Predicted Rating: {movie['predicted_rating']}/5.0")

def display_recommendations(recommendations, title="🎯 RECOMMENDATIONS"):
    """Display a list of recommendations"""
    if not recommendations:
        print("❌ No recommendations found!")
        return
    
    print(f"\n{'='*50}")
    print(f"{title}")
    print('='*50)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['title']}")
        print(f"   Genres: {rec.get('genres', 'N/A')}")
        if 'similarity_score' in rec:
            print(f"   Match: {rec['similarity_score']}%")
        if 'predicted_rating' in rec:
            print(f"   Predicted: {rec['predicted_rating']}/5.0")

def get_user_input(prompt, options=None):
    """Get user input with validation"""
    while True:
        user_input = input(prompt).strip()
        
        if options is not None:
            if user_input.lower() in [opt.lower() for opt in options]:
                return user_input
            print(f"Please enter one of: {', '.join(options)}")
        else:
            if user_input:
                return user_input
            print("Input cannot be empty!")

def calculate_rating_stats(ratings_df):
    """Calculate rating statistics"""
    stats = {
        'total_ratings': len(ratings_df),
        'unique_users': ratings_df['userId'].nunique(),
        'unique_movies': ratings_df['movieId'].nunique(),
        'avg_rating': ratings_df['rating'].mean(),
        'min_rating': ratings_df['rating'].min(),
        'max_rating': ratings_df['rating'].max()
    }
    return stats

def get_top_movies(movie_stats, n=10):
    """Get top N movies by rating"""
    return movie_stats.nlargest(n, 'avg_rating')

def get_popular_movies(movie_stats, n=10):
    """Get N most popular movies by number of ratings"""
    return movie_stats.nlargest(n, 'rating_count')

def filter_by_genre(movies_df, genre):
    """Filter movies by genre"""
    return movies_df[movies_df['genres'].str.contains(genre, case=False, na=False)]

if __name__ == "__main__":
    # Test utilities
    print("✅ Utility functions loaded successfully!")