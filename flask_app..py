"""
MAIN APPLICATION - MOVIE RECOMMENDATION AI
Command Line Interface (CLI) version
"""

import sys
import os
from recommender import MovieRecommender
from utils import display_recommendations, get_user_input, display_movie_info

class MovieRecommendationApp:
    def __init__(self):
        self.recommender = MovieRecommender()
        self.is_ready = False
        
    def initialize(self):
        """Initialize the application"""
        print("\n" + "="*60)
        print("🎬 MOVIE RECOMMENDATION AI SYSTEM")
        print("="*60)
        print("\n📂 Loading data and building models...")
        
        # Load data and build models
        if self.recommender.load_data():
            self.recommender.build_content_based_model()
            self.recommender.build_collaborative_model()
            self.recommender.save_model()
            self.is_ready = True
            print("\n✅ System ready! 🎉")
            return True
        else:
            print("\n❌ Failed to initialize system!")
            return False
    
    def show_menu(self):
        """Display main menu"""
        print("\n" + "="*50)
        print("📋 MAIN MENU")
        print("="*50)
        print("1. Get Movie Recommendations (Content-Based)")
        print("2. Get Recommendations for User (Collaborative)")
        print("3. Hybrid Recommendations (Both Methods)")
        print("4. Search for a Movie")
        print("5. View Top Rated Movies")
        print("6. View Popular Movies")
        print("7. Show All Movies")
        print("8. Exit")
        print("="*50)
    
    def content_based_recommendations(self):
        """Get content-based recommendations"""
        print("\n🎬 Content-Based Recommendations")
        print("-" * 30)
        
        # Show available movies
        print("\n📋 Available Movies:")
        for i, title in enumerate(self.recommender.movie_titles[:10], 1):
            print(f"  {i}. {title}")
        print("  ... (and more)")
        
        # Get user input
        movie_title = get_user_input("\nEnter a movie title: ")
        
        # Get recommendations
        recs = self.recommender.get_content_recommendations(movie_title, 5)
        
        if recs:
            display_recommendations(recs, f"🎯 Recommendations based on '{movie_title}'")
        else:
            print("❌ Movie not found or no recommendations available!")
    
    def collaborative_recommendations(self):
        """Get collaborative recommendations for a user"""
        print("\n👤 Collaborative Filtering Recommendations")
        print("-" * 30)
        
        # Show available users
        users = self.recommender.user_movie_matrix.index.tolist()
        print(f"\n📋 Available User IDs: {users[:5]}...")
        
        # Get user ID
        user_id_input = get_user_input("\nEnter User ID: ")
        try:
            user_id = int(user_id_input)
        except ValueError:
            print("❌ Please enter a valid number!")
            return
        
        # Get recommendations
        recs = self.recommender.get_collaborative_recommendations(user_id, 5)
        
        if recs:
            display_recommendations(recs, f"🎯 Recommendations for User #{user_id}")
        else:
            print("❌ User not found or no recommendations available!")
    
    def hybrid_recommendations(self):
        """Get hybrid recommendations"""
        print("\n🔀 Hybrid Recommendations (Best of Both)")
        print("-" * 30)
        
        # Get movie input
        movie_title = get_user_input("Enter a movie title (or press Enter to skip): ", options=[])
        
        # Get user input
        user_id_input = get_user_input("Enter User ID (or press Enter to skip): ", options=[])
        user_id = int(user_id_input) if user_id_input else None
        
        # Get recommendations
        recs = self.recommender.hybrid_recommendations(
            movie_title if movie_title else None,
            user_id,
            5
        )
        
        if recs:
            display_recommendations(recs, "🎯 Hybrid Recommendations")
        else:
            print("❌ No recommendations available! Please provide movie title or user ID.")
    
    def search_movie(self):
        """Search for a specific movie"""
        print("\n🔍 Search Movies")
        print("-" * 30)
        
        search_term = get_user_input("Enter movie name to search: ")
        
        # Search in movies dataframe
        results = self.recommender.movies_df[
            self.recommender.movies_df['title'].str.contains(search_term, case=False, na=False)
        ]
        
        if not results.empty:
            print(f"\n📋 Found {len(results)} movies:")
            for _, movie in results.iterrows():
                movie_info = {
                    'title': movie['title'],
                    'genres': movie['genres']
                }
                # Add rating info if available
                rating_info = self.recommender.movie_stats[
                    self.recommender.movie_stats['title'] == movie['title']
                ]
                if not rating_info.empty:
                    movie_info['avg_rating'] = rating_info.iloc[0]['avg_rating']
                    movie_info['rating_count'] = rating_info.iloc[0]['rating_count']
                
                display_movie_info(movie_info)
        else:
            print("❌ No movies found matching your search!")
    
    def view_top_rated(self):
        """View top rated movies"""
        from utils import get_top_movies
        
        print("\n⭐ Top Rated Movies")
        print("-" * 30)
        
        top_movies = get_top_movies(self.recommender.movie_stats, 10)
        
        for i, (_, movie) in enumerate(top_movies.iterrows(), 1):
            print(f"\n{i}. {movie['title']}")
            print(f"   Rating: {movie['avg_rating']:.2f}/5.0")
            print(f"   Votes: {int(movie['rating_count'])}")
    
    def view_popular_movies(self):
        """View most popular movies"""
        from utils import get_popular_movies
        
        print("\n🔥 Popular Movies (Most Rated)")
        print("-" * 30)
        
        popular = get_popular_movies(self.recommender.movie_stats, 10)
        
        for i, (_, movie) in enumerate(popular.iterrows(), 1):
            print(f"\n{i}. {movie['title']}")
            print(f"   Votes: {int(movie['rating_count'])}")
            print(f"   Rating: {movie['avg_rating']:.2f}/5.0")
    
    def show_all_movies(self):
        """Show all available movies"""
        print("\n📋 All Available Movies")
        print("-" * 30)
        
        for i, title in enumerate(self.recommender.movie_titles, 1):
            print(f"{i}. {title}")
    
    def run(self):
        """Run the main application loop"""
        if not self.initialize():
            return
        
        while True:
            self.show_menu()
            choice = get_user_input("\nSelect an option (1-8): ", 
                                   options=['1', '2', '3', '4', '5', '6', '7', '8'])
            
            if choice == '1':
                self.content_based_recommendations()
            elif choice == '2':
                self.collaborative_recommendations()
            elif choice == '3':
                self.hybrid_recommendations()
            elif choice == '4':
                self.search_movie()
            elif choice == '5':
                self.view_top_rated()
            elif choice == '6':
                self.view_popular_movies()
            elif choice == '7':
                self.show_all_movies()
            elif choice == '8':
                print("\n👋 Thank you for using Movie Recommendation AI!")
                print("🎬 Happy watching!")
                break

if __name__ == "__main__":
    app = MovieRecommendationApp()
    app.run()