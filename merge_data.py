"""
MERGE MOVIES AND BOOKS
Run this script to combine your existing data with new data
"""

import pandas as pd
import os

def merge_movies():
    """Merge existing movies with new movies"""
    try:
        # Load existing data
        existing = pd.read_csv('data/movies.csv')
        print(f"✅ Existing movies: {len(existing)}")
        
        # Load new data
        new = pd.read_csv('data/new_movies.csv')
        print(f"✅ New movies: {len(new)}")
        
        # Combine them
        combined = pd.concat([existing, new], ignore_index=True)
        
        # Reset movieId (re-number from 1)
        combined['movieId'] = range(1, len(combined) + 1)
        
        # Save back to movies.csv
        combined.to_csv('data/movies.csv', index=False)
        print(f"✅ Total movies: {len(combined)}")
        
        # Delete the temporary file
        os.remove('data/new_movies.csv')
        print("✅ Deleted new_movies.csv")
        
    except FileNotFoundError:
        print("❌ Error: Make sure movies.csv and new_movies.csv exist in data/ folder")

def merge_books():
    """Merge existing books with new books"""
    try:
        # Load existing data
        existing = pd.read_csv('data/books.csv')
        print(f"✅ Existing books: {len(existing)}")
        
        # Load new data
        new = pd.read_csv('data/new_books.csv')
        print(f"✅ New books: {len(new)}")
        
        # Combine them
        combined = pd.concat([existing, new], ignore_index=True)
        
        # Reset bookId (re-number from 1)
        combined['bookId'] = range(1, len(combined) + 1)
        
        # Save back to books.csv
        combined.to_csv('data/books.csv', index=False)
        print(f"✅ Total books: {len(combined)}")
        
        # Delete the temporary file
        os.remove('data/new_books.csv')
        print("✅ Deleted new_books.csv")
        
    except FileNotFoundError:
        print("❌ Error: Make sure books.csv and new_books.csv exist in data/ folder")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("📂 MERGE DATA SCRIPT")
    print("="*50)
    
    print("\n1️⃣ Merging movies...")
    merge_movies()
    
    print("\n2️⃣ Merging books...")
    merge_books()
    
    print("\n" + "="*50)
    print("✅ DONE! Your data has been updated.")
    print("👉 Now delete models/*.pkl and restart Flask")
    print("="*50)