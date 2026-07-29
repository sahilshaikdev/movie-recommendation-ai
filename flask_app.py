"""
FLASK BACKEND + FRONTEND
Serves the HTML interface and REST API
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from recommender_ai import GenAIRecommender

app = Flask(__name__)
CORS(app)

# Initialize the recommender
recommender = GenAIRecommender()

# Serve the frontend HTML
@app.route('/')
def serve_frontend():
    return send_from_directory('templates', 'index_backend.html')

# API endpoint
@app.route('/api/search', methods=['GET'])
def search():
    query = request.args.get('query', '')
    top_n = int(request.args.get('top_n', 5))
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    results = recommender.search(query, top_n=top_n)
    return jsonify({
        'movies': results['movies'],
        'books': results['books']
    })

if __name__ == '__main__':
    print("🚀 Starting Movie & Book Recommender API + Frontend...")
    print("📍 Open your browser at: http://localhost:5000")
    app.run(debug=True, port=5000)