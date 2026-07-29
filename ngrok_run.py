"""
MOVIE & BOOK RECOMMENDER – NGROK VERSION
Share with anyone, anywhere!
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from recommender_ai import GenAIRecommender
from pyngrok import ngrok

app = Flask(__name__)
CORS(app)

# Initialize recommender
recommender = GenAIRecommender()

@app.route('/')
def serve_frontend():
    return send_from_directory('templates', 'index_backend.html')

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
    # ✅ REPLACE WITH YOUR ACTUAL TOKEN
    ngrok.set_auth_token("3GoEtqkGaetz707akyuJXUk855D_6GGqMzmVg7e4ZtdYTVVGd")
    
    # Create public URL
    public_url = ngrok.connect(5000)
    print("\n" + "="*60)
    print("🎬📚 MOVIE & BOOK RECOMMENDER – PUBLIC URL")
    print("="*60)
    print(f"🔗 Share this URL with anyone: {public_url}")
    print("="*60)
    
    app.run(debug=True, port=5000)