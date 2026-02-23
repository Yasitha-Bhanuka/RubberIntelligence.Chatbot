"""
RubberIntelligence Chatbot — Flask API Server
Fully offline RAG-based chatbot for rubber domain knowledge.
"""
import logging
import os
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from services.chat_service import ChatService

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask
app = Flask(__name__)
CORS(app)  # Allow React Native to connect

# Initialize Chat Service (loads models + knowledge on startup)
logger.info("Initializing RubberBot...")
chat_service = ChatService()
logger.info("RubberBot ready! 🌿")


# ─── Endpoints ────────────────────────────────────────────────

@app.route('/api/chat', methods=['POST'])
def chat():
    """Process a chat message and return AI response."""
    data = request.get_json()

    if not data or 'message' not in data:
        return jsonify({'error': 'Message is required'}), 400

    message = data['message'].strip()
    session_id = data.get('sessionId', str(uuid.uuid4()))

    if not message:
        return jsonify({'error': 'Message cannot be empty'}), 400

    logger.info(f"[{session_id[:8]}] User: {message[:80]}")

    try:
        response = chat_service.process_message(message, session_id)
        logger.info(
            f"[{session_id[:8]}] Bot: confidence={response['confidence']:.3f} "
            f"level={response['confidence_level']} category={response.get('category', '')}"
        )
        return jsonify(response)
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({
            'reply': 'Sorry, I encountered an error processing your question. Please try again.',
            'confidence': 0,
            'confidence_level': 'low',
            'sources': [],
            'suggested_topics': []
        }), 500


@app.route('/api/chat/welcome', methods=['GET'])
def welcome():
    """Get the welcome message with available categories."""
    return jsonify(chat_service.get_welcome_message())


@app.route('/api/chat/topics', methods=['GET'])
def topics():
    """Get all available topics grouped by category."""
    return jsonify(chat_service.get_topics())


@app.route('/api/chat/health', methods=['GET'])
def health():
    """Health check endpoint."""
    kb_count = len(chat_service.embedding_service.knowledge)
    categories = chat_service.embedding_service.get_categories()
    return jsonify({
        'status': 'healthy',
        'service': 'RubberIntelligence Chatbot',
        'knowledge_entries': kb_count,
        'categories': categories,
        'embedding_backend': 'sentence-transformers + FAISS'
            if hasattr(chat_service.embedding_service, 'model')
            else 'TF-IDF (fallback)'
    })


# ─── Run ──────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5008))
    logger.info(f"Starting RubberBot on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)
