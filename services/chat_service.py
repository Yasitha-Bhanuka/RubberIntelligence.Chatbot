import logging
from .embedding_service import EmbeddingService
from .db_service import DbService

logger = logging.getLogger(__name__)

# Confidence thresholds
HIGH_CONFIDENCE = 0.65
MEDIUM_CONFIDENCE = 0.30

# Category emoji mapping
CATEGORY_ICONS = {
    "Diseases": "🦠",
    "Pests": "🐛",
    "Weeds": "🌿",
    "Latex Quality": "🧪",
    "Cultivation": "🌱",
    "Climate": "🌤️",
    "Economics": "💰",
    "Processing": "🏭",
    "General": "📘",
}


class ChatService:
    """
    RAG-based chat service for rubber domain knowledge.
    Retrieves relevant knowledge, scores confidence, and composes responses.
    """

    def __init__(self, knowledge_path: str = None):
        self.embedding_service = EmbeddingService(knowledge_path)
        self.db_service = DbService()
        self.conversation_contexts = {}  # sessionId -> last category

    def process_message(self, message: str, session_id: str = None, latitude: float = None, longitude: float = None) -> dict:
        """
        Process a user message and return a structured response.

        Returns: {
            reply: str,
            confidence: float,
            confidence_level: 'high' | 'medium' | 'low',
            sources: [{ category, question }],
            suggested_topics: [str]
        }
        """
        # Search knowledge base
        results = self.embedding_service.search(message, top_k=3)

        if not results:
            return self._fallback_response()

        top_entry, top_score = results[0]

        # Track conversation context
        if session_id:
            self.conversation_contexts[session_id] = top_entry.get('category', '')

        # High confidence — direct expert answer
        if top_score >= HIGH_CONFIDENCE:
            response = self._high_confidence_response(results, top_entry, top_score, session_id)
        # Medium confidence — partial match
        elif top_score >= MEDIUM_CONFIDENCE:
            response = self._medium_confidence_response(results, top_score, session_id)
        # Low confidence — out of domain
        else:
            response = self._low_confidence_response(session_id)

        # ─── RAG Contextual Awareness via MongoDB ───
        if latitude is not None and longitude is not None:
            # Check if user specifically asked about their location
            is_location_query = any(keyword in message.lower() for keyword in ["near me", "my place", "my location", "nearby", "around me", "here"])
            
            cat_str = response.get('category', '')
            if is_location_query or (response['confidence_level'] in ['high', 'medium'] and any(c in cat_str for c in ['Diseases', 'Pests', 'Weeds', 'Multiple Topics'])):
                
                nearby_diseases = self.db_service.get_nearby_diseases(float(latitude), float(longitude))
                
                # We check the top entry text and the actual reply text
                combined_text = (response.get('reply', '') + " " + top_entry.get('question', '')).lower()
                
                matched_disease = None
                for disease in nearby_diseases:
                    search_term = disease.replace('_', ' ').lower()
                    if search_term in combined_text:
                        matched_disease = disease
                        break
                
                # Scenario 1: User asked a specific question that matched a specific nearby disease
                if matched_disease:
                    formatted_disease = matched_disease.replace('_', ' ')
                    alert = f"🚨 **Action Recommended!** We have recently detected **{formatted_disease}** within 5km of your location. Please carefully review the guidance below.\n\n"
                    response['reply'] = alert + response['reply']
                    
                    response['action'] = {
                        "type": "NAVIGATE_TO_MAP",
                        "params": {
                            "diseaseName": matched_disease,
                            "latitude": float(latitude),
                            "longitude": float(longitude)
                        }
                    }
                # Scenario 2: User asked a general location question and there ARE diseases nearby
                elif is_location_query and nearby_diseases:
                    disease_list = ", ".join([d.replace('_', ' ') for d in nearby_diseases])
                    alert = f"🚨 **Warning:** We have detected the following issues within 5km of your location recently: **{disease_list}**.\n\nI recommend routinely checking your plantation for signs of these issues.\n\nHere is some general information:\n"
                    
                    if response['confidence_level'] == 'low':
                         response['reply'] = alert + "\nTry asking me specifically about one of those issues!"
                         response['category'] = "Local Alert"
                    else:
                         response['reply'] = alert + response['reply']
                         
                    # Add generic map action for the first disease found
                    response['action'] = {
                        "type": "NAVIGATE_TO_MAP",
                        "params": {
                            "diseaseName": list(nearby_diseases)[0],
                            "latitude": float(latitude),
                            "longitude": float(longitude)
                        }
                    }
                # Scenario 3: User asked about a specific disease, but it's NOT nearby
                elif not is_location_query and response['confidence_level'] in ['high', 'medium'] and any(c in cat_str for c in ['Diseases', 'Pests']):
                    reassurance = f"ℹ️ **Good News:** This issue has not been detected within 5km of your location recently. Here is the general information:\n\n"
                    response['reply'] = reassurance + response['reply']
                # Scenario 4: User asked "what is near me" but the area is clear
                elif is_location_query and not nearby_diseases:
                    response['reply'] = f"✅ **Great News!** We haven't detected any major rubber diseases or pests within 5km of your location recently.\n\n" + (response['reply'] if response['confidence_level'] != 'low' else "Is there anything specific you'd like to learn about?")

        return response

    def _high_confidence_response(self, results, top_entry, score, session_id):
        """Generate response for high-confidence matches."""
        category = top_entry.get('category', 'General')
        icon = CATEGORY_ICONS.get(category, '📘')

        reply = f"{top_entry['answer']}"

        # Add related topics from same category
        related = self._get_related_topics(top_entry, results)

        # Build source references
        sources = [{
            'category': entry.get('category', ''),
            'question': entry.get('question', ''),
            'score': round(s, 3)
        } for entry, s in results if s >= MEDIUM_CONFIDENCE]

        return {
            'reply': reply,
            'confidence': round(score, 3),
            'confidence_level': 'high',
            'category': f"{icon} {category}",
            'sources': sources,
            'suggested_topics': related
        }

    def _medium_confidence_response(self, results, score, session_id):
        """Generate response for medium-confidence (partial) matches."""
        # Compose from top results
        reply_parts = ["I found some related information that might help:\n"]

        for i, (entry, s) in enumerate(results):
            if s >= MEDIUM_CONFIDENCE:
                icon = CATEGORY_ICONS.get(entry.get('category', ''), '📘')
                reply_parts.append(
                    f"**{icon} {entry['question']}**\n{entry['answer'][:200]}..."
                    if len(entry['answer']) > 200
                    else f"**{icon} {entry['question']}**\n{entry['answer']}"
                )

        reply = '\n\n'.join(reply_parts)

        # Suggest clearer questions
        suggested = [entry['question'] for entry, s in results if s >= MEDIUM_CONFIDENCE]

        sources = [{
            'category': entry.get('category', ''),
            'question': entry.get('question', ''),
            'score': round(s, 3)
        } for entry, s in results if s >= MEDIUM_CONFIDENCE]

        return {
            'reply': reply,
            'confidence': round(score, 3),
            'confidence_level': 'medium',
            'category': 'Multiple Topics',
            'sources': sources,
            'suggested_topics': suggested[:3]
        }

    def _low_confidence_response(self, session_id):
        """Generate response for out-of-domain questions."""
        categories = self.embedding_service.get_categories()
        topics = self.embedding_service.get_topics_by_category()

        # Suggest some popular topics
        suggested = []
        for cat in categories[:4]:
            if cat in topics and topics[cat]:
                suggested.append(topics[cat][0]['question'])

        reply = (
            "I'm specialized in rubber cultivation and processing topics. "
            "I couldn't find a strong match for your question.\n\n"
            "Here are some topics I can help with:\n"
        )

        for cat in categories:
            icon = CATEGORY_ICONS.get(cat, '📘')
            reply += f"• {icon} {cat}\n"

        reply += "\nTry asking me about rubber diseases, latex quality, tapping, processing, or pests!"

        return {
            'reply': reply,
            'confidence': 0.0,
            'confidence_level': 'low',
            'category': 'Out of Domain',
            'sources': [],
            'suggested_topics': suggested
        }

    def _fallback_response(self):
        """Fallback when no results at all."""
        return {
            'reply': "I'm sorry, I wasn't able to search my knowledge base. Please try again.",
            'confidence': 0.0,
            'confidence_level': 'low',
            'category': 'Error',
            'sources': [],
            'suggested_topics': []
        }

    def _get_related_topics(self, current_entry, results):
        """Get related topics from same category but different from current."""
        category = current_entry.get('category', '')
        related = []
        for entry in self.embedding_service.knowledge:
            if entry['category'] == category and entry['id'] != current_entry['id']:
                related.append(entry['question'])
            if len(related) >= 3:
                break
        return related

    def get_topics(self):
        """Get all available topics grouped by category."""
        return self.embedding_service.get_topics_by_category()

    def get_welcome_message(self):
        """Return the initial welcome message."""
        count = len(self.embedding_service.knowledge)
        categories = self.embedding_service.get_categories()

        return {
            'reply': (
                f"Hello! I'm **RubberBot** 🌿, your rubber cultivation expert assistant.\n\n"
                f"I have knowledge on **{count}+ topics** across {len(categories)} categories:\n"
                + ''.join(f"• {CATEGORY_ICONS.get(c, '📘')} {c}\n" for c in categories) +
                f"\nAsk me anything about rubber diseases, latex quality, tapping, "
                f"processing, pest control, and more!"
            ),
            'confidence': 1.0,
            'confidence_level': 'high',
            'category': 'Welcome',
            'sources': [],
            'suggested_topics': [
                "What is Corynespora leaf fall disease?",
                "What is DRC and how to measure it?",
                "How to make ribbed smoked sheets?",
                "What are the recommended rubber clones for Sri Lanka?"
            ]
        }
