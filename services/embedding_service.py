import json
import os
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Try to import ML libraries. If unavailable, fall back to TF-IDF.
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    USE_TRANSFORMERS = True
    logger.info("Using sentence-transformers + FAISS for embeddings")
except ImportError:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
    USE_TRANSFORMERS = False
    logger.info("Falling back to TF-IDF (install sentence-transformers + faiss-cpu for better results)")


class EmbeddingService:
    """
    Semantic embedding and retrieval service for rubber knowledge.
    Uses sentence-transformers + FAISS if available, falls back to TF-IDF.
    """

    def __init__(self, knowledge_path: str = None):
        if knowledge_path is None:
            knowledge_path = os.path.join(
                os.path.dirname(__file__), '..', 'knowledge', 'rubber_knowledge.json'
            )

        # Load knowledge base
        with open(knowledge_path, 'r', encoding='utf-8') as f:
            self.knowledge = json.load(f)

        logger.info(f"Loaded {len(self.knowledge)} knowledge entries")

        # Build search texts (combine question + keywords for better matching)
        self.search_texts = []
        for entry in self.knowledge:
            keywords_str = ' '.join(entry.get('keywords', []))
            text = f"{entry['question']} {keywords_str} {entry.get('category', '')}"
            self.search_texts.append(text)

        # Build the index
        if USE_TRANSFORMERS:
            self._build_faiss_index()
        else:
            self._build_tfidf_index()

    def _build_faiss_index(self):
        """Build FAISS index using sentence-transformers embeddings."""
        logger.info("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        logger.info("Encoding knowledge base...")
        self.embeddings = self.model.encode(self.search_texts, show_progress_bar=True)

        # Normalize for cosine similarity
        faiss.normalize_L2(self.embeddings)

        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product = cosine on normalized vectors
        self.index.add(self.embeddings)

        logger.info(f"FAISS index built with {self.index.ntotal} vectors (dim={dimension})")

    def _build_tfidf_index(self):
        """Build TF-IDF index as fallback."""
        logger.info("Building TF-IDF index...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.search_texts)
        logger.info(f"TF-IDF index built with {self.tfidf_matrix.shape[0]} documents")

    def search(self, query: str, top_k: int = 3):
        """
        Search for the most relevant knowledge entries.

        Returns: List of (entry, score) tuples, sorted by relevance.
        """
        if USE_TRANSFORMERS:
            return self._search_faiss(query, top_k)
        else:
            return self._search_tfidf(query, top_k)

    def _search_faiss(self, query: str, top_k: int):
        """Search using FAISS + sentence-transformers."""
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.knowledge):
                results.append((self.knowledge[idx], float(score)))

        return results

    def _search_tfidf(self, query: str, top_k: int):
        """Search using TF-IDF cosine similarity."""
        query_vec = self.vectorizer.transform([query])
        similarities = sklearn_cosine(query_vec, self.tfidf_matrix).flatten()

        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append((self.knowledge[idx], float(similarities[idx])))

        return results

    def get_categories(self):
        """Get all unique categories from the knowledge base."""
        return sorted(set(entry['category'] for entry in self.knowledge))

    def get_topics_by_category(self):
        """Get topics grouped by category."""
        topics = {}
        for entry in self.knowledge:
            cat = entry['category']
            if cat not in topics:
                topics[cat] = []
            topics[cat].append({
                'id': entry['id'],
                'question': entry['question']
            })
        return topics
