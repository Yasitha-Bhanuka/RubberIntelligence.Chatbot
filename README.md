# RubberIntelligence Chatbot 🌿

A fully offline, RAG-based (Retrieval-Augmented Generation) chatbot for rubber cultivation and processing knowledge. Built as a university research component — no external APIs required.

## Architecture

```
User Question
    ↓
sentence-transformers (local model, ~80MB)
    ↓ embed question into vector
FAISS Vector Store (in-memory)
    ↓ find top-3 most similar knowledge entries
Confidence Scoring
    ↓
    ├── ≥ 0.65  →  ✅ Expert answer + related topics
    ├── 0.30–0.65 →  🟡 Partial matches + suggestions
    └── < 0.30  →  🔴 Out of domain + available categories
```

## Knowledge Base

**50+ expert-curated entries** across 8 categories:

| Category      | Icon | Topics                                                                  |
| ------------- | ---- | ----------------------------------------------------------------------- |
| Diseases      | 🦠    | Corynespora, Phytophthora, Oidium, White Root, Pink Disease, Brown Bast |
| Pests         | 🐛    | Mites, scale insects, termites, bark borers, white grubs                |
| Weeds         | 🌿    | Imperata, Mikania, cover crops, weeding schedules                       |
| Latex Quality | 🧪    | DRC, VFA, MST, pH, ammonia, TSR/RSS grading                             |
| Cultivation   | 🌱    | Tapping, clones, budding, fertilizer, land preparation                  |
| Climate       | 🌤️    | Temperature, rainfall, soil, monsoon effects                            |
| Economics     | 💰    | Prices, yield optimization, costs, intercropping                        |
| Processing    | 🏭    | RSS, crepe, centrifuged latex, block rubber                             |

## Setup Instructions

### Prerequisites

- **Python 3.9+** ([Download](https://www.python.org/downloads/))
- **pip** (comes with Python)

### Step 1: Create Virtual Environment

```bash
cd RubberIntelligence.Chatbot
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The `sentence-transformers` package will download the `all-MiniLM-L6-v2` model (~80MB) on first run. This only happens once.

### Step 3: Run the Server

if using portable device for checking this app.4

netsh advfirewall firewall add rule name="RubberBot Chatbot" dir=in action=allow protocol=TCP localport=5008


netsh advfirewall firewall show rule name=all dir=in  | Select-String -Pattern "5008" -Context 3,0

then we can see : There's already a firewall rule allowing TCP traffic on port 5008 inbound. So the firewall is not blocking the chatbot.

```bash
python app.py
```

The server starts on **http://localhost:5008**.

### Step 4: Test

```bash
# Health check
curl http://localhost:5008/api/chat/health

# Send a message
curl -X POST http://localhost:5008/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is Corynespora leaf fall?", "sessionId": "test-123"}'

# Get available topics
curl http://localhost:5008/api/chat/topics

# Get welcome message
curl http://localhost:5008/api/chat/welcome
```

## API Endpoints

| Method | Endpoint            | Description                      |
| ------ | ------------------- | -------------------------------- |
| POST   | `/api/chat`         | Send a message, get AI response  |
| GET    | `/api/chat/welcome` | Get welcome message + categories |
| GET    | `/api/chat/topics`  | Get all topics by category       |
| GET    | `/api/chat/health`  | Health check + stats             |

### POST /api/chat

**Request:**
```json
{
  "message": "How much ammonia to preserve latex?",
  "sessionId": "optional-session-id"
}
```

**Response:**
```json
{
  "reply": "Ammonia is the primary preservative...",
  "confidence": 0.847,
  "confidence_level": "high",
  "category": "🧪 Latex Quality",
  "sources": [
    { "category": "Latex Quality", "question": "How much ammonia...", "score": 0.847 }
  ],
  "suggested_topics": [
    "What is VFA number and why is it important?",
    "What is the ideal storage temperature for latex?"
  ]
}
```

## Extending the Knowledge Base

To add new entries, edit `knowledge/rubber_knowledge.json`:

```json
{
  "id": "d011",
  "category": "Diseases",
  "question": "Your question here?",
  "answer": "Detailed expert answer...",
  "keywords": ["keyword1", "keyword2"]
}
```

Then restart the server — the FAISS index is rebuilt automatically on startup.

## Research Evaluation

For your thesis, you can evaluate:

1. **Retrieval Accuracy**: Test with 50+ questions → measure if the correct knowledge entry is in top-3
2. **Confidence Calibration**: Check if high-confidence responses are actually correct
3. **Domain Boundaries**: Test with out-of-domain questions → should return low confidence
4. **Comparison**: Compare RAG responses vs a general-purpose chatbot on the same rubber questions

## Fallback Mode

If `sentence-transformers` and `faiss-cpu` are not installed, the system automatically falls back to **TF-IDF** vectorization with cosine similarity. This is still functional but less accurate for semantic understanding (e.g., "latex going bad" won't match "VFA number increasing" as well).

## Project Structure

```
RubberIntelligence.Chatbot/
├── app.py                         # Flask server (port 5008)
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── knowledge/
│   └── rubber_knowledge.json      # 50+ curated knowledge entries
└── services/
    ├── __init__.py
    ├── embedding_service.py       # sentence-transformers + FAISS indexing
    └── chat_service.py            # RAG pipeline + confidence scoring
```
