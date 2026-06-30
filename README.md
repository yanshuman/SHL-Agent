# SHL Conversational Assessment Recommender

A stateless FastAPI service that helps hiring managers find SHL Individual Test Solutions through natural dialogue.

## Quick Start (Local)

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt
pip install 'huggingface-hub<0.24'  # compatibility fix for sentence-transformers 2.2.2

# 3. Build the search index (one-time)
python -c "from src.retriever import CatalogRetriever; CatalogRetriever('data/catalog.json', 'data/catalog.index')"

# 4. Run the server
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

Test:
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hiring a Java developer"}]}'
```

## LLM Setup (Optional but Recommended)

The service works without an LLM using rule-based replies, but natural language quality improves significantly with a lightweight model.

**Option A: OpenRouter (free tier available)**
```bash
export LLM_PROVIDER=openrouter
export LLM_API_KEY=sk-or-v1-...
export LLM_MODEL=meta-llama/llama-3.1-8b-instruct:free
```

**Option B: Groq (free tier available)**
```bash
export LLM_PROVIDER=groq
export LLM_API_KEY=gsk_...
export LLM_MODEL=llama-3.1-8b-instant
```

**Option C: Google Gemini (free tier available)**
```bash
export LLM_PROVIDER=gemini
export LLM_API_KEY=...
export LLM_MODEL=gemini-1.5-flash
```

## Using Official SHL Catalog Data

If SHL provided an official catalog file (CSV/JSON), replace `data/catalog.json` with it. The expected schema per item:
```json
{
  "name": "Assessment Name",
  "url": "https://www.shl.com/...",
  "test_type": "K",
  "description": "...",
  "keywords": ["..."],
  "duration": "30 min",
  "remote_testing": "Y",
  "adaptive": "N",
  "job_levels": ["Entry", "Mid"]
}
```

Delete `data/catalog.index` and `data/catalog.index.mapping` so the index rebuilds with the new data.

## Deployment

### Render (Recommended)
1. Fork/push this repo to GitHub.
2. Create a new Web Service on [Render](https://render.com).
3. Connect your repo and use the settings from `render.yaml`.
4. Add your `LLM_API_KEY` as an environment variable.
5. Deploy. The service will be live in ~2 minutes.

### Docker
```bash
docker build -t shl-agent .
docker run -p 8000:8000 -e LLM_API_KEY=... shl-agent
```

## API Specification

### GET /health
Response:
```json
{"status": "ok"}
```

### POST /chat
Request:
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "Sure. What is seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}
```

Response:
```json
{
  "reply": "Got it. Here are 5 assessments that fit a mid-level Java dev with stakeholder needs.",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32r", "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

- `recommendations` is empty when clarifying or refusing.
- `recommendations` contains 1–10 items when a shortlist is committed.
- `end_of_conversation` is `true` only when the task is considered complete.

## Testing

```bash
# Run all behavior probes
python tests/test_api.py
```

## Project Structure

```
shl-agent/
├── data/
│   ├── catalog.json          # SHL assessment catalog
│   ├── catalog.index         # FAISS vector index (auto-generated)
│   └── catalog.index.mapping # Name mapping (auto-generated)
├── src/
│   ├── main.py               # FastAPI app
│   ├── agent.py              # Conversation logic & LLM integration
│   └── retriever.py          # Semantic search with FAISS
├── tests/
│   └── test_api.py           # Automated behavior probes
├── requirements.txt
├── Dockerfile
├── render.yaml
├── APPROACH.md
└── README.md
```
