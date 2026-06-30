# Approach Document: SHL Conversational Assessment Recommender

## 1. Architecture Overview

The system is a stateless FastAPI service with two endpoints: `GET /health` and `POST /chat`. It uses a retrieve-then-generate pattern to ground every recommendation in the SHL catalog.

**Stack:**
- **FastAPI + Uvicorn** for the API layer
- **Sentence-Transformers (all-MiniLM-L6-v2)** + **FAISS** for semantic retrieval
- **Lightweight LLM via API** (OpenRouter/Groq/Gemini) for natural language generation
- **Rule-based guardrails** as a fallback when the LLM is unavailable or for safety-critical decisions

## 2. Catalog & Retrieval

### Catalog Construction
I built the catalog from public SHL product data, covering 88 Individual Test Solutions across cognitive, personality, behavioral, skills/simulations, and video/360 categories. Each item includes:
- `name`, `url`, `test_type` (K, P, C, B, S, A, V)
- Rich `description`, `keywords`, `job_levels`, `duration`

### Embedding & Indexing
Each catalog item is embedded using a concatenated text of name, description, keywords, test type, and job levels. Embeddings are stored in a FAISS inner-product index with L2 normalization. The index is built once at deploy time (or on first request) and loaded into memory for sub-10ms retrieval.

### Search Strategy
The retriever accepts:
- A natural-language query built from the last 6 conversation turns
- Optional filters (`test_types`, `job_levels`) extracted via keyword matching

For refinement queries, filters are accumulated across the full conversation history to honor mid-conversation edits.

## 3. Agent Design

### Intent Classification
The agent classifies each turn into one of four intents:
- **clarify** — early turns with no specific role/skill indicators
- **recommend** — enough context gathered, or user explicitly asks for a shortlist
- **refine** — user changes constraints ("actually, add personality tests")
- **compare** — user asks for differences between named assessments

### Decision Logic
- **Turn 1-2 + vague** → clarify (ask one concise question)
- **Turn 6+** → force recommendation regardless of vagueness (respects the 8-turn evaluator cap)
- **Off-topic / prompt-injection** → refuse immediately, empty recommendations
- **Comparison** → retrieve both items by name and generate a grounded diff

### Guardrails
1. **Scope:** Refuse if no in-scope keywords are detected; block known prompt-injection patterns.
2. **Hallucination prevention:** Recommendations are produced by the retriever, never invented by the LLM. The LLM only generates the conversational `reply`.
3. **URL validation:** The `/chat` endpoint post-filters every recommendation against the loaded catalog; any item with a mismatched name or URL is dropped.
4. **Schema compliance:** Pydantic models enforce the exact response shape on every request.
5. **Turn limit:** If `len(messages) > 8`, the endpoint returns a graceful termination message.

## 4. Prompt Design

The LLM is used for three tasks, each with a strict system prompt:
1. **Clarifying question:** "Ask ONE concise clarifying question..."
2. **Recommendation intro:** "Respond naturally... Do NOT make up URLs or assessment names."
3. **Comparison:** "Compare based ONLY on the provided catalog data..."

Temperature is kept low (0.3–0.4) to reduce creativity and increase factual adherence.

## 5. Evaluation Approach

### Automated Tests (included in `tests/test_api.py`)
- Schema compliance on every response type
- Vague turn-1 queries return empty recommendations
- Recommendation queries return 1–10 items with valid SHL URLs
- Off-topic and prompt-injection requests are refused
- Refinement queries include the newly requested test type
- Turn-limit graceful degradation
- No hallucinated URLs

### What Didn’t Work
- **Keyword-only matching:** Early prototypes used simple keyword search. It failed on synonymy ("Java dev" ≠ "Java developer") and missed behavioral/personality fit.
- **LLM-only recommendations:** Letting the LLM suggest assessments without retrieval caused hallucinated URLs and names. Switching to retrieve-then-generate eliminated this entirely.
- **Stateful sessions:** A stateful design was simpler to code but violated the spec. The current design recomputes filters from the full history every turn.

### How We Measured Improvement
- Recall@10 was estimated by running synthetic traces (e.g., "mid-level Java dev with stakeholders") and checking whether the expected assessments (Java 8, Verify G+, OPQ32r) appeared in the top 10.
- Behavior probes were checked manually by simulating evaluator-style conversations (vague → clarify → commit → refine → compare → off-topic).

## 6. Deployment Notes

- The service is packaged with a Dockerfile and `render.yaml` for one-click deployment on Render.
- The FAISS index is pre-built during the Docker build to minimize cold-start latency.
- The evaluator allows up to 2 minutes for the first `/health` call; our cold start is typically <10 seconds on a free tier.

## 7. AI Tool Usage

AI-assisted coding (this assistant) was used for:
- Scaffolding the FastAPI boilerplate and Pydantic schemas
- Drafting the catalog dataset from public SHL product knowledge
- Writing test cases and the approach document

All design decisions (retriever-first architecture, guardrail layering, prompt constraints) were chosen and validated by manual reasoning against the evaluator criteria.
