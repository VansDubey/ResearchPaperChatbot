# Archie: arXiv Research Paper Chatbot

## 1. Thirty-second interview explanation

Archie is a conversational RAG application for arXiv papers. A user supplies an
arXiv URL. The system downloads the paper, splits it into
overlapping chunks, converts each chunk into an embedding, and stores the vectors
in an in-memory FAISS index. For each question, LangChain first rewrites follow-up
questions into standalone queries, retrieves the two most relevant chunks, and
asks a locally hosted Ollama model to answer using that context. Streamlit provides the UI,
and session-scoped chat history enables multi-turn conversation.

## 2. End-to-end request flow

1. `streamlit_app.py` collects the arXiv URL/ID, Ollama settings, and session ID.
2. `extract_arxiv_id()` validates and normalizes the paper identifier.
3. `ArxivLoader` downloads and parses the paper and its metadata.
4. `RecursiveCharacterTextSplitter` creates 512-character chunks with a
   16-character overlap. Overlap reduces the chance of losing meaning at a boundary.
5. `OllamaEmbeddings` with `nomic-embed-text` maps every chunk to a dense vector.
6. `FAISS.from_documents()` builds an in-memory similarity-search index.
7. When a user asks a question, the history-aware prompt turns a follow-up such as
   "What were its limitations?" into a standalone question.
8. FAISS retrieves the top two semantically similar chunks.
9. The QA prompt combines paper metadata, retrieved chunks, and the question.
10. `ChatOllama` generates the answer, `ask_question()` streams it to Streamlit,
    and `RunnableWithMessageHistory` saves the turn under the session ID.

## 3. File responsibilities

- `streamlit_app.py`: page, sidebar, chat rendering, session state, error messages.
- `rag.py`: ingestion, chunking, embeddings, FAISS, retriever, LLM, memory wiring.
- `prompts.py`: question-rewriting and grounded-answer prompt templates.
- `utils.py`: random session IDs and streaming adapter.
- `requirements.txt`: pinned Python dependencies.
- `Dockerfile`: reproducible Python 3.11 container and health check.

## 4. Why RAG instead of fine-tuning?

RAG can ingest a new paper at runtime, keeps answers tied to source text, and avoids
training a model for every document. Fine-tuning is better for changing behavior or
style; it is not the best primary mechanism for injecting an arbitrary new paper.

## 5. Important design choices and trade-offs

- FAISS is fast and simple for a demo, but this index disappears when the process
  restarts and is not shared across replicas.
- In-memory chat history is equally simple, but should become Redis or a database
  in production.
- `k=2` lowers prompt cost but can miss supporting evidence. It should be tuned with
  an evaluation set; hybrid search and reranking are natural improvements.
- Character-based chunks are simple but not token- or section-aware. A production
  version should preserve headings and page/source metadata.
- RAG reduces hallucinations but does not eliminate them. Answers should expose
  citations and decline when retrieved evidence is insufficient.
- Creating an index on every button click is wasteful. Cache by paper ID or persist
  the vector index in production.

## 6. Interview questions to expect

**What is an embedding?** A numeric vector that represents semantic meaning, so
similar text is close under a distance metric.

**How does FAISS retrieval work?** The question is embedded into the same vector
space, and nearest-neighbor search returns the chunks with the closest vectors.

**Why is question rewriting needed?** A retriever cannot reliably interpret pronouns
from previous turns. Rewriting adds the missing conversational context.

**How would you evaluate it?** Build paper-specific questions with expected answers,
then separately measure retrieval recall/precision and answer faithfulness,
correctness, citation accuracy, latency, and token cost.

**How would you scale it?** Move indexing to a background job, persist vectors in a
shared vector database, store history in Redis/Postgres, cache per paper, add auth
and rate limiting, run stateless app replicas, and add tracing and RAG evaluations.

**Main security concerns?** Never log or commit API keys; validate URLs/IDs; impose
paper-size, request, and token limits; rate-limit users; treat retrieved text as
untrusted prompt input; and keep dependencies patched.

## 7. Local run

Use Python 3.10 or 3.11 (the current pinned FAISS stack may not provide a wheel for
newer Python versions).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env  # Windows; use `cp` on macOS/Linux
streamlit run streamlit_app.py
```

Install Ollama, then download and start the models before running Streamlit:

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
ollama serve
```

No external LLM API key is required. `.env` can override the Ollama URL and model names.

## 8. Deployment

### Streamlit Community Cloud

Streamlit Community Cloud cannot reach Ollama running on a developer laptop. Use a
publicly reachable, secured Ollama server or deploy the app and Ollama together on
a VM/container platform. Never expose an unauthenticated Ollama port publicly.

### Docker

```bash
docker build -t archie-rag .
docker run --rm -p 8501:8501 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 archie-rag
```

Open `http://localhost:8501`. The same image can be deployed to Cloud Run, Render,
Azure Container Apps, ECS, or another container service. Configure port `8501`, the
health path `/_stcore/health`. The Ollama endpoint must be reachable from the app
container and have sufficient CPU/RAM or GPU resources for the selected model.

## 9. Current completion boundary

The repository is deployment-ready for a portfolio/demo environment. A genuine
production launch still needs persistent/shared storage, authentication, rate
limits, observability, source citations, automated RAG evaluation, and cost controls.
