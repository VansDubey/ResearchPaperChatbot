# Archie: arXiv Research Paper Chatbot

## 1. Thirty-second interview explanation

Archie is a conversational RAG application for arXiv papers. A user supplies an
arXiv URL. The system downloads the paper, splits it into
overlapping chunks, converts each chunk into an embedding, and stores the vectors
in an in-memory FAISS index. For each question, LangChain first rewrites follow-up
questions into standalone queries, retrieves the two most relevant chunks, and
asks a Groq-hosted model to answer using that context. Streamlit provides the UI,
and session-scoped chat history enables multi-turn conversation.

## 2. End-to-end request flow

1. `streamlit_app.py` collects the arXiv URL/ID, Groq key, and session ID.
2. `extract_arxiv_id()` validates and normalizes the paper identifier.
3. `ArxivLoader` downloads and parses the paper and its metadata.
4. `RecursiveCharacterTextSplitter` creates 512-character chunks with a
   16-character overlap. Overlap reduces the chance of losing meaning at a boundary.
5. `FastEmbedEmbeddings` maps every chunk to a dense vector locally.
6. `FAISS.from_documents()` builds an in-memory similarity-search index.
7. When a user asks a question, the history-aware prompt turns a follow-up such as
   "What were its limitations?" into a standalone question.
8. FAISS retrieves the top two semantically similar chunks.
9. The QA prompt combines paper metadata, retrieved chunks, and the question.
10. `ChatGroq` generates the answer, `ask_question()` streams it to Streamlit,
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

Create a free Groq API key, put it in `.env`, and run Streamlit:

```bash
GROQ_API_KEY=gsk_replace_me
streamlit run streamlit_app.py
```

The key can instead be entered in the sidebar. Embeddings remain local; only the
question, retrieved context, and chat history are sent to Groq.

## 8. Deployment

### Streamlit Community Cloud

Push the repository to GitHub, create a Streamlit Community Cloud app, select
`streamlit_app.py`, and add `GROQ_API_KEY` in the app's Secrets settings. The app
can also deploy without a stored secret and ask each user for their own key.

### Docker

```bash
docker build -t archie-rag .
docker run --rm -p 8501:8501 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 archie-rag
```

Open `http://localhost:8501`. The same image can be deployed to Cloud Run, Render,
Azure Container Apps, ECS, or another container service. Configure port `8501`, the
health path `/_stcore/health`, and inject `GROQ_API_KEY` with the platform's secret
manager.

## 9. Current completion boundary

The repository is deployment-ready for a portfolio/demo environment. A genuine
production launch still needs persistent/shared storage, authentication, rate
limits, observability, source citations, automated RAG evaluation, and cost controls.
