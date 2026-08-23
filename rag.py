from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.arxiv import ArxivLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
from urllib.parse import urlparse
import re

from prompts import create_history_prompt, create_qa_prompt

load_dotenv()


def extract_arxiv_id(url):
    """Return an arXiv identifier from a PDF URL, abstract URL, or raw ID."""
    value = (url or "").strip()
    if not value:
        return None
    parsed = urlparse(value)
    candidate = parsed.path if parsed.scheme else value
    candidate = re.sub(r"^/(?:pdf|abs)/", "", candidate).strip("/")
    candidate = re.sub(r"\.pdf$", "", candidate, flags=re.IGNORECASE)
    pattern = r"(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?"
    match = re.fullmatch(pattern, candidate, flags=re.IGNORECASE)
    return match.group(0) if match else None


def create_arxiv_retriever(
    pdf_url,
    embedding_model="BAAI/bge-small-en-v1.5",
):
    arxiv_id = extract_arxiv_id(pdf_url)
    if not arxiv_id:
        raise ValueError("Enter a valid arXiv PDF URL, abstract URL, or paper ID.")
    loader = ArxivLoader(query=arxiv_id)
    documents = loader.load()
    if not documents:
        raise ValueError(f"No arXiv paper was found for '{arxiv_id}'.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=16)
    documents = text_splitter.split_documents(documents=documents)
    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=FastEmbedEmbeddings(model_name=embedding_model),
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    metadata = documents[0].metadata

    return retriever, documents, metadata


store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def create_conversational_rag_chain(
    pdf_url,
    groq_api_key,
    chat_model="openai/gpt-oss-20b",
    embedding_model="BAAI/bge-small-en-v1.5",
):
    if not groq_api_key:
        raise ValueError("A Groq API key is required.")
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model=chat_model,
        temperature=0,
        max_retries=2,
    )
    retriever, documents, metadata = create_arxiv_retriever(
        pdf_url=pdf_url,
        embedding_model=embedding_model,
    )

    history_prompt = create_history_prompt()
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, history_prompt
    )

    qa_prompt = create_qa_prompt(metadata)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain
