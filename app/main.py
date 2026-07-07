from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.api.schemas import QueryOutput, QueryInput, Topic
from app.config import get_settings
from app.generation.pipeline import RagPipeline
from app.generation.study import get_topics, get_suggestions
from app.models.factory import get_embedder, get_chatter
from app.retrieval.faiss_store import FaissStore

state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    store = FaissStore.load(settings.index_dir, expected_embedder=settings.embedder)
    state["rag"] = RagPipeline(get_embedder(), store, get_chatter(), settings)
    state["chunks"] = store._chunks
    yield
    state.clear()

app = FastAPI(
    title="Notes-RAG",
    version="1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks": len(state.get("chunks", []))
    }

@app.get("/query", response_model=QueryOutput)
def query(body: QueryInput):
    try:
        return state["rag"].answer(body.question, rewrite=body.rewrite)
    except Exception as e:
        raise HTTPException(status_code=500, detail=e)

@app.get("/topics", response_model=list[Topic])
def topics(subject: str | None = None):
    chunks = state["chunks"]
    if subject:
        chunks = [c for c in chunks if c.subject.lower() == subject.lower()]
    return get_topics(state["rag"].chatter, chunks)

@app.get("/suggestions")
def suggestions(subject: str | None = None):
    chunks = state["chunks"]
    if subject:
        chunks = [
            c for c in chunks if c.subject.lower() == subject.lower()
        ]
    return {
        "suggestions": get_suggestions(state["rag"].chatter, chunks)
    }