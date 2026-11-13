from functools import lru_cache
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from .retrieval import KnowledgeAssistant, ROOT

app = FastAPI(title="Northstar Knowledge Lab", version="1.0.0")


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    doc_id: str | None = Field(default=None, max_length=80)


@lru_cache
def assistant():
    return KnowledgeAssistant()


@app.get("/health")
def health():
    return {"status": "ok", "mode": "local-extractive"}


@app.get("/")
def home():
    return FileResponse(ROOT / "web/index.html")


@app.post("/ask")
def ask(body: Question):
    return assistant().ask(body.question, body.doc_id)
