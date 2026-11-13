"""Deterministic local retrieval; no API keys, model downloads or paid calls."""

from pathlib import Path
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from llama_index.core.node_parser import SentenceSplitter
from langchain_core.runnables import RunnableLambda

ROOT = Path(__file__).resolve().parents[1]
PROMPT_VERSION = "grounded-extractive-v1"


class KnowledgeAssistant:
    def __init__(self, directory=None):
        self.chunks = []
        splitter = SentenceSplitter(chunk_size=100, chunk_overlap=15)
        for path in sorted(Path(directory or ROOT / "data/documents").glob("*.md")):
            for i, text in enumerate(
                splitter.split_text(path.read_text(encoding="utf-8"))
            ):
                self.chunks.append(
                    {"id": f"{path.stem}:{i}", "doc_id": path.stem, "text": text}
                )
        if not self.chunks:
            raise ValueError("No Markdown documents found")
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.sparse = self.vectorizer.fit_transform(c["text"] for c in self.chunks)
        # Corpus-fitted latent semantic analysis, NOT pretrained neural embeddings.
        self.svd = TruncatedSVD(
            n_components=min(4, self.sparse.shape[0] - 1), random_state=42
        )
        self.dense = normalize(self.svd.fit_transform(self.sparse))
        self.chain = RunnableLambda(self._retrieve_request) | RunnableLambda(
            self._answer
        )

    def retrieve(self, question, doc_id=None, k=3):
        query = self.vectorizer.transform([question])
        lexical = (self.sparse @ query.T).toarray().ravel()
        latent = (self.dense @ normalize(self.svd.transform(query)).T).ravel()
        scores = 0.85 * lexical + 0.15 * np.clip(latent, 0, 1)
        eligible = [
            i
            for i, c in enumerate(self.chunks)
            if doc_id is None or c["doc_id"] == doc_id
        ]
        ranked = sorted(eligible, key=lambda i: float(scores[i]), reverse=True)[:k]
        return [
            {
                **self.chunks[i],
                "score": round(float(scores[i]), 4),
                "lexical_score": round(float(lexical[i]), 4),
            }
            for i in ranked
            if lexical[i] >= 0.06
        ]

    def _retrieve_request(self, request):
        return {
            **request,
            "sources": self.retrieve(request["question"], request.get("doc_id")),
        }

    def _answer(self, request):
        sources = request["sources"]
        if not sources:
            return {
                "answer": "I could not find supporting information in these documents.",
                "sources": [],
                "abstained": True,
                "mode": "extractive",
                "prompt_version": PROMPT_VERSION,
            }
        # Return source sentences, never execute instructions found in documents.
        content = "\n".join(
            line
            for line in sources[0]["text"].splitlines()
            if not line.lstrip().startswith("#")
        )
        sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", content) if s.strip()
        ]
        matrix = self.vectorizer.transform(sentences)
        query = self.vectorizer.transform([request["question"]])
        scores = (matrix @ query.T).toarray().ravel()
        chosen = [i for i in np.argsort(-scores)[:2] if scores[i] > 0.05]
        if not chosen:
            chosen = [int(np.argmax(scores))]
        answer = " ".join(sentences[int(i)] for i in sorted(chosen))
        return {
            "answer": answer + f" [{sources[0]['id']}]",
            "sources": sources,
            "abstained": False,
            "mode": "extractive",
            "prompt_version": PROMPT_VERSION,
        }

    def ask(self, question, doc_id=None):
        return self.chain.invoke({"question": question, "doc_id": doc_id})
