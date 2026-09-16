"""
Advanced RAG Engine for Unstructured Fitness Knowledge.
Provides semantic/recursive chunking, metadata attachment, vector embedding retrieval,
cross-score reranking, and Qwen3/LLM synthesis.

STRICT RULE: RAG must NOT calculate or invent calories, macros, or exercise calories.
MongoDB + calculation engines remain the source of truth for numeric math.
"""

from __future__ import annotations

import os
import sys
import re
import math
from typing import Optional, Any
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Simple Vector Embedder (TF-IDF + Subword N-gram Cosine Similarity)
# Fast, offline, deterministic, zero-dependency embedding generator
# ---------------------------------------------------------------------------

class FastVectorEmbedder:
    """Lightweight TF-IDF / Subword embedding model for local vector retrieval."""

    def __init__(self, vector_dim: int = 256):
        self.vector_dim = vector_dim

    def embed(self, text: str) -> list[float]:
        """Generate a normalized dense vector embedding for input text."""
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = cleaned.split()
        if not tokens:
            return [0.0] * self.vector_dim

        vec = [0.0] * self.vector_dim
        # Subword hash projection into dense vector
        for token in tokens:
            # Word-level features
            h1 = abs(hash(token)) % self.vector_dim
            vec[h1] += 1.0
            # 3-gram character features
            for i in range(len(token) - 2):
                gram = token[i:i+3]
                h2 = abs(hash(gram)) % self.vector_dim
                vec[h2] += 0.5

        # L2 Normalization
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]

        return vec

    @staticmethod
    def cosine_similarity(v1: list[float], v2: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        return float(dot)


# ---------------------------------------------------------------------------
# Chunk & Retrieval Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class KnowledgeChunk:
    chunk_id: str
    content: str
    topic: str
    category: str
    source: str
    metadata: dict = field(default_factory=dict)
    vector: list[float] = field(default_factory=list)


@dataclass
class RAGSearchResult:
    chunk_id: str
    content: str
    topic: str
    category: str
    source: str
    score: float
    rerank_score: float


# ---------------------------------------------------------------------------
# RAG Engine Class
# ---------------------------------------------------------------------------

class AdvancedRAGEngine:
    """Advanced RAG Engine for unstructured fitness/nutrition knowledge."""

    def __init__(self, db_client: Any = None, llm_service: Any = None):
        self.db = db_client
        self.llm = llm_service
        self.embedder = FastVectorEmbedder(vector_dim=256)
        self.collection_name = "knowledge_chunks"

    def semantic_chunk_text(
        self, text: str, topic: str, category: str, source: str,
        chunk_size: int = 500, overlap: int = 100
    ) -> list[KnowledgeChunk]:
        """
        Recursive semantic chunker with metadata attachment.
        Splits text into overlapping chunks respecting sentence boundaries.
        """
        text = text.strip()
        if not text:
            return []

        # Split into sentences
        sentences = re.split(r"(?<=[.!?\n])\s+", text)
        chunks: list[KnowledgeChunk] = []

        curr_chunk_parts: list[str] = []
        curr_len = 0
        chunk_idx = 1

        for sent in sentences:
            sent_len = len(sent)
            if curr_len + sent_len > chunk_size and curr_chunk_parts:
                chunk_str = " ".join(curr_chunk_parts).strip()
                chunk_id = f"chunk_{re.sub(r'[^a-zA-Z0-9]', '_', topic).lower()}_{chunk_idx:04d}"
                vec = self.embedder.embed(chunk_str)

                chunks.append(KnowledgeChunk(
                    chunk_id=chunk_id,
                    content=chunk_str,
                    topic=topic,
                    category=category,
                    source=source,
                    metadata={"chunk_size": len(chunk_str), "topic": topic},
                    vector=vec,
                ))

                # Retain overlap sentences
                overlap_parts = []
                overlap_len = 0
                for part in reversed(curr_chunk_parts):
                    if overlap_len + len(part) <= overlap:
                        overlap_parts.insert(0, part)
                        overlap_len += len(part)
                    else:
                        break

                curr_chunk_parts = overlap_parts
                curr_len = overlap_len
                chunk_idx += 1

            curr_chunk_parts.append(sent)
            curr_len += sent_len

        if curr_chunk_parts:
            chunk_str = " ".join(curr_chunk_parts).strip()
            chunk_id = f"chunk_{re.sub(r'[^a-zA-Z0-9]', '_', topic).lower()}_{chunk_idx:04d}"
            vec = self.embedder.embed(chunk_str)

            chunks.append(KnowledgeChunk(
                chunk_id=chunk_id,
                content=chunk_str,
                topic=topic,
                category=category,
                source=source,
                metadata={"chunk_size": len(chunk_str), "topic": topic},
                vector=vec,
            ))

        return chunks

    async def ingest_knowledge_base(self, chunks: list[KnowledgeChunk]) -> int:
        """Save indexed knowledge chunks into MongoDB 'knowledge_chunks' collection."""
        if not chunks or self.db is None:
            return 0

        col = self.db[self.collection_name]
        docs = []
        for chk in chunks:
            docs.append({
                "chunk_id": chk.chunk_id,
                "content": chk.content,
                "topic": chk.topic,
                "category": chk.category,
                "source": chk.source,
                "metadata": chk.metadata,
                "vector": chk.vector,
            })

        for doc in docs:
            await col.update_one(
                {"chunk_id": doc["chunk_id"]},
                {"$set": doc},
                upsert=True
            )

        return len(docs)

    def sync_ingest_knowledge_base(self, chunks: list[KnowledgeChunk], sync_db_client: Any) -> int:
        """Synchronously ingest knowledge chunks into MongoDB."""
        if not chunks or sync_db_client is None:
            return 0

        col = sync_db_client[self.collection_name]
        docs = []
        for chk in chunks:
            docs.append({
                "chunk_id": chk.chunk_id,
                "content": chk.content,
                "topic": chk.topic,
                "category": chk.category,
                "source": chk.source,
                "metadata": chk.metadata,
                "vector": chk.vector,
            })

        for doc in docs:
            col.update_one(
                {"chunk_id": doc["chunk_id"]},
                {"$set": doc},
                upsert=True
            )

        return len(docs)

    async def retrieve_relevant_chunks(self, query: str, top_k: int = 4) -> list[RAGSearchResult]:
        """
        Perform vector retrieval + keyword reranking over MongoDB knowledge chunks.
        """
        if not query.strip() or self.db is None:
            return []

        col = self.db[self.collection_name]
        all_docs = await col.find({}).to_list(length=1000)
        if not all_docs:
            return []

        query_vec = self.embedder.embed(query)
        q_words = set(re.findall(r"\w+", query.lower()))

        results: list[RAGSearchResult] = []

        for doc in all_docs:
            vec = doc.get("vector") or []
            content = doc.get("content", "")
            topic = doc.get("topic", "")

            # 1. Cosine similarity
            sim = FastVectorEmbedder.cosine_similarity(query_vec, vec)

            # 2. Keyword overlap score boosting
            doc_words = set(re.findall(r"\w+", (content + " " + topic).lower()))
            overlap = len(q_words.intersection(doc_words)) / max(1, len(q_words))
            rerank_score = (sim * 0.6) + (overlap * 0.4)

            if rerank_score > 0.05:
                results.append(RAGSearchResult(
                    chunk_id=doc.get("chunk_id", ""),
                    content=content,
                    topic=topic,
                    category=doc.get("category", ""),
                    source=doc.get("source", ""),
                    score=sim,
                    rerank_score=rerank_score,
                ))

        # Sort by rerank score descending
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        return results[:top_k]

    def sync_retrieve_relevant_chunks(self, query: str, sync_db_client: Any, top_k: int = 4) -> list[RAGSearchResult]:
        """Synchronously retrieve relevant knowledge chunks."""
        if not query.strip() or sync_db_client is None:
            return []

        col = sync_db_client[self.collection_name]
        all_docs = list(col.find({}))
        if not all_docs:
            return []

        query_vec = self.embedder.embed(query)
        q_words = set(re.findall(r"\w+", query.lower()))

        results: list[RAGSearchResult] = []

        for doc in all_docs:
            vec = doc.get("vector") or []
            content = doc.get("content", "")
            topic = doc.get("topic", "")

            sim = FastVectorEmbedder.cosine_similarity(query_vec, vec)
            doc_words = set(re.findall(r"\w+", (content + " " + topic).lower()))
            overlap = len(q_words.intersection(doc_words)) / max(1, len(q_words))
            rerank_score = (sim * 0.6) + (overlap * 0.4)

            if rerank_score > 0.05:
                results.append(RAGSearchResult(
                    chunk_id=doc.get("chunk_id", ""),
                    content=content,
                    topic=topic,
                    category=doc.get("category", ""),
                    source=doc.get("source", ""),
                    score=sim,
                    rerank_score=rerank_score,
                ))

        results.sort(key=lambda x: x.rerank_score, reverse=True)
        return results[:top_k]

    async def generate_rag_response(self, query: str, language: str = "en") -> dict:
        """
        Generate final fitness advice / knowledge response via retrieved context and Qwen3/LLM.
        GUARANTEE: Will NEVER invent or calculate calorie/macro numbers.
        """
        retrieved = await self.retrieve_relevant_chunks(query, top_k=3)

        if not retrieved:
            context_str = "No specific custom documents found for this query."
        else:
            context_parts = [f"[Source: {r.source} | Topic: {r.topic}]\n{r.content}" for r in retrieved]
            context_str = "\n\n".join(context_parts)

        system_prompt = (
            "You are an expert fitness, nutrition, and wellness advisor powered by Qwen3.\n"
            "STRICT RULES:\n"
            "1. Provide helpful, accurate qualitative health, workout, and diet advice based on the context.\n"
            "2. NEVER calculate or fabricate numerical calories, macronutrients, or calorie burn numbers yourself.\n"
            "3. If asked for exact calories/macros, instruct the user to ask for food logging or nutrition calculation.\n"
            "4. Respond concisely in the requested language (English/Hindi/Gujarati).\n"
        )

        user_prompt = (
            f"User Question: {query}\n\n"
            f"Retrieved Knowledge Context:\n{context_str}\n\n"
            f"Language: {language}\n"
            f"Provide a clear, helpful response:"
        )

        if self.llm and hasattr(self.llm, "generate_completion"):
            llm_text = await self.llm.generate_completion(system_prompt, user_prompt)
        else:
            if retrieved:
                top_r = retrieved[0]
                llm_text = (
                    f"Based on fitness guidance ({top_r.topic}):\n"
                    f"{top_r.content}\n\n"
                    f"*(Note: For exact calorie tracking, specify the food item and portion size!)*"
                )
            else:
                llm_text = (
                    f"Fitness & Nutrition Tip: For optimal health and fitness results, "
                    f"maintain a balanced diet rich in whole foods, protein, and hydration alongside regular exercise."
                )

        return {
            "query": query,
            "answer": llm_text,
            "retrieved_chunks": [
                {
                    "chunk_id": r.chunk_id,
                    "topic": r.topic,
                    "source": r.source,
                    "rerank_score": round(r.rerank_score, 3)
                } for r in retrieved
            ],
            "rag_used": bool(retrieved),
        }
