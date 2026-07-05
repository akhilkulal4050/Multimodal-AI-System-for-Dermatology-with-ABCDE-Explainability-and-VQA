"""
Stage 7 RAG - Retriever.
DermatologyRAG: embeds query, retrieves top-k chunks from ChromaDB.
This is the interface used by the Question Router and Orchestrator.
"""
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class DermatologyRAG:
    """
    Semantic retrieval from the dermatology knowledge base.

    Usage:
        rag = DermatologyRAG()
        results = rag.retrieve("What causes melanoma?", top_k=3)
        # returns: list[{text, source, doc_id, score}]

        context = rag.retrieve_context_string("What causes melanoma?")
        # returns: single string, ready for prompt injection
    """

    def __init__(self, embedding_model=None, chroma_db_dir=None,
                 collection_name=None, top_k=None, similarity_threshold=None):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import rag_config as cfg
        self.top_k = top_k if top_k is not None else cfg.TOP_K
        self.sim_threshold = (similarity_threshold if similarity_threshold is not None
                              else cfg.SIMILARITY_THRESHOLD)

        print(f"Loading embedding model: {embedding_model or cfg.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(embedding_model or cfg.EMBEDDING_MODEL)

        db_dir   = Path(chroma_db_dir or cfg.CHROMA_DB_DIR)
        col_name = collection_name or cfg.CHROMA_COLLECTION_NAME
        client   = chromadb.PersistentClient(
            path=str(db_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        try:
            self.collection = client.get_collection(col_name)
            print(f"ChromaDB collection '{col_name}' loaded "
                  f"({self.collection.count():,} chunks).")
        except Exception as e:
            raise RuntimeError(
                f"ChromaDB collection '{col_name}' not found at {db_dir}. "
                f"Run build_rag_index.py first. (Original error: {e})"
            )

    def retrieve(self, question: str, top_k: int = None,
                 source_filter: str = None) -> list:
        """
        Returns top-k relevant chunks above similarity_threshold.

        Each chunk: {text, source, doc_id, score}
        score = cosine similarity in [0, 1] (higher = more relevant).
        Chunks below threshold are silently dropped.
        """
        k = top_k or self.top_k
        embedding = self.model.encode([question], normalize_embeddings=True).tolist()

        where = {'source': source_filter} if source_filter else None
        res = self.collection.query(
            query_embeddings=embedding,
            n_results=max(k * 2, k),
            where=where,
        )

        if not res['ids'] or not res['ids'][0]:
            return []

        out = []
        for doc, meta, dist in zip(res['documents'][0],
                                    res['metadatas'][0],
                                    res['distances'][0]):
            similarity = 1.0 - dist          # Chroma cosine: dist = 1 - sim
            if similarity < self.sim_threshold:
                continue
            out.append({
                'text':    doc,
                'source':  meta.get('source', 'unknown'),
                'doc_id':  meta.get('doc_id', ''),
                'score':   round(float(similarity), 4),
            })
            if len(out) >= k:
                break

        return out

    def retrieve_context_string(self, question: str, top_k: int = None,
                                 source_filter: str = None) -> str:
        """
        Convenience wrapper: returns chunks joined into a single context string.
        Drop this directly into the R-LLaVA or Llama prompt.
        """
        chunks = self.retrieve(question, top_k=top_k, source_filter=source_filter)
        if not chunks:
            return ''
        parts = []
        for c in chunks:
            parts.append(f"[{c['source']}] {c['text']}")
        return ' '.join(parts)

    def is_available(self) -> bool:
        """Check if the ChromaDB collection exists and has documents."""
        try:
            return self.collection.count() > 0
        except Exception:
            return False
