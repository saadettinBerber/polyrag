from typing import List
import numpy as np

from polyrag.core.ports.retriever_port import RetrieverPort
from polyrag.core.ports.vector_store_port import VectorStorePort
from polyrag.core.models.models import RetrievalResult, Chunk, ChunkType
from polyrag.adapters.embedding.colbert_adapter import ColBERTAdapter


class ColBERTRetriever(RetrieverPort):
    """
    Retriever using ColBERT's late interaction mechanism.
    
    Uses MaxSim scoring between query tokens and document tokens
    for more precise retrieval than single-vector approaches.
    """

    def __init__(
        self,
        colbert_adapter: ColBERTAdapter,
        vector_store: VectorStorePort,
        collection_name: str
    ):
        """
        Initialize the ColBERT retriever.

        Args:
            colbert_adapter: ColBERT embedding adapter.
            vector_store: Vector store for initial retrieval.
            collection_name: Collection name for retrieval.
        """
        self._colbert = colbert_adapter
        self._vector_store = vector_store
        self._collection_name = collection_name

    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[RetrievalResult]:
        """
        Retrieves relevant chunks using ColBERT late interaction.

        Args:
            query: The search query.
            limit: Maximum number of results.
            **kwargs: Additional parameters.

        Returns:
            List of RetrievalResult objects.
        """
        # Get query token embeddings
        query_tokens = self._colbert.embed_query_tokens(query)
        
        # Get pooled query embedding for initial retrieval
        query_vector = self._colbert.embed_text(query)
        
        # Initial retrieval with more candidates
        candidates = self._vector_store.search(
            collection_name=self._collection_name,
            query_vector=query_vector,
            limit=limit * 3  # Over-retrieve for re-ranking
        )
        
        # Re-rank using MaxSim
        reranked = []
        for result in candidates:
            doc_content = result.chunk.content
            doc_tokens = self._colbert.embed_doc_tokens(doc_content)
            
            # Compute MaxSim score
            maxsim_score = self._compute_maxsim(query_tokens, doc_tokens)
            
            reranked.append(RetrievalResult(
                chunk=result.chunk,
                score=maxsim_score,
                source="colbert",
                metadata={
                    **result.metadata,
                    "original_score": result.score,
                    "maxsim_score": maxsim_score
                }
            ))
        
        # Sort by MaxSim score
        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked[:limit]

    def _compute_maxsim(self, query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
        """
        Compute MaxSim score between query and document tokens.

        Args:
            query_tokens: Query token embeddings (Q x D).
            doc_tokens: Document token embeddings (T x D).

        Returns:
            MaxSim score.
        """
        # Compute similarity matrix
        sim_matrix = np.dot(query_tokens, doc_tokens.T)
        
        # MaxSim: max over doc tokens for each query token, then sum
        max_sims = sim_matrix.max(axis=1)
        return float(max_sims.sum())
