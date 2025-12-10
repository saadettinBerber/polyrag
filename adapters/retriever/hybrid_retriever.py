from typing import List, Optional

from polyrag.core.ports.retriever_port import RetrieverPort
from polyrag.core.models.models import RetrievalResult


class HybridRetriever(RetrieverPort):
    """
    Combines multiple retrievers with weighted scoring.
    
    Useful for combining vector search with graph traversal
    to get both semantic similarity and structural context.
    """

    def __init__(
        self,
        retrievers: List[RetrieverPort],
        weights: Optional[List[float]] = None
    ):
        """
        Initialize the HybridRetriever.

        Args:
            retrievers: List of retriever adapters to combine.
            weights: Optional weights for each retriever (should sum to 1.0).
                     If not provided, equal weights are used.
        """
        self._retrievers = retrievers
        
        if weights is None:
            self._weights = [1.0 / len(retrievers)] * len(retrievers)
        else:
            if len(weights) != len(retrievers):
                raise ValueError("Number of weights must match number of retrievers")
            # Normalize weights
            total = sum(weights)
            self._weights = [w / total for w in weights]

    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[RetrievalResult]:
        """
        Retrieves and combines results from all retrievers.

        Args:
            query: The search query.
            limit: Maximum number of final results.
            **kwargs: Additional parameters passed to each retriever.

        Returns:
            Combined and re-ranked list of RetrievalResult objects.
        """
        # Collect results from all retrievers
        all_results: dict[str, RetrievalResult] = {}
        
        for retriever, weight in zip(self._retrievers, self._weights):
            results = retriever.retrieve(query, limit=limit * 2, **kwargs)
            
            for result in results:
                chunk_id = result.chunk.id
                weighted_score = result.score * weight
                
                if chunk_id in all_results:
                    # Combine scores
                    existing = all_results[chunk_id]
                    combined_score = existing.score + weighted_score
                    all_results[chunk_id] = RetrievalResult(
                        chunk=existing.chunk,
                        score=combined_score,
                        source="hybrid",
                        metadata={
                            **existing.metadata,
                            f"{result.source}_score": result.score
                        }
                    )
                else:
                    all_results[chunk_id] = RetrievalResult(
                        chunk=result.chunk,
                        score=weighted_score,
                        source="hybrid",
                        metadata={
                            **result.metadata,
                            f"{result.source}_score": result.score
                        }
                    )
        
        # Sort by combined score and limit
        sorted_results = sorted(
            all_results.values(),
            key=lambda r: r.score,
            reverse=True
        )
        
        return sorted_results[:limit]
