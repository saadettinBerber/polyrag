from typing import List

from polyrag.core.ports.retriever_port import RetrieverPort
from polyrag.core.ports.graph_store_port import GraphStorePort
from polyrag.core.ports.embedding_port import TextEmbeddingPort
from polyrag.core.models.models import RetrievalResult, Chunk, ChunkType


class GraphRetriever(RetrieverPort):
    """
    Retriever that uses graph traversal to find relevant context.
    
    First finds relevant nodes via text matching or embedding,
    then expands to subgraph for richer context.
    """

    def __init__(
        self,
        graph_store: GraphStorePort,
        embedding_adapter: TextEmbeddingPort = None,
        node_label: str = "Chunk",
        depth: int = 1
    ):
        """
        Initialize the GraphRetriever.

        Args:
            graph_store: Graph store adapter.
            embedding_adapter: Optional embedding adapter for semantic search.
            node_label: Label of nodes to search.
            depth: Depth of subgraph expansion.
        """
        self._graph_store = graph_store
        self._embedding = embedding_adapter
        self._node_label = node_label
        self._depth = depth

    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[RetrievalResult]:
        """
        Retrieves relevant chunks from the graph.

        Args:
            query: The search query.
            limit: Maximum number of results.
            **kwargs: Additional parameters.

        Returns:
            List of RetrievalResult objects with graph context.
        """
        # Search for matching nodes using text search
        cypher = f"""
        MATCH (n:{self._node_label})
        WHERE toLower(n.content) CONTAINS toLower($query)
        RETURN n
        LIMIT $limit
        """
        
        nodes = self._graph_store.query(cypher, {"query": query, "limit": limit})
        
        results = []
        for record in nodes:
            node = record.get("n", {})
            node_id = node.get("id")
            
            # Expand subgraph for context
            subgraph = None
            if node_id:
                subgraph = self._graph_store.get_subgraph(node_id, depth=self._depth)
            
            # Build context from subgraph
            context = self._build_context_from_subgraph(node, subgraph)
            
            chunk = Chunk(
                content=context,
                chunk_type=ChunkType.TEXT,
                source_document_id=node.get("source_document_id", "graph"),
                id=node_id or "",
                metadata={
                    "original_content": node.get("content", ""),
                    "subgraph_nodes": len(subgraph["nodes"]) if subgraph else 0,
                    "subgraph_relations": len(subgraph["relationships"]) if subgraph else 0
                }
            )
            
            results.append(RetrievalResult(
                chunk=chunk,
                score=1.0,  # Graph traversal doesn't produce scores
                source="graph",
                metadata={"subgraph": subgraph}
            ))
        
        return results

    def _build_context_from_subgraph(self, center_node: dict, subgraph: dict) -> str:
        """
        Builds a text context from subgraph data.

        Args:
            center_node: The central node.
            subgraph: Subgraph data with nodes and relationships.

        Returns:
            Formatted context string.
        """
        if not subgraph:
            return center_node.get("content", "")
        
        parts = [f"Main: {center_node.get('content', '')}"]
        
        # Add related node contents
        for node in subgraph.get("nodes", []):
            if node.get("id") != center_node.get("id"):
                content = node.get("properties", {}).get("content", "")
                if content:
                    parts.append(f"Related: {content[:200]}...")
        
        # Add relationship info
        for rel in subgraph.get("relationships", []):
            parts.append(f"[{rel.get('type', 'RELATED')}]")
        
        return "\n".join(parts)
