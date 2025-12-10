from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class GraphStorePort(ABC):
    """Abstract interface for Graph Databases."""

    @abstractmethod
    def connect(self, **kwargs):
        """Establishes connection to the graph database."""
        pass

    @abstractmethod
    def add_node(self, label: str, properties: Dict[str, Any]):
        """Adds a node to the graph."""
        pass

    @abstractmethod
    def add_edge(self, source_node_id: str, target_node_id: str, relation_type: str, properties: Dict[str, Any] = None):
        """Adds an edge between two nodes."""
        pass

    @abstractmethod
    def query(self, cypher_query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Executes a Cypher query."""
        pass

    @abstractmethod
    def get_subgraph(self, node_id: str, depth: int = 1) -> Any:
        # Return type Any for now, ideally strictly typed or a specific Graph structure
        """Retrieves a subgraph around a specific node."""
        pass
