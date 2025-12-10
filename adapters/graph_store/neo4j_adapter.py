from typing import List, Dict, Any, Optional
import uuid

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None

from polyrag.core.ports.graph_store_port import GraphStorePort


class Neo4jAdapter(GraphStorePort):
    """Adapter for Neo4j graph database."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j"
    ):
        """
        Initialize the Neo4j adapter.

        Args:
            uri: Neo4j connection URI.
            user: Username for authentication.
            password: Password for authentication.
            database: Database name.

        Raises:
            ImportError: If neo4j driver is not installed.
        """
        if GraphDatabase is None:
            raise ImportError("neo4j is required. Install with: pip install neo4j")
        
        self._uri = uri
        self._user = user
        self._password = password
        self._database = database
        self._driver = None

    def connect(self, **kwargs):
        """Establishes connection to Neo4j."""
        self._driver = GraphDatabase.driver(
            self._uri,
            auth=(self._user, self._password)
        )
        # Verify connectivity
        self._driver.verify_connectivity()

    def _ensure_connected(self):
        """Ensure driver is connected."""
        if self._driver is None:
            self.connect()

    def add_node(self, label: str, properties: Dict[str, Any]) -> str:
        """
        Adds a node to the graph.

        Args:
            label: Node label (e.g., "Document", "Chunk", "Entity").
            properties: Node properties.

        Returns:
            The node ID.
        """
        self._ensure_connected()
        
        # Generate ID if not provided
        if "id" not in properties:
            properties["id"] = str(uuid.uuid4())
        
        query = f"""
        CREATE (n:{label} $props)
        RETURN n.id as id
        """
        
        with self._driver.session(database=self._database) as session:
            result = session.run(query, props=properties)
            record = result.single()
            return record["id"]

    def add_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        relation_type: str,
        properties: Dict[str, Any] = None
    ):
        """
        Adds an edge between two nodes.

        Args:
            source_node_id: Source node ID.
            target_node_id: Target node ID.
            relation_type: Relationship type (e.g., "CONTAINS", "RELATED_TO").
            properties: Relationship properties.
        """
        self._ensure_connected()
        properties = properties or {}
        
        query = f"""
        MATCH (a {{id: $source_id}}), (b {{id: $target_id}})
        CREATE (a)-[r:{relation_type} $props]->(b)
        RETURN r
        """
        
        with self._driver.session(database=self._database) as session:
            session.run(
                query,
                source_id=source_node_id,
                target_id=target_node_id,
                props=properties
            )

    def query(self, cypher_query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Executes a Cypher query.

        Args:
            cypher_query: The Cypher query string.
            parameters: Query parameters.

        Returns:
            List of result records as dictionaries.
        """
        self._ensure_connected()
        parameters = parameters or {}
        
        with self._driver.session(database=self._database) as session:
            result = session.run(cypher_query, **parameters)
            return [dict(record) for record in result]

    def get_subgraph(self, node_id: str, depth: int = 1) -> Dict[str, Any]:
        """
        Retrieves a subgraph around a specific node.

        Args:
            node_id: The central node ID.
            depth: How many hops to traverse.

        Returns:
            Dictionary with 'nodes' and 'relationships' lists.
        """
        self._ensure_connected()
        
        query = f"""
        MATCH path = (n {{id: $node_id}})-[*0..{depth}]-(m)
        RETURN collect(distinct n) + collect(distinct m) as nodes,
               [r in relationships(path) | {{
                   type: type(r),
                   start: startNode(r).id,
                   end: endNode(r).id,
                   properties: properties(r)
               }}] as relationships
        """
        
        with self._driver.session(database=self._database) as session:
            result = session.run(query, node_id=node_id)
            record = result.single()
            
            if record is None:
                return {"nodes": [], "relationships": []}
            
            # Process nodes
            nodes = []
            for node in record["nodes"]:
                nodes.append({
                    "id": node.get("id"),
                    "labels": list(node.labels) if hasattr(node, 'labels') else [],
                    "properties": dict(node)
                })
            
            return {
                "nodes": nodes,
                "relationships": record["relationships"]
            }

    def close(self):
        """Close the driver connection."""
        if self._driver:
            self._driver.close()
            self._driver = None

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
