"""
Weaviate vector store implementation
"""

from typing import List, Dict, Optional
import uuid
import weaviate
from weaviate.classes.config import Configure, Property, DataType

from .base import BaseVectorStore
from ..utils.config import get_config


class WeaviateStore(BaseVectorStore):
    """Weaviate vector store implementation"""

    def __init__(self, collection_name: str, dimension: int):
        """
        Initialize Weaviate store

        Parameters:
        -----------
        collection_name : str
            Name of the collection (class)
        dimension : int
            Dimension of embeddings
        """
        super().__init__(collection_name, dimension)

        # Capitalize collection name for Weaviate (class names must start with uppercase)
        self.class_name = self.collection_name.capitalize()

        # Load configuration
        config = get_config()

        # Initialize Weaviate client
        try:
            if config.weaviate_api_key:
                self.client = weaviate.connect_to_custom(
                    http_host=config.weaviate_url.replace('http://', '').replace('https://', ''),
                    http_port=8080,
                    http_secure=False,
                    grpc_host=config.weaviate_url.replace('http://', '').replace('https://', ''),
                    grpc_port=50051,
                    grpc_secure=False,
                    auth_credentials=weaviate.auth.AuthApiKey(config.weaviate_api_key)
                )
            else:
                # For local Weaviate instance
                self.client = weaviate.connect_to_local(
                    host=config.weaviate_url.replace('http://', '').replace('https://', '').split(':')[0],
                    port=int(config.weaviate_url.split(':')[-1]) if ':' in config.weaviate_url else 8080
                )
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Weaviate client: {str(e)}")

    def create_collection(self) -> bool:
        """
        Create collection (class) if it doesn't exist

        Returns:
        --------
        bool
            True if created, False if already exists
        """
        try:
            # Check if class exists
            if self.client.collections.exists(self.class_name):
                return False

            # Create class
            self.client.collections.create(
                name=self.class_name,
                properties=[
                    Property(
                        name="text",
                        data_type=DataType.TEXT
                    ),
                    Property(
                        name="metadata",
                        data_type=DataType.OBJECT
                    )
                ],
                vectorizer_config=Configure.Vectorizer.none(),
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=weaviate.classes.config.VectorDistances.COSINE
                )
            )
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to create collection: {str(e)}")

    def insert(self, texts: List[str], embeddings: List[List[float]],
               metadata: Optional[List[Dict]] = None, ids: Optional[List[str]] = None) -> bool:
        """
        Insert texts with embeddings

        Parameters:
        -----------
        texts : List[str]
            List of text chunks
        embeddings : List[List[float]]
            List of embeddings
        metadata : List[Dict], optional
            List of metadata dictionaries
        ids : List[str], optional
            List of IDs (auto-generated if None)

        Returns:
        --------
        bool
            True if successful
        """
        try:
            # Get collection
            collection = self.client.collections.get(self.class_name)

            # Prepare metadata
            if metadata is None:
                metadata = [{} for _ in range(len(texts))]

            # Insert objects
            with collection.batch.dynamic() as batch:
                for i in range(len(texts)):
                    obj_id = ids[i] if ids and i < len(ids) else str(uuid.uuid4())

                    batch.add_object(
                        properties={
                            "text": texts[i],
                            "metadata": metadata[i]
                        },
                        vector=embeddings[i],
                        uuid=obj_id
                    )

            return True
        except Exception as e:
            raise RuntimeError(f"Failed to insert documents: {str(e)}")

    def search(self, query_embedding: List[float], top_k: int = 10,
               filters: Optional[Dict] = None) -> List[Dict]:
        """
        Search for similar vectors

        Parameters:
        -----------
        query_embedding : List[float]
            Query embedding
        top_k : int
            Number of results to return
        filters : Dict, optional
            Metadata filters

        Returns:
        --------
        List[Dict]
            List of results with text, metadata, and score
        """
        try:
            # Get collection
            collection = self.client.collections.get(self.class_name)

            # Build filter if provided
            where_filter = None
            if filters:
                # Build Weaviate filter
                # Note: Complex filters may need more sophisticated logic
                conditions = []
                for key, value in filters.items():
                    conditions.append({
                        "path": ["metadata", key],
                        "operator": "Equal",
                        "valueText": str(value) if not isinstance(value, str) else value
                    })

                if len(conditions) == 1:
                    where_filter = conditions[0]
                elif len(conditions) > 1:
                    where_filter = {
                        "operator": "And",
                        "operands": conditions
                    }

            # Perform search
            if where_filter:
                response = collection.query.near_vector(
                    near_vector=query_embedding,
                    limit=top_k,
                    where=where_filter,
                    return_metadata=weaviate.classes.query.MetadataQuery(distance=True)
                )
            else:
                response = collection.query.near_vector(
                    near_vector=query_embedding,
                    limit=top_k,
                    return_metadata=weaviate.classes.query.MetadataQuery(distance=True)
                )

            # Format results
            formatted_results = []
            for obj in response.objects:
                # Convert distance to similarity score (1 - distance for cosine)
                similarity = 1 - obj.metadata.distance if obj.metadata.distance else 0.0

                formatted_results.append({
                    'text': obj.properties.get('text', ''),
                    'metadata': obj.properties.get('metadata', {}),
                    'score': similarity,
                    'id': str(obj.uuid)
                })

            return formatted_results
        except Exception as e:
            raise RuntimeError(f"Failed to search: {str(e)}")

    def delete_collection(self) -> bool:
        """
        Delete collection (class)

        Returns:
        --------
        bool
            True if deleted
        """
        try:
            self.client.collections.delete(self.class_name)
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to delete collection: {str(e)}")

    def get_count(self) -> int:
        """
        Get number of vectors in collection

        Returns:
        --------
        int
            Number of vectors
        """
        try:
            collection = self.client.collections.get(self.class_name)
            aggregate = collection.aggregate.over_all(total_count=True)
            return aggregate.total_count if aggregate.total_count else 0
        except Exception as e:
            # Collection might not exist
            return 0

    def __del__(self):
        """Close connection on deletion"""
        if hasattr(self, 'client') and self.client:
            self.client.close()
