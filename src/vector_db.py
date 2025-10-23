import os
from qdrant_client import QdrantClient, models
from loguru import logger
import uuid
from typing import List, Dict, Optional


class VectorDB:
    """
    A class for managing Qdrant vector database operations.
    Handles collection creation, deletion, and face data uploads.
    """
    
    def __init__(self, url: str = "http://localhost:6333"):
        """
        Initialize the VectorDB connection.
        
        Args:
            url: Qdrant server URL
        """
        logger.info("Initializing Qdrant connection")
        self.client = QdrantClient(url=url)
        self.valid_distances = ["Cosine", "Euclidean", "Dot"]
        logger.info(f"Connected to Qdrant at {url}")
    
    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists in Qdrant.
        
        Args:
            collection_name: Name of the collection to check
            
        Returns:
            True if collection exists, False otherwise
        """
        return self.client.collection_exists(collection_name)

    def create_collection(self, collection_name: str, vector_size: int, distance: str) -> None:
        """
        Create a new collection in Qdrant.
        
        Args:
            collection_name: Name of the collection to create
            vector_size: Size of the vectors to be stored
            distance: Distance metric ("Cosine", "Euclidean", or "Dot")
        """
        self._validate_collection_params(vector_size, distance)
        
        if self.collection_exists(collection_name):
            raise ValueError(f"Collection {collection_name} already exists")

        logger.info(f"Creating collection {collection_name} with vector size {vector_size} and distance {distance}")

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=distance)
        )
        logger.info(f"Collection {collection_name} created successfully")

    def delete_collection(self, collection_name: str) -> None:
        """
        Delete a collection from Qdrant.
        
        Args:
            collection_name: Name of the collection to delete
        """
        if not self.collection_exists(collection_name):
            raise ValueError(f"Collection {collection_name} does not exist")
        
        self.client.delete_collection(collection_name=collection_name)
        logger.info(f"Collection {collection_name} deleted successfully")

    def _validate_collection_params(self, vector_size: int, distance: str) -> None:
        """
        Validate collection creation parameters.
        
        Args:
            vector_size: Vector size to validate
            distance: Distance metric to validate
        """
        if not isinstance(vector_size, int):
            raise ValueError("Vector size must be an integer")
        
        if distance not in self.valid_distances:
            raise ValueError(f"Invalid distance: {distance}. Must be one of: {self.valid_distances}")

    def upload_detected_faces_to_qdrant(
        self,
        collection_name: str,
        detected_faces_list: List[Dict],
        image_path: str,
        distance: str = "Cosine",
        vector_size: int = 512,
        labels: List[str] = None
    ) -> None:
        """
        Upload detected faces to Qdrant collection.
        
        Args:
            collection_name: Name of the collection to upload to
            detected_faces_list: List of face detection results
            image_path: Path to the source image
            distance: Distance metric for collection (if creating new)
            vector_size: Vector size for collection (if creating new)
            labels: List of labels for faces (defaults to "Unknown" if not provided)
        """
        # Create collection if it doesn't exist
        if not self.collection_exists(collection_name):
            self.create_collection(collection_name, vector_size, distance)

        # Handle labels
        labels = self._prepare_labels(labels, len(detected_faces_list))

        # Convert results to points
        points = self._create_points(detected_faces_list, labels, image_path)

        # Upload to Qdrant
        self.client.upsert(collection_name=collection_name, points=points)
        logger.info(f"{len(points)} points uploaded to collection {collection_name}")

    def _prepare_labels(self, labels: Optional[List[str]], num_faces: int) -> List[str]:
        """
        Prepare labels for faces, filling with "Unknown" if needed.
        
        Args:
            labels: Provided labels list
            num_faces: Number of faces detected
            
        Returns:
            Complete labels list
        """
        if not labels:
            return ["Unknown"] * num_faces
        elif len(labels) < num_faces:
            labels.extend(["Unknown"] * (num_faces - len(labels)))
        return labels

    def _create_points(self, detected_faces_list: List[Dict], labels: List[str], image_path: str) -> List[models.PointStruct]:
        """
        Create Qdrant point structures from face data.
        
        Args:
            detected_faces_list: List of face detection results
            labels: Labels for each face
            image_path: Path to the source image
            
        Returns:
            List of Qdrant PointStruct objects
        """
        points = []
        for face_data, label in zip(detected_faces_list, labels):
            point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector=face_data["embedding"],
                payload={
                    "facial_area": face_data["facial_area"],
                    "face_confidence": face_data["face_confidence"],
                    "image_path": image_path,
                    "label": label
                }
            )
            points.append(point)
        return points


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    vector_db = VectorDB()
    
    try:
        # Create a dummy collection
        collection_name = "dummy_test_collection"
        logger.info(f"Creating dummy collection: {collection_name}")
        vector_db.create_collection(collection_name=collection_name, vector_size=512, distance="Cosine")
        
        # Create dummy face data
        dummy_faces = [
            {
                "embedding": [0.1] * 512,  # Dummy 512-dimensional vector
                "facial_area": {"x": 100, "y": 50, "w": 200, "h": 250},
                "face_confidence": 0.95
            },
            {
                "embedding": [0.2] * 512,  # Another dummy vector
                "facial_area": {"x": 300, "y": 80, "w": 180, "h": 220},
                "face_confidence": 0.92
            }
        ]
        
        # Add points to the collection
        logger.info("Adding dummy points to collection")
        vector_db.upload_detected_faces_to_qdrant(
            collection_name=collection_name,
            detected_faces_list=dummy_faces,
            image_path="dummy_image.jpg",
            labels=["PersonA", "PersonB"]
        )
        
        # Verify collection exists
        exists = vector_db.collection_exists(collection_name=collection_name)
        logger.info(f"Collection exists after upload: {exists}")
        
        # Delete the collection
        logger.info(f"Deleting collection: {collection_name}")
        vector_db.delete_collection(collection_name=collection_name)
        
        # Verify collection is deleted
        exists_after_delete = vector_db.collection_exists(collection_name=collection_name)
        logger.info(f"Collection exists after deletion: {exists_after_delete}")
        
        logger.info("VectorDB example completed successfully")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")