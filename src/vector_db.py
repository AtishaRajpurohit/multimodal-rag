import os
from qdrant_client import QdrantClient, models
from loguru import logger
#What aret these? 
from typing import List, Dict

class VectorDB:
    '''
    A class to create collections on Qdrant. Update the points to qdrant and perform vector searches.
    '''

    def __init__(self):
        logger.info("[1] Initializing Qdrant")
        self.client = QdrantClient(url="http://localhost:6333")
        logger.info("[2] Connected to Qdrant at localhost:6333")
    
    def collection_exists(self, collection_name: str) -> bool:
        return self.client.collection_exists(collection_name)

    def create_collection(self, collection_name: str, vector_size: int, distance: str):

        if not isinstance(vector_size, int):
            raise ValueError("Vector size must be an integer")  
        
        #Check if distance is valid
        valid_distances = ["Cosine", "Euclidean", "Dot"]
        if distance not in valid_distances:
            raise ValueError(f"Invalid distance: {distance}. Must be one of: {valid_distances}")

        if self.collection_exists(collection_name):
            raise ValueError(f"Collection {collection_name} already exists")

        logger.info(f" [3] Creating collection {collection_name} with vector size {vector_size} and distance {distance}")

        self.client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_size,distance=distance)
        )
        logger.info(f" [4] Collection {collection_name} created successfully")

    def delete_collection(self, collection_name: str):
        if not self.collection_exists(collection_name):
            raise ValueError(f"Collection {collection_name} does not exist")
        self.client.delete_collection(collection_name=collection_name)
        logger.info(f" [5] Collection {collection_name} deleted successfully")


    def upload_to_qdrant(self, collection_name: str, points = List[models.PointStruct]):
        if not self.collection_exists(collection_name):
            raise ValueError(f"Collection {collection_name} does not exist")
        self.client.upsert(collection_name=collection_name, points=points)
        logger.info(f" [6] {len(points)} points uploaded to collection {collection_name}")


        
        

if __name__ == "__main__":
    vector_db = VectorDB()
    # vector_db.create_collection(collection_name="test",vector_size=512,distance="Cosine")
    # vector_db.delete_collection(collection_name="test")
    # exists = vector_db.collection_exists(collection_name="test")
    # logger.info(f"Collection exists: {exists}")


    '''For tomorrow, update the database to qdrant for detected faces. Make the reference collection!'''