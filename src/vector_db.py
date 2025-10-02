import os
from qdrant_client import QdrantClient, models
from loguru import logger

class VectorDB:
    '''
    A class to create collections on Qdrant. Update the points to qdrant and perform vector searches.
    '''
    