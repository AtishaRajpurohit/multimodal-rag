#detect.py
from pathlib import Path
from typing import List, Dict
from loguru import logger
import os
import cv2
from PIL import Image
import numpy as np
from deepface import DeepFace

#Importing pillow-heif for HEIC support
from pillow_heif import register_heif_opener
register_heif_opener()

#vector_db.py
import os
from qdrant_client import QdrantClient, models
from loguru import logger

#Importing created modules
from detect import Facial_Detection
from vector_db import VectorDB


if __name__ == "__main__":
    logger.info("Starting the pipeline!")
    image_path = "data/query_images/IMG_2237.HEIC"

    #Creating an instance
    detector = Facial_Detection(image_path)

    #Preprocessing
    processed_image = detector.preprocess_image(resize=(512, 512))

    #Facial detection
    results = detector.perform_facial_detection()

    #Uploading to Qdrant
    vector_db = VectorDB()

    logger.info(f"Uploading faces to Qdrant")
    vector_db.upload_to_qdrant(collection_name="detected_faces_collection", detected_faces_list=results, image_path=image_path)

    logger.info("Pipeline completed successfully! :) Check http://localhost:6333/dashboard")
