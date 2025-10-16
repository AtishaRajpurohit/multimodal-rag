from pathlib import Path
from typing import List, Dict
from loguru import logger
import os
import cv2
from PIL import Image
import numpy as np
from deepface import DeepFace
from qdrant_client import QdrantClient, models


#Importing pillow-heif for HEIC support
from pillow_heif import register_heif_opener
register_heif_opener()

from detect import Facial_Detection
from vector_db import VectorDB

class Reference_Dataset_Creation:
    '''
    A class to create reference dataset of cropped faces and store them in an output directory, that is already created.
        '''
    def __init__(self,input_image_path: str,output_image_path: str):
        self.input_image_path = input_image_path
        self.output_image_path = output_image_path

        #Checking for input image path
        logger.info(f"[1] Checking for input image path: {self.input_image_path}")
        if not os.path.exists(self.input_image_path):
            raise FileNotFoundError(f"Image file not found in input image path: {self.input_image_path}")
        if type(self.input_image_path) != str:
            raise TypeError(f"Input Image path must be a string, current type: {type(self.input_image_path)}")
        logger.info(f"[2] {self.input_image_path} is valid")

        #Checking for output image path
        logger.info(f"[3] Checking for output image path: {self.output_image_path}")
        if not os.path.exists(self.output_image_path):
            raise FileNotFoundError(f"Image file not found in output image path: {self.output_image_path}")
        if type(self.output_image_path) != str:
            raise TypeError(f"Output Image path must be a string, current type: {type(self.output_image_path)}")
        logger.info(f"[4] {self.output_image_path} is valid")

    def extract_detected_faces_and_save_as_jpg(self):
        '''
        Extracts the detected faces and stores the cropped images in a jpg format in the output image path.
        
        deepface.extract_faces returns a list of dictionaries with the keys:
        "face": numpy array of the cropped face
        "facial_area": list of 4 integers, the coordinates of the face in the image
        "confidence": float, the confidence score of the face detection

        deepface.represent returns a list of dictionaries with the keys:

        "embedding": numpy array of the facial embedding
        "facial_area": list of 4 integers, the coordinates of the face in the image
        "face_confidence": float, the confidence score of the face detection
        '''
        #Calling the preprocessing function from detect.py
        detector = Facial_Detection(self.input_image_path)
        self.processed_image = detector.preprocess_image(resize=(512, 512))
                
        faces = DeepFace.extract_faces(
            img_path=self.processed_image,
            detector_backend="retinaface")
        logger.info(f"Keys in faces: {faces[0].keys()}")


        logger.info(f"[5] Extracted {len(faces)} faces from {self.input_image_path}")
        
        for i, face in enumerate(faces):
            # Saving faces by rescaling, converting RGB to BGR since something is happening in the processing function qith the HEIC format images.
            cropped_face = (face["face"] * 255).astype("uint8")  # DeepFace normalizes, so rescale back
            cropped_face_bgr = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)  # convert RGB → BGR
            cv2.imwrite(f"{self.output_image_path}/face_{i+1}_{self.input_image_path.split('/')[-1]}.jpg", cropped_face_bgr)

        logger.info(f"[6] Stored {len(faces)} faces in {self.output_image_path}")

    def embed_detected_faces(self):
        '''
        Embeds the faces detected using facial_detection_embedding function for a face.
        '''
        embedding = DeepFace.represent(
            detector_backend="retinaface",
            img_path = self.processed_image,
            model_name = "ArcFace",
            align = True,
            normalization = "ArcFace"
        )

        logger.info(f"[7] Detected {len(embedding)} faces in {self.input_image_path}")        




        #Create a collection and upload to the collection, call the functions from VectorDB.

        
        
#Test the code!    
if __name__ == "__main__":
    input_image_path = "data/query_images/IMG_8916.HEIC"
    output_image_path = "data/reference_images_faces"
    reference_dataset_creation = Reference_Dataset_Creation(input_image_path, output_image_path)
    reference_dataset_creation.extract_detected_faces_and_save_as_jpg()
    reference_dataset_creation.embed_detected_faces()
    
    #Uploading to Qdrant
    vector_db = VectorDB()

    logger.info(f"Creating a collection and uploading the faces to Qdrant")
    vector_db.upload_detected_faces_to_qdrant(
        collection_name="reference_dataset_collection",
        detected_faces_list=results,
        image_path=image_path,
        )

        

        
