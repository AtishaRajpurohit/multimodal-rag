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

        #return faces

    def embed_upload_to_qdrant(
        self,
        collection_name: str,
        labels: List[str]
        ):
        '''
        Embeds the faces and uploads them to Qdrant.
        '''
        #Calling the facial_detection_embedding function from detect.py
        detector = Facial_Detection(self.input_image_path)
        processed_image = detector.preprocess_image(resize=(512, 512))
        results = detector.facial_detection_embedding(img_array=processed_image)

        #Uploading to Qdrant
        vector_db = VectorDB()
        vector_db.upload_detected_faces_to_qdrant(
        collection_name=collection_name,
        detected_faces_list=results,
        image_path=self.input_image_path,
        labels=labels
        )

    # def reference_dataset_creation_main(
    #     self,
    #     input_image_path: str,
    #     output_image_path: str,
    #     collection_name: str,
    #     labels: List[str]
    #     ):

    #     #Creating a reference dataset instance
    #     reference_dataset = Reference_Dataset_Creation(input_image_path, output_image_path)

    #     #Detecting faces and storing the cropped faces as .jpg files
    #     reference_dataset.extract_detected_faces_and_save_as_jpg()
    #     logger.info(f"[Reference Dataset Creation] Creating a collection and uploading the faces to Qdrant...")

    #     #Embedding the faces and uploading them to Qdrant, along with the labels
    #     reference_dataset.embed_upload_to_qdrant(collection_name, labels)
    #     logger.info("[Reference Dataset Creation] Reference dataset creation completed successfully. Check http://localhost:6333/dashboard")

        

        
#Test the code!    
if __name__ == "__main__":
    """
    STEP 1: RUN FACE CROPPING
    -------------------------
    Detects and saves cropped faces for each image in input_image_paths.
    After this step, review the cropped faces and prepare one label list per image.
    """

    input_image_paths = [
        "data/query_images/IMG_8916.HEIC",
        "data/query_images/query.JPG"
    ]

    output_image_path = "data/reference_images_faces"
    collection_name = "reference_dataset_collection"

    # Step 1: Crop and save faces
    for input_image_path in input_image_paths:
        logger.info(f"Processing image for cropping: {input_image_path}")
        reference_dataset = Reference_Dataset_Creation(input_image_path, output_image_path)
        reference_dataset.extract_detected_faces_and_save_as_jpg()

    logger.info("All faces extracted and saved successfully.")
    logger.info("Please review cropped images and prepare label lists for each image.")

    """
    STEP 2: RUN EMBEDDING + UPLOAD
    ------------------------------
    After verifying cropped faces, create a list of label lists.
    Each inner list must correspond to the number and order of faces detected in that image.
    """

    # Example: one list of labels per image (in same order as input_image_paths)
    all_labels = [
        ["Raghav","Sonali","Vinayak","Tala","Avi","Olivier","Atisha","Matene"],     # Labels for first image
        ["Rajan Uncle", "Atisha", "Mom", "Rivaan"]         # Labels for second image
    ]

    #Step 2: Embed and upload each image with its corresponding labels
    for input_image_path, labels in zip(input_image_paths, all_labels):
        logger.info(f"Embedding and uploading faces for {input_image_path}")
        reference_dataset = Reference_Dataset_Creation(input_image_path, output_image_path)
        reference_dataset.embed_upload_to_qdrant(collection_name, labels)

    logger.info("All labeled faces embedded and uploaded to Qdrant successfully.")
    logger.info("Check http://localhost:6333/dashboard for the new collection.")

    logger.info("FINAL : Code works!")

'''For tomorrow : Make the reusability better, so reference dataset creation can be used for other images as well.'''




    