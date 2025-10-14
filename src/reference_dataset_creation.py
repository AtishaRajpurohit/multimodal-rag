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

from detect import Facial_Detection

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
        '''
        #Calling the preprocessing function from detect.py
        detector = Facial_Detection(self.input_image_path)
        processed_image = detector.preprocess_image(resize=(512, 512))
        
        faces = DeepFace.extract_faces(img_path=processed_image, detector_backend="retinaface")
        logger.info(f"[5] Extracted {len(faces)} faces from {self.input_image_path}")
        
        # faces is a list of dicts, each with a cropped face
        for i, face in enumerate(faces):

            # 'face' contains a numpy array of the cropped face
            cropped_face = (face["face"] * 255).astype("uint8")  # DeepFace normalizes, so rescale back
            cropped_face_bgr = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)  # convert RGB → BGR

            cv2.imwrite(f"{self.output_image_path}/face_{i+1}_{self.input_image_path.split('/')[-1]}.jpg", cropped_face_bgr)
        logger.info(f"[6] Stored {len(faces)} faces in {self.output_image_path}")


#Test the code!    
if __name__ == "__main__":
    input_image_path =["data/ref_images/IMG_2872.HEIC","data/ref_images/FullSizeRender.HEIC"]

    output_image_path = "data/reference_images_faces"
    reference_dataset_creation = Reference_Dataset_Creation(input_image_path,output_image_path)
    reference_dataset_creation.extract_detected_faces_and_save_as_jpg()
