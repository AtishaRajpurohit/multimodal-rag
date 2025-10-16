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

#Supported image formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif','.webp', '.heic', '.heif', '.gif', '.ico'}

# HEIC/HEIF formats that need special handling
HEIC_FORMATS = {'.heic', '.heif'}

# Formats that OpenCV can read directly
OPENCV_NATIVE_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


class Facial_Detection:
    '''
    A class to preprocess uploaded images, check for format, convert format if needed.
    Perform facial detection, embedding and extracting facial crops.
    Usage : 
    '''
    def __init__(self,image_path: str):
        self.image_path = image_path
        #Checking for image path
        logger.info(f"[1] Checking for image path: {self.image_path}")
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image file not found: {self.image_path}")
        if type(self.image_path) != str:
            raise TypeError(f"Image path must be a string, current type: {type(self.image_path)}")
        logger.info(f"[2] {self.image_path} is valid")


    def preprocess_image(
        self,
        image_path: str = None,
        convert_to : str = "RGB",
        resize : tuple = None):

        '''
        Preprocess the image, check for format, convert format if needed.
        '''
        image_path = image_path or self.image_path
        logger.info(f"[3] Preprocessing image: {self.image_path}")

        #Get image format - Convert to lowercase, split text and extract extension. Ignore former (img name)
        _, ext = os.path.splitext(self.image_path.lower())

        #Check if image format is supported
        if ext not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported image format: {ext}")
        
        # if ext not in HEIC_FORMATS:
        #     self.image = cv2.imread(self.image_path).convert(convert_to).resize(resize)
        #     if self.image is None:
        #         raise ValueError(f"OpenCV failed to read image: {self.image_path}")

        if ext not in HEIC_FORMATS:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"OpenCV failed to read image: {image_path}")
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Converts BGR to RGB
            if resize:
                image = cv2.resize(image, resize)
            return image #RGB
        
        #Converting to HEIC formats.        
        else:
            pil_image = Image.open(image_path).convert(convert_to).resize(resize) #RGB
            if resize:
                pil_image = pil_image.resize(resize)
            image = np.array(pil_image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #BGR
            logger.info(f"[4] Converted HEIC image: {image_path}")
        # LEARNING : Since image needs to be used just for this function, and not anytime for this class. Just use it ias a temp variable.
        return image #BGR
            
    # To be applied to query image in camera_image_matching.py        
    def facial_detection_embedding(
        self,
        detector_backend: str = "retinaface",
        model_name: str = "ArcFace",
        img_array: str = None
        ):

        '''
        Takes in the image, that is the output of preprocess_image
        CONFIRM : If retinaface has issues detecting, compatibility issues and ArcFace issues for images
        Perform facial detection, embedding and extracting facial crops.

        '''
        # Determine what to send to DeepFace
        img_input = img_array if img_array is not None else self.image_path
        logger.info(f"[5] Performing facial detection on the image")
        
        results = DeepFace.represent(

            img_path = img_input,            
            detector_backend = detector_backend,
            model_name = model_name,
            enforce_detection = True,
            align = True,
            #What does this do ? 
            normalization = "ArcFace"
        )

        logger.info("[6] Facial detection completed")
        return results


if __name__ == "__main__":
    image_path = "data/query_images/IMG_2237.HEIC"

    try:
        # Step 1: Create instance
        detector = Facial_Detection(image_path)

        # Step 2: Preprocess image
        processed_image = detector.preprocess_image(resize=(512, 512))
        logger.info(f"[A] Preprocessed image shape: {processed_image.shape}, dtype: {processed_image.dtype}")

        # Step 3: Generate embeddings
        results = detector.facial_detection_embedding(img_array=processed_image)
        logger.info(f"[B]Detected {len(results)} face(s)")

        # Step 4: Inspect keys
        logger.info(f"[C] Keys in first result: {list(results[0].keys())}")

        logger.info("[D] Code executed successfully.")

    except Exception as e:
        logger.error(f"[E] Test failed: {e}")



