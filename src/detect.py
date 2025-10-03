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


    def preprocess_image(self,convert_to : str = "RGB", resize : tuple = None):
        #Learning Note : Only using self here, since we have instantiated self.image_path and that
        #is now a part of the class.
        '''
        Preprocess the image, check for format, convert format if needed.
        '''
        logger.info(f"[3] Preprocessing image: {self.image_path}")

        #Get image format - Convert to lowercase, split text and extract extension. Ignore former (img name)
        _, ext = os.path.splitext(self.image_path.lower())

        #Check if image format is supported
        if ext not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported image format: {ext}")
        
        if ext not in HEIC_FORMATS:
            self.image = cv2.imread(self.image_path).convert(convert_to).resize(resize)
            if self.image is None:
                raise ValueError(f"OpenCV failed to read image: {self.image_path}")
        
        #Converting to HEIC formats.        
        if ext in HEIC_FORMATS:
            # if not HEIC_SUPPORT:
            #     raise Exception("HEIC support not available. Please install pillow-heif: uv add pillow-heif")

            #Open the image using pillow-heif
            pil_image = Image.open(self.image_path).convert(convert_to).resize(resize)
            self.image = np.array(pil_image)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
            logger.info(f"[4] Successfully converted HEIC image: {self.image_path}")
        # LEARNING : Since image needs to be used just for this function, and not anytime for this class. Just use it ias a temp variable.
        return self.image
            
            
    def perform_facial_detection(self,detector_backend: str = "retinaface",model_name: str = "ArcFace"):
        '''
        CONFIRM : If retinaface has issues detecting, compatibility issues and ArcFace issues for images
        Perform facial detection, embedding and extracting facial crops.
        '''
        logger.info(f"[5] Performing facial detection on the image: {self.image_path}")
        results = DeepFace.represent(
            img_path = self.image,
            detector_backend = detector_backend,
            model_name = model_name,
            enforce_detection = True,
            align = True,
            #What does this do ? 
            normalization = "base"
        )
        logger.info("[6] Facial detection completed")
        output = []
        for res in results:
            embedding = res["embedding"]
            #Aligned cropped face as np.array
            face_crop = res["facial_area"]
            output.append({
                "embedding": embedding,
                "face_crop": face_crop
            })

        return output
        logger.info(f"[7] Extracted {len(output)} faces from {self.image_path}")
        logger.info(f"[8] Dimensions of the face crop embeddings: {len(output[0]['embedding'])}")



if __name__ == "__main__":
    image_path = "data/query_images/IMG_2237.HEIC"

    #Creating an instance
    detector = Facial_Detection(image_path)

    #Preprocessing
    processed_image = detector.preprocess_image(resize=(512, 512))

    #Facial detection
    results = detector.perform_facial_detection()

   

'''Code works!'''


