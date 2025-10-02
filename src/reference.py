from pathlib import Path
from typing import List, Dict
from loguru import logger
import os
import cv2
from PIL import Image
import numpy as np

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
        logger.info(f"Checking for image path: {self.image_path}")
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image file not found: {self.image_path}")
        if type(self.image_path) != str:
            raise TypeError(f"Image path must be a string, current type: {type(self.image_path)}")
        logger.info(f"{self.image_path} is valid")


    def preprocess_image(self,convert_to : str = "RGB", resize : tuple = None):
        #Learning Note : Only using self here, since we have instantiated self.image_path and that
        #is now a part of the class.
        '''
        Preprocess the image, check for format, convert format if needed.
        '''
        logger.info(f"Preprocessing image: {self.image_path}")

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
            pil_image = Image.open(self.image_path).convert(convert_to).resize(resize)
            self.image = np.array(pil_image)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
            logger.info(f"Successfully converted HEIC image: {self.image_path}")
        # LEARNING : Since image needs to be used just for this function, and not anytime for this class. Just use it ias a temp variable.
        return self.image
            
            
    def perform_facial_detection(self):
        '''
        Perform facial detection, embedding and extracting facial crops.
        '''
        pass
