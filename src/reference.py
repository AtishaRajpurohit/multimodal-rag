from pathlib import Path
from typing import List, Dict
from loguru import logger
import os


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

        



    def preprocess_image(self):
        '''
        Preprocess the image, check for format, convert format if needed.
        '''
        pass

    def perform_facial_detection(self):
        '''
        Perform facial detection, embedding and extracting facial crops.
        '''
        pass
