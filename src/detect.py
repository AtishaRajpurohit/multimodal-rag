from pathlib import Path
from typing import List, Dict, Optional, Tuple
from loguru import logger
import os
import cv2
from PIL import Image
import numpy as np
import setuptools.dist as distutils
from deepface import DeepFace
from pillow_heif import register_heif_opener

# Register HEIF opener
register_heif_opener()

# Constants
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.heic', '.heif', '.gif', '.ico'}
HEIC_FORMATS = {'.heic', '.heif'}
OPENCV_NATIVE_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


class FacialDetector:
    """
    A class for facial detection and embedding extraction from images.
    Handles various image formats including HEIC/HEIF.
    """
    
    def __init__(self, image_path: str):
        """
        Initialize the FacialDetector.
        
        Args:
            image_path: Path to the image file
        """
        self.image_path = image_path
        self._validate_image_path()
        logger.info(f"Initialized FacialDetector for: {self.image_path}")
    
    def _validate_image_path(self) -> None:
        """Validate the image path and format."""
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image file not found: {self.image_path}")
        
        if not isinstance(self.image_path, str):
            raise TypeError(f"Image path must be a string, got: {type(self.image_path)}")
        
        _, ext = os.path.splitext(self.image_path.lower())
        if ext not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported image format: {ext}")

    def preprocess_image(
        self,
        image_path: str = None,
        convert_to: str = "RGB",
        resize: tuple = None
    ) -> np.ndarray:
        """
        Preprocess the image, check for format, convert format if needed.
        
        Args:
            image_path: Path to image (uses self.image_path if None)
            convert_to: Color conversion format
            resize: Optional tuple (width, height) to resize image
            
        Returns:
            Preprocessed image as numpy array
        """
        image_path = image_path or self.image_path
        logger.info(f"Preprocessing image: {image_path}")

        # Get image format
        _, ext = os.path.splitext(image_path.lower())

        # Check if image format is supported
        if ext not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported image format: {ext}")

        if ext not in HEIC_FORMATS:
            # Process standard formats with OpenCV
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"OpenCV failed to read image: {image_path}")
            
            if resize:
                image = cv2.resize(image, resize)
            return image
        
        else:
            # Process HEIC/HEIF formats with PIL
            pil_image = Image.open(image_path).convert(convert_to)
            if resize:
                pil_image = pil_image.resize(resize)
            
            image = np.array(pil_image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            logger.info(f"Converted HEIC image: {image_path}")
            return image

    def facial_detection_embedding(
        self,
        detector_backend: str = "retinaface",
        model_name: str = "ArcFace",
        img_array: str = None
    ) -> List[Dict]:
        """
        Perform facial detection, embedding and extracting facial crops.
        
        Args:
            detector_backend: Face detection backend ("retinaface", "opencv", etc.)
            model_name: Face embedding model ("ArcFace", "Facenet", etc.)
            img_array: Optional preprocessed image array
            
        Returns:
            List of face detection results with embeddings
        """
        # Determine what to send to DeepFace
        img_input = img_array if img_array is not None else self.image_path
        logger.info(f"Performing facial detection on the image")
        
        results = DeepFace.represent(
            img_path=img_input,
            detector_backend=detector_backend,
            model_name=model_name,
            enforce_detection=True,
            align=True,
            normalization="ArcFace"
        )

        logger.info("Facial detection completed")
        return results


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    image_path = "data/query_images/IMG_2237.HEIC"

    try:
        # Step 1: Create instance (validation happens automatically)
        detector = FacialDetector(image_path)

        # Step 2: Preprocess image
        processed_image = detector.preprocess_image(resize=(512, 512))
        logger.info(f"Preprocessed image shape: {processed_image.shape}, dtype: {processed_image.dtype}")

        # Step 3: Generate embeddings
        results = detector.facial_detection_embedding(img_array=processed_image)
        logger.info(f"Detected {len(results)} face(s)")

        # Step 4: Inspect keys
        if results:
            logger.info(f"Keys in first result: {list(results[0].keys())}")

        logger.info("Code executed successfully.")

    except Exception as e:
        logger.error(f"Test failed: {e}")