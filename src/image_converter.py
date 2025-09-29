"""
Robust image format detection and conversion utility.

This module provides functions to detect image formats and convert them
to OpenCV format (BGR numpy array) for consistent processing.
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import Union, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import pillow-heif for HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORT = True
    logger.info("HEIC support enabled via pillow-heif")
except ImportError:
    HEIC_SUPPORT = False
    logger.warning("pillow-heif not available. HEIC files will not be supported.")

# Supported image formats
SUPPORTED_FORMATS = {
    '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', 
    '.webp', '.heic', '.heif', '.gif', '.ico'
}

# HEIC/HEIF formats that need special handling
HEIC_FORMATS = {'.heic', '.heif'}

# Formats that OpenCV can read directly
OPENCV_NATIVE_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def detect_image_format(image_path: str) -> str:
    """
    Detect the image format from file extension.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Lowercase file extension including the dot (e.g., '.jpg', '.heic')
        
    Raises:
        ValueError: If the file format is not supported
        FileNotFoundError: If the file doesn't exist
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Get file extension
    _, ext = os.path.splitext(image_path.lower())
    
    if not ext:
        raise ValueError(f"No file extension found for: {image_path}")
    
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported image format: {ext}. Supported formats: {SUPPORTED_FORMATS}")
    
    return ext


def convert_heic_to_cv2(image_path: str) -> np.ndarray:
    """
    Convert HEIC/HEIF image to OpenCV format.
    
    Args:
        image_path (str): Path to the HEIC/HEIF image file
        
    Returns:
        np.ndarray: Image in BGR format (OpenCV standard)
        
    Raises:
        Exception: If conversion fails or HEIC support is not available
    """
    if not HEIC_SUPPORT:
        raise Exception("HEIC support not available. Please install pillow-heif: pip install pillow-heif")
    
    try:
        # Open HEIC image with PIL (now with HEIC support)
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert PIL image to numpy array
            img_array = np.array(img)
            
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            logger.info(f"Successfully converted HEIC image: {image_path}")
            return img_bgr
            
    except Exception as e:
        logger.error(f"Failed to convert HEIC image {image_path}: {str(e)}")
        raise Exception(f"HEIC conversion failed: {str(e)}")


def convert_pil_to_cv2(image_path: str) -> np.ndarray:
    """
    Convert image using PIL and then to OpenCV format.
    This is used for formats that PIL handles better than OpenCV.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        np.ndarray: Image in BGR format (OpenCV standard)
        
    Raises:
        Exception: If conversion fails
    """
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert PIL image to numpy array
            img_array = np.array(img)
            
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            logger.info(f"Successfully converted image using PIL: {image_path}")
            return img_bgr
            
    except Exception as e:
        logger.error(f"Failed to convert image with PIL {image_path}: {str(e)}")
        raise Exception(f"PIL conversion failed: {str(e)}")


def load_image_as_cv2(image_path: str, 
                      fallback_to_pil: bool = True) -> np.ndarray:
    """
    Load an image and convert it to OpenCV format (BGR numpy array).
    
    This function handles various image formats:
    - JPG/JPEG: Direct OpenCV loading
    - PNG, BMP, TIFF: Direct OpenCV loading
    - HEIC/HEIF: PIL conversion then OpenCV format
    - Other formats: PIL conversion then OpenCV format (if fallback enabled)
    
    Args:
        image_path (str): Path to the image file
        fallback_to_pil (bool): Whether to use PIL as fallback for unsupported formats
        
    Returns:
        np.ndarray: Image in BGR format (OpenCV standard)
        
    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the image format is not supported
        Exception: If image loading/conversion fails
    """
    # Validate input
    if not isinstance(image_path, str):
        raise TypeError(f"Image path must be a string, got {type(image_path)}")
    
    if not image_path.strip():
        raise ValueError("Image path cannot be empty")
    
    # Detect image format
    try:
        file_format = detect_image_format(image_path)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Image format detection failed: {str(e)}")
        raise
    
    logger.info(f"Processing image: {image_path} (format: {file_format})")
    
    # Handle HEIC/HEIF formats
    if file_format in HEIC_FORMATS:
        if not HEIC_SUPPORT:
            raise Exception(f"HEIC support not available. Please install pillow-heif: uv add pillow-heif")
        try:
            return convert_heic_to_cv2(image_path)
        except Exception as e:
            logger.error(f"HEIC conversion failed: {str(e)}")
            raise Exception(f"Failed to convert HEIC image: {str(e)}")
    
    # Handle formats that OpenCV can read directly
    elif file_format in OPENCV_NATIVE_FORMATS:
        try:
            # Try OpenCV first
            img = cv2.imread(image_path)
            if img is not None:
                logger.info(f"Successfully loaded image with OpenCV: {image_path}")
                return img
            else:
                logger.warning(f"OpenCV failed to load {image_path}, trying PIL fallback")
                if fallback_to_pil:
                    return convert_pil_to_cv2(image_path)
                else:
                    raise Exception("OpenCV failed to load image and PIL fallback disabled")
                    
        except Exception as e:
            logger.error(f"OpenCV loading failed: {str(e)}")
            if fallback_to_pil:
                logger.info("Attempting PIL fallback")
                try:
                    return convert_pil_to_cv2(image_path)
                except Exception as pil_e:
                    logger.error(f"PIL fallback also failed: {str(pil_e)}")
                    raise Exception(f"Both OpenCV and PIL failed. OpenCV error: {str(e)}, PIL error: {str(pil_e)}")
            else:
                raise Exception(f"OpenCV loading failed and PIL fallback disabled: {str(e)}")
    
    # Handle other formats with PIL
    else:
        if fallback_to_pil:
            try:
                return convert_pil_to_cv2(image_path)
            except Exception as e:
                logger.error(f"PIL conversion failed for {file_format}: {str(e)}")
                raise Exception(f"Failed to convert {file_format} image: {str(e)}")
        else:
            raise ValueError(f"Format {file_format} not supported by OpenCV and PIL fallback disabled")


def get_image_info(image_path: str) -> dict:
    """
    Get information about an image file.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        dict: Dictionary containing image information
    """
    try:
        # Load image
        img = load_image_as_cv2(image_path)
        
        # Get file info
        file_size = os.path.getsize(image_path)
        file_format = detect_image_format(image_path)
        
        # Get image dimensions
        height, width = img.shape[:2]
        channels = img.shape[2] if len(img.shape) == 3 else 1
        
        return {
            'path': image_path,
            'format': file_format,
            'file_size_bytes': file_size,
            'file_size_mb': round(file_size / (1024 * 1024), 2),
            'width': width,
            'height': height,
            'channels': channels,
            'shape': img.shape
        }
        
    except Exception as e:
        logger.error(f"Failed to get image info for {image_path}: {str(e)}")
        raise


# Example usage and testing function
def test_image_conversion(image_path: str) -> None:
    """
    Test function to demonstrate image conversion.
    
    Args:
        image_path (str): Path to the image file to test
    """
    try:
        print(f"\n=== Testing Image Conversion ===")
        print(f"Image path: {image_path}")
        
        # Get image info
        info = get_image_info(image_path)
        print(f"Image info: {info}")
        
        # Load image
        img = load_image_as_cv2(image_path)
        print(f"Successfully loaded image with shape: {img.shape}")
        print(f"Image dtype: {img.dtype}")
        print(f"Image range: {img.min()} - {img.max()}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
        test_image_conversion(test_image_path)
    else:
        print("Usage: python image_converter.py <image_path>")
        print("Example: python image_converter.py /path/to/image.jpg")
