"""
Robust image format detection and conversion utility.

This module provides functions to detect image formats and convert them
to OpenCV format (BGR numpy array) for consistent processing.
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import Union, Optional, Tuple, List, Dict
import logging

# Set up logging (only if not already configured)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s:%(name)s:%(message)s')
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
            
            logger.debug(f"Successfully converted HEIC image: {image_path}")
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
            
            logger.debug(f"Successfully converted image using PIL: {image_path}")
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
                logger.debug(f"Successfully loaded image with OpenCV: {image_path}")
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


def process_directory_images(directory_path: str, 
                           file_extensions: List[str] = None,
                           recursive: bool = False) -> Dict:
    """
    Process all images in a directory and convert them to OpenCV format.
    
    Args:
        directory_path (str): Path to directory containing images
        file_extensions (List[str], optional): List of file extensions to process
        recursive (bool): Whether to search subdirectories recursively
        
    Returns:
        Dict: Processing results with success/failure counts and image data
    """
    if file_extensions is None:
        file_extensions = ['.jpg', '.jpeg', '.png', '.heic', '.heif', '.bmp', '.tiff', '.tif', '.webp']
    
    import glob
    
    try:
        # Find all image files
        image_files = []
        for ext in file_extensions:
            # Search for both lowercase and uppercase extensions
            patterns = [
                os.path.join(directory_path, f"*{ext}"),
                os.path.join(directory_path, f"*{ext.upper()}"),
            ]
            
            if recursive:
                # Add recursive patterns
                patterns.extend([
                    os.path.join(directory_path, "**", f"*{ext}"),
                    os.path.join(directory_path, "**", f"*{ext.upper()}"),
                ])
            
            for pattern in patterns:
                image_files.extend(glob.glob(pattern, recursive=recursive))
        
        # Remove duplicates and sort
        image_files = sorted(list(set(image_files)))
        
        logger.info(f"Found {len(image_files)} images in {directory_path}")
        
        # Process each image
        results = {
            "total_images": len(image_files),
            "successful_images": 0,
            "failed_images": 0,
            "processed_images": [],
            "failed_files": [],
            "total_faces_detected": 0
        }
        
        for i, image_path in enumerate(image_files, 1):
            logger.info(f"Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
            
            try:
                # Get image info
                image_info = get_image_info(image_path)
                
                # Load image as OpenCV format
                img = load_image_as_cv2(image_path)
                
                # Store processed image data
                processed_image = {
                    "path": image_path,
                    "filename": os.path.basename(image_path),
                    "image_id": os.path.splitext(os.path.basename(image_path))[0],
                    "opencv_image": img,
                    "info": image_info
                }
                
                results["processed_images"].append(processed_image)
                results["successful_images"] += 1
                
                logger.info(f"✅ Successfully processed {os.path.basename(image_path)}")
                
            except Exception as e:
                logger.error(f"❌ Failed to process {image_path}: {str(e)}")
                results["failed_images"] += 1
                results["failed_files"].append({
                    "path": image_path,
                    "error": str(e)
                })
        
        logger.info(f"Directory processing complete: {results['successful_images']}/{results['total_images']} successful")
        return results
        
    except Exception as e:
        logger.error(f"Failed to process directory {directory_path}: {str(e)}")
        raise


def batch_convert_images(directory_path: str, 
                        output_directory: str = None,
                        file_extensions: List[str] = None) -> Dict:
    """
    Batch convert images in a directory to a standardized format.
    
    Args:
        directory_path (str): Path to directory containing images
        output_directory (str, optional): Directory to save converted images
        file_extensions (List[str], optional): List of file extensions to process
        
    Returns:
        Dict: Processing results
    """
    if file_extensions is None:
        file_extensions = ['.jpg', '.jpeg', '.png', '.heic', '.heif', '.bmp', '.tiff', '.tif', '.webp']
    
    try:
        # Process all images
        results = process_directory_images(directory_path, file_extensions)
        
        # Save converted images if output directory specified
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
            
            for processed_image in results["processed_images"]:
                try:
                    # Generate output filename (convert to JPG)
                    base_name = os.path.splitext(processed_image["filename"])[0]
                    output_path = os.path.join(output_directory, f"{base_name}_converted.jpg")
                    
                    # Save as JPG
                    cv2.imwrite(output_path, processed_image["opencv_image"])
                    logger.info(f"Saved converted image: {output_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to save converted image: {str(e)}")
        
        return results
        
    except Exception as e:
        logger.error(f"Batch conversion failed: {str(e)}")
        raise


# Example usage and testing function
def test_image_conversion(image_path: str) -> None:
    """
    Test function to demonstrate image conversion.
    
    Args:
        image_path (str): Path to the image file to test
    """
    try:
        logger.info(f"\n=== Testing Image Conversion ===")
        logger.info(f"Image path: {image_path}")
        
        # Get image info
        info = get_image_info(image_path)
        logger.info(f"Image info: {info}")
        
        # Load image
        img = load_image_as_cv2(image_path)
        logger.info(f"Successfully loaded image with shape: {img.shape}")
        logger.info(f"Image dtype: {img.dtype}")
        logger.info(f"Image range: {img.min()} - {img.max()}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
        test_image_conversion(test_image_path)
    else:
        logger.info("Usage: python image_converter.py <image_path>")
        logger.info("Example: python image_converter.py /path/to/image.jpg")

