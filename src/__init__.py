"""
Multimodal RAG - Facial Recognition System

This package provides tools for facial detection, embedding extraction,
and vector database management for facial recognition applications.
"""

from .image_converter import (
    load_image_as_cv2,
    get_image_info,
    process_directory_images,
    batch_convert_images
)

from .facial_detection import (
    FacialDetectionSystem,
    process_image_with_faces,
    process_directory_with_faces
)

__version__ = "1.0.0"
__author__ = "Multimodal RAG Team"

__all__ = [
    # Image converter functions
    "load_image_as_cv2",
    "get_image_info", 
    "process_directory_images",
    "batch_convert_images",
    
    # Facial detection classes and functions
    "FacialDetectionSystem",
    "process_image_with_faces",
    "process_directory_with_faces"
]
