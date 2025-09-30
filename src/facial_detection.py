"""
Facial Detection and Embedding System

This module provides functions to detect faces in images, extract embeddings,
and upload them to Qdrant vector database for facial recognition tasks.
"""

import os
import uuid
import logging
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import cv2

# DeepFace for face detection and embeddings
from deepface import DeepFace

# Qdrant client
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Import our image converter
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from image_converter import load_image_as_cv2, get_image_info

# Set up logging (only if not already configured)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Reduce DeepFace logging verbosity but keep error messages visible
logging.getLogger('deepface').setLevel(logging.WARNING)  # Keep warnings visible
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('requests').setLevel(logging.ERROR)

# Suppress DeepFace print statements
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA warnings

# Redirect stdout temporarily during DeepFace calls
import sys
from contextlib import contextmanager
from io import StringIO

class EmbeddingFilter:
    """Custom stdout filter to suppress embedding arrays while keeping other output."""
    
    def __init__(self, original_stdout):
        self.original_stdout = original_stdout
        self.buffer = ""
    
    def write(self, text):
        # Check if this looks like an embedding array
        if self._is_embedding_array(text):
            return  # Suppress embedding arrays
        else:
            self.original_stdout.write(text)
    
    def _is_embedding_array(self, text):
        """Check if text looks like an embedding array."""
        text = text.strip()
        # Look for patterns that indicate embedding arrays
        embedding_patterns = [
            'array([',  # numpy array
            'tensor([',  # pytorch tensor
            '[[',  # nested list start
            '[-0.',  # negative float start
            '[0.',   # positive float start
            '[-1.',  # negative float start
            '[1.',   # positive float start
        ]
        
        # Check if text starts with any embedding pattern
        for pattern in embedding_patterns:
            if text.startswith(pattern) and len(text) > 50:  # Long arrays are likely embeddings
                return True
        
        # Check if it's a very long line with numbers (likely embedding)
        if len(text) > 100 and any(char.isdigit() for char in text) and '[' in text:
            return True
            
        return False
    
    def flush(self):
        self.original_stdout.flush()

@contextmanager
def suppress_embeddings():
    """Context manager to suppress embedding arrays while keeping other output."""
    old_stdout = sys.stdout
    sys.stdout = EmbeddingFilter(old_stdout)
    try:
        yield
    finally:
        sys.stdout = old_stdout


class FacialDetectionSystem:
    """
    A comprehensive facial detection and embedding system that integrates
    with the image converter and Qdrant vector database.
    """
    
    def __init__(self, 
                 qdrant_url: str = "http://localhost:6333",
                 collection_name: str = "detected_faces_collection",
                 model_name: str = "ArcFace",
                 detector_backend: str = "retinaface"):
        """
        Initialize the facial detection system.
        
        Args:
            qdrant_url (str): URL of the Qdrant server
            collection_name (str): Name of the Qdrant collection
            model_name (str): DeepFace model for embeddings (ArcFace, Facenet, etc.)
            detector_backend (str): Face detector backend (retinaface, mtcnn, etc.)
        """
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.client = None
        
        # Initialize Qdrant client
        self._initialize_qdrant_client()
        
        # Create or get collection
        self._setup_collection()
    
    def _initialize_qdrant_client(self):
        """Initialize Qdrant client connection."""
        try:
            self.client = QdrantClient(url=self.qdrant_url)
            logger.info(f"Connected to Qdrant at {self.qdrant_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            raise
    
    def _setup_collection(self):
        """Create or recreate the Qdrant collection for face embeddings."""
        try:
            # Get embedding dimension based on model
            embedding_dim = self._get_embedding_dimension()
            
            # Recreate collection (this will delete existing data)
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=embedding_dim, 
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Created/recreated collection '{self.collection_name}' with {embedding_dim} dimensions")
            
        except Exception as e:
            logger.error(f"Failed to setup collection: {str(e)}")
            raise
    
    def _get_embedding_dimension(self) -> int:
        """Get the embedding dimension for the specified model."""
        model_dims = {
            "ArcFace": 512,
            "Facenet": 128,
            "Facenet512": 512,
            "VGG-Face": 4096,
            "OpenFace": 128,
            "DeepFace": 4096,
            "DeepID": 160,
            "Dlib": 128
        }
        
        if self.model_name not in model_dims:
            logger.warning(f"Unknown model {self.model_name}, defaulting to 512 dimensions")
            return 512
        
        return model_dims[self.model_name]
    
    def extract_faces_and_embeddings(self, image_path: str) -> List[Dict]:
        """
        Extract face crops and embeddings from a single image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            List[Dict]: List of dictionaries containing face crops and embeddings
                       Each dict has keys: 'face_crop', 'embedding', 'facial_area'
        """
        try:
            # Load image using our robust image converter
            img = load_image_as_cv2(image_path)
            logger.debug(f"Loaded image: {image_path} with shape {img.shape}")
            
            # Extract faces and embeddings using DeepFace with better error handling
            try:
                with suppress_embeddings():
                    results = DeepFace.represent(
                        img_path=image_path,
                        model_name=self.model_name,
                        detector_backend=self.detector_backend,
                        enforce_detection=False,  # Don't fail if no faces found
                        align=True,  # Align faces for better detection
                        normalization='base'  # Use base normalization
                    )
            except Exception as deepface_error:
                logger.warning(f"DeepFace failed for {os.path.basename(image_path)}: {str(deepface_error)}")
                # Try with different detector as fallback
                try:
                    logger.info(f"Trying fallback detector 'opencv' for {os.path.basename(image_path)}")
                    with suppress_embeddings():
                        results = DeepFace.represent(
                            img_path=image_path,
                            model_name=self.model_name,
                            detector_backend="opencv",  # Fallback detector
                            enforce_detection=False,
                            align=True,
                            normalization='base'
                        )
                except Exception as fallback_error:
                    logger.error(f"Both detectors failed for {os.path.basename(image_path)}: {str(fallback_error)}")
                    return []  # Return empty list instead of raising
            
            # Process results
            face_data = []
            for i, result in enumerate(results):
                try:
                    # Get facial area coordinates
                    facial_area = result["facial_area"]
                    x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
                    
                    # Validate coordinates
                    if w <= 0 or h <= 0 or x < 0 or y < 0:
                        logger.warning(f"Invalid facial area coordinates for face {i} in {os.path.basename(image_path)}: {facial_area}")
                        continue
                    
                    # Ensure coordinates are within image bounds
                    x = max(0, min(x, img.shape[1] - 1))
                    y = max(0, min(y, img.shape[0] - 1))
                    w = min(w, img.shape[1] - x)
                    h = min(h, img.shape[0] - y)
                    
                    # Crop face from image
                    face_crop = img[y:y+h, x:x+w]
                    
                    # Validate face crop
                    if face_crop.size == 0:
                        logger.warning(f"Empty face crop for face {i} in {os.path.basename(image_path)}")
                        continue
                    
                    # Store face data
                    face_data.append({
                        "face_crop": face_crop,
                        "embedding": result["embedding"],
                        "facial_area": facial_area,
                        "face_index": i
                    })
                    
                except Exception as face_error:
                    logger.warning(f"Error processing face {i} in {os.path.basename(image_path)}: {str(face_error)}")
                    continue
            
            logger.info(f"Extracted {len(face_data)} faces from {os.path.basename(image_path)}")
            return face_data
            
        except Exception as e:
            logger.error(f"Failed to extract faces from {image_path}: {str(e)}")
            return []  # Return empty list instead of raising
    
    def process_single_image(self, image_path: str, image_id: Optional[str] = None) -> List[Dict]:
        """
        Process a single image and return face embeddings with metadata.
        
        Args:
            image_path (str): Path to the image file
            image_id (str, optional): Custom image ID. If None, uses filename
            
        Returns:
            List[Dict]: List of face records ready for Qdrant upload
        """
        try:
            # Generate image ID if not provided
            if image_id is None:
                image_id = os.path.splitext(os.path.basename(image_path))[0]
            
            # Get image info
            image_info = get_image_info(image_path)
            
            # Extract faces and embeddings
            face_data = self.extract_faces_and_embeddings(image_path)
            
            # Prepare records for Qdrant
            records = []
            for face in face_data:
                record = {
                    "id": str(uuid.uuid4()),
                    "vector": face["embedding"],
                    "payload": {
                        "image_id": image_id,
                        "image_url": os.path.abspath(image_path),
                        "image_filename": os.path.basename(image_path),
                        "face_index": face["face_index"],
                        "facial_area": face["facial_area"],
                        "image_info": {
                            "width": image_info["width"],
                            "height": image_info["height"],
                            "format": image_info["format"],
                            "file_size_mb": image_info["file_size_mb"]
                        }
                    }
                }
                records.append(record)
            
            logger.info(f"Processed {len(records)} faces from image {image_id}")
            return records
            
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {str(e)}")
            raise
    
    def upload_faces_to_qdrant(self, face_records: List[Dict]) -> bool:
        """
        Upload face embeddings to Qdrant collection.
        
        Args:
            face_records (List[Dict]): List of face records to upload
            
        Returns:
            bool: True if upload successful, False otherwise
        """
        try:
            if not face_records:
                logger.warning("No face records to upload")
                return False
            
            # Convert to Qdrant PointStruct objects
            points = []
            for record in face_records:
                point = models.PointStruct(
                    id=record["id"],
                    vector=record["vector"],
                    payload=record["payload"]
                )
                points.append(point)
            
            # Upload to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Successfully uploaded {len(points)} face embeddings to Qdrant")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload faces to Qdrant: {str(e)}")
            return False
    
    def process_and_upload_image(self, image_path: str, image_id: Optional[str] = None) -> bool:
        """
        Process a single image and upload face embeddings to Qdrant.
        
        Args:
            image_path (str): Path to the image file
            image_id (str, optional): Custom image ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Process image
            face_records = self.process_single_image(image_path, image_id)
            
            # Upload to Qdrant
            success = self.upload_faces_to_qdrant(face_records)
            
            if success:
                logger.info(f"Successfully processed and uploaded {len(face_records)} faces from {image_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to process and upload image {image_path}: {str(e)}")
            return False
    
    def process_processed_images(self, processed_images: List[Dict]) -> Dict:
        """
        Process a list of already converted images (from image_converter.py).
        
        Args:
            processed_images (List[Dict]): List of processed images from process_directory_images()
            
        Returns:
            Dict: Summary of processing results
        """
        try:
            logger.info(f"Processing {len(processed_images)} pre-converted images")
            
            results = {
                "total_images": len(processed_images),
                "successful_images": 0,
                "failed_images": 0,
                "total_faces": 0,
                "failed_files": [],
                "face_records": []
            }
            
            for i, processed_image in enumerate(processed_images, 1):
                image_path = processed_image["path"]
                image_id = processed_image["image_id"]
                opencv_image = processed_image["opencv_image"]
                
                logger.info(f"Processing image {i}/{len(processed_images)}: {processed_image['filename']}")
                
                try:
                    # Extract faces and embeddings using the pre-loaded OpenCV image
                    face_data = self._extract_faces_from_opencv_image(
                        opencv_image, 
                        image_path, 
                        image_id
                    )
                    
                    if face_data:
                        # Prepare records for Qdrant
                        face_records = []
                        for face in face_data:
                            record = {
                                "id": str(uuid.uuid4()),
                                "vector": face["embedding"],
                                "payload": {
                                    "image_id": image_id,
                                    "image_url": os.path.abspath(image_path),
                                    "image_filename": processed_image["filename"],
                                    "face_index": face["face_index"],
                                    "facial_area": face["facial_area"],
                                    "image_info": processed_image["info"]
                                }
                            }
                            face_records.append(record)
                        
                        # Upload to Qdrant
                        upload_success = self.upload_faces_to_qdrant(face_records)
                        
                        if upload_success:
                            results["successful_images"] += 1
                            results["total_faces"] += len(face_records)
                            results["face_records"].extend(face_records)
                            logger.info(f"✅ Successfully processed {len(face_records)} faces from {processed_image['filename']}")
                        else:
                            results["failed_images"] += 1
                            results["failed_files"].append({
                                "path": image_path,
                                "error": "Failed to upload to Qdrant"
                            })
                    else:
                        logger.info(f"No faces detected in {processed_image['filename']}")
                        results["successful_images"] += 1  # Still count as successful
                        
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {str(e)}")
                    results["failed_images"] += 1
                    results["failed_files"].append({
                        "path": image_path,
                        "error": str(e)
                    })
            
            logger.info(f"Batch processing complete: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to process pre-converted images: {str(e)}")
            raise
    
    def _extract_faces_from_opencv_image(self, opencv_image: np.ndarray, 
                                       image_path: str, image_id: str) -> List[Dict]:
        """
        Extract faces and embeddings from an already loaded OpenCV image.
        
        Args:
            opencv_image (np.ndarray): Pre-loaded OpenCV image
            image_path (str): Original image path (for DeepFace)
            image_id (str): Image identifier
            
        Returns:
            List[Dict]: List of face data
        """
        try:
            # Use DeepFace with the original image path (DeepFace needs file path)
            try:
                with suppress_embeddings():
                    results = DeepFace.represent(
                        img_path=image_path,
                        model_name=self.model_name,
                        detector_backend=self.detector_backend,
                        enforce_detection=False,
                        align=True,
                        normalization='base'
                    )
            except Exception as deepface_error:
                logger.warning(f"DeepFace failed for {image_id}: {str(deepface_error)}")
                # Try with different detector as fallback
                try:
                    logger.info(f"Trying fallback detector 'opencv' for {image_id}")
                    with suppress_embeddings():
                        results = DeepFace.represent(
                            img_path=image_path,
                            model_name=self.model_name,
                            detector_backend="opencv",
                            enforce_detection=False,
                            align=True,
                            normalization='base'
                        )
                except Exception as fallback_error:
                    logger.error(f"Both detectors failed for {image_id}: {str(fallback_error)}")
                    return []
            
            # Process results
            face_data = []
            for i, result in enumerate(results):
                try:
                    # Get facial area coordinates
                    facial_area = result["facial_area"]
                    x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
                    
                    # Validate coordinates
                    if w <= 0 or h <= 0 or x < 0 or y < 0:
                        logger.warning(f"Invalid facial area coordinates for face {i} in {image_id}: {facial_area}")
                        continue
                    
                    # Ensure coordinates are within image bounds
                    x = max(0, min(x, opencv_image.shape[1] - 1))
                    y = max(0, min(y, opencv_image.shape[0] - 1))
                    w = min(w, opencv_image.shape[1] - x)
                    h = min(h, opencv_image.shape[0] - y)
                    
                    # Crop face from the pre-loaded OpenCV image
                    face_crop = opencv_image[y:y+h, x:x+w]
                    
                    # Validate face crop
                    if face_crop.size == 0:
                        logger.warning(f"Empty face crop for face {i} in {image_id}")
                        continue
                    
                    # Store face data
                    face_data.append({
                        "face_crop": face_crop,
                        "embedding": result["embedding"],
                        "facial_area": facial_area,
                        "face_index": i
                    })
                    
                except Exception as face_error:
                    logger.warning(f"Error processing face {i} in {image_id}: {str(face_error)}")
                    continue
            
            logger.info(f"Extracted {len(face_data)} faces from {image_id}")
            return face_data
            
        except Exception as e:
            logger.error(f"Failed to extract faces from {image_id}: {str(e)}")
            return []

    def process_directory(self, directory_path: str, file_extensions: List[str] = None) -> Dict:
        """
        Process all images in a directory and upload face embeddings to Qdrant.
        
        Args:
            directory_path (str): Path to directory containing images
            file_extensions (List[str], optional): List of file extensions to process
            
        Returns:
            Dict: Summary of processing results
        """
        if file_extensions is None:
            file_extensions = ['.jpg', '.jpeg', '.png', '.heic', '.heif', '.bmp', '.tiff']
        
        try:
            # Find all image files
            image_files = []
            for ext in file_extensions:
                pattern = os.path.join(directory_path, f"*{ext}")
                import glob
                image_files.extend(glob.glob(pattern))
                # Also check uppercase extensions
                pattern = os.path.join(directory_path, f"*{ext.upper()}")
                image_files.extend(glob.glob(pattern))
            
            # Remove duplicates and sort
            image_files = sorted(list(set(image_files)))
            
            logger.info(f"Found {len(image_files)} images in {directory_path}")
            
            # Process each image
            results = {
                "total_images": len(image_files),
                "successful_images": 0,
                "failed_images": 0,
                "total_faces": 0,
                "failed_files": []
            }
            
            for i, image_path in enumerate(image_files, 1):
                logger.info(f"Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
                
                try:
                    # Process and upload
                    success = self.process_and_upload_image(image_path)
                    
                    if success:
                        results["successful_images"] += 1
                        # Count faces from the processed image
                        face_records = self.process_single_image(image_path)
                        results["total_faces"] += len(face_records)
                    else:
                        results["failed_images"] += 1
                        results["failed_files"].append(image_path)
                        
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {str(e)}")
                    results["failed_images"] += 1
                    results["failed_files"].append(image_path)
            
            logger.info(f"Processing complete: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to process directory {directory_path}: {str(e)}")
            raise
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the Qdrant collection.
        
        Returns:
            Dict: Collection statistics
        """
        try:
            # Get collection info
            collection_info = self.client.get_collection(self.collection_name)
            
            # Count total points
            points = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000  # Adjust based on expected size
            )
            
            stats = {
                "collection_name": self.collection_name,
                "total_faces": len(points[0]) if points[0] else 0,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance,
                "status": collection_info.status
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {}


    def process_directory_integrated(self, directory_path: str, 
                                   file_extensions: List[str] = None) -> Dict:
        """
        Integrated pipeline: Convert images + Extract faces + Upload to Qdrant.
        This is the main function that combines image conversion and facial detection.
        
        Args:
            directory_path (str): Path to directory containing images
            file_extensions (List[str], optional): List of file extensions to process
            
        Returns:
            Dict: Complete processing results
        """
        try:
            logger.info(f"🚀 Starting integrated pipeline for directory: {directory_path}")
            
            # Step 1: Convert all images to OpenCV format
            logger.info("📸 Step 1: Converting images to OpenCV format...")
            from image_converter import process_directory_images
            
            conversion_results = process_directory_images(directory_path, file_extensions)
            
            if conversion_results["successful_images"] == 0:
                logger.error("❌ No images were successfully converted")
                return {
                    "conversion_results": conversion_results,
                    "facial_detection_results": None,
                    "total_faces": 0,
                    "success": False
                }
            
            logger.info(f"✅ Converted {conversion_results['successful_images']}/{conversion_results['total_images']} images")
            
            # Step 2: Extract faces and upload to Qdrant
            logger.info("👤 Step 2: Extracting faces and uploading to Qdrant...")
            facial_results = self.process_processed_images(conversion_results["processed_images"])
            
            # Combine results
            final_results = {
                "conversion_results": conversion_results,
                "facial_detection_results": facial_results,
                "total_faces": facial_results["total_faces"],
                "success": facial_results["successful_images"] > 0,
                "summary": {
                    "total_images_found": conversion_results["total_images"],
                    "images_converted": conversion_results["successful_images"],
                    "images_with_faces": facial_results["successful_images"],
                    "total_faces_detected": facial_results["total_faces"],
                    "conversion_failures": conversion_results["failed_images"],
                    "facial_detection_failures": facial_results["failed_images"]
                }
            }
            
            logger.info(f"🎉 Integrated pipeline completed!")
            logger.info(f"📊 Summary: {final_results['summary']}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Integrated pipeline failed: {str(e)}")
            raise


# Convenience functions for easy usage
def process_image_with_faces(image_path: str, 
                           collection_name: str = "detected_faces_collection",
                           qdrant_url: str = "http://localhost:6333") -> bool:
    """
    Convenience function to process a single image and upload face embeddings.
    
    Args:
        image_path (str): Path to the image file
        collection_name (str): Qdrant collection name
        qdrant_url (str): Qdrant server URL
        
    Returns:
        bool: True if successful, False otherwise
    """
    system = FacialDetectionSystem(
        qdrant_url=qdrant_url,
        collection_name=collection_name
    )
    return system.process_and_upload_image(image_path)


def process_directory_with_faces(directory_path: str,
                               collection_name: str = "detected_faces_collection",
                               qdrant_url: str = "http://localhost:6333") -> Dict:
    """
    Convenience function to process all images in a directory.
    
    Args:
        directory_path (str): Path to directory containing images
        collection_name (str): Qdrant collection name
        qdrant_url (str): Qdrant server URL
        
    Returns:
        Dict: Processing results summary
    """
    system = FacialDetectionSystem(
        qdrant_url=qdrant_url,
        collection_name=collection_name
    )
    return system.process_directory(directory_path)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        # Initialize system
        system = FacialDetectionSystem()
        
        # Process single image
        success = system.process_and_upload_image(image_path)
        
        if success:
            logger.info(f"✅ Successfully processed {image_path}")
            stats = system.get_collection_stats()
            logger.info(f"📊 Collection stats: {stats}")
        else:
            logger.error(f"❌ Failed to process {image_path}")
    else:
        logger.info("Usage: python facial_detection.py <image_path>")
        logger.info("Example: python facial_detection.py data/query_images/IMG_2576.JPG")
