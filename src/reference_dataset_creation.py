from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger
import os
import cv2
from PIL import Image
import numpy as np
import setuptools.dist as distutils
from deepface import DeepFace
from qdrant_client import QdrantClient, models
from pillow_heif import register_heif_opener

# Register HEIF opener
register_heif_opener()

from detect import FacialDetector
from vector_db import VectorDB


class ReferenceDatasetCreator:
    """
    A class for creating reference datasets with facial crops and uploading to Qdrant.
    Handles two-phase workflow: face extraction and embedding upload.
    """
    
    def __init__(self, output_dir: str = "data/reference_images_faces"):
        """
        Initialize the ReferenceDatasetCreator.
        
        Args:
            output_dir: Directory to save cropped face images
        """
        self.output_dir = output_dir
        self._ensure_output_dir()
        logger.info(f"Initialized ReferenceDatasetCreator with output directory: {output_dir}")
    
    def _ensure_output_dir(self) -> None:
        """Ensure the output directory exists."""
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Output directory ready: {self.output_dir}")

    def extract_faces_from_image(self, image_path: str) -> int:
        """
        Extract faces from an image and save cropped faces to output directory.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Number of faces detected and saved
        """
        self._validate_image_path(image_path)
        
        logger.info(f"Processing image for face extraction: {image_path}")
        
        # Preprocess image
        detector = FacialDetector(image_path)
        processed_image = detector.preprocess_image(resize=(512, 512))
        
        # Extract faces
        faces = DeepFace.extract_faces(
            img_path=processed_image,
            detector_backend="retinaface"
        )
        
        logger.info(f"Extracted {len(faces)} faces from {image_path}")
        
        # Save cropped faces
        self._save_cropped_faces(faces, image_path)
        
        return len(faces)
    
    def _validate_image_path(self, image_path: str) -> None:
        """Validate the input image path."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if not isinstance(image_path, str):
            raise TypeError(f"Image path must be a string, got: {type(image_path)}")
    
    def _save_cropped_faces(self, faces: List[Dict], image_path: str) -> None:
        """
        Save cropped faces as JPG files.
        
        Args:
            faces: List of face data from DeepFace.extract_faces
            image_path: Original image path for naming
        """
        image_name = os.path.basename(image_path)
        
        for i, face in enumerate(faces):
            # DeepFace normalizes, so rescale back to 0-255
            cropped_face = (face["face"] * 255).astype("uint8")
            # Convert RGB to BGR for OpenCV
            cropped_face_bgr = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
            
            # Save with descriptive filename
            output_filename = f"face_{i+1}_{image_name}.jpg"
            output_path = os.path.join(self.output_dir, output_filename)
            cv2.imwrite(output_path, cropped_face_bgr)
            
            logger.info(f"Saved face {i+1}: {output_filename}")
    
    def process_multiple_images(self, image_paths: List[str]) -> Dict[str, int]:
        """
        Process multiple images for face extraction.
        
        Args:
            image_paths: List of image paths to process
            
        Returns:
            Dictionary mapping image paths to number of faces detected
        """
        results = {}
        
        for image_path in image_paths:
            try:
                face_count = self.extract_faces_from_image(image_path)
                results[image_path] = face_count
                logger.info(f"Image: {image_path} -> {face_count} faces")
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results[image_path] = 0
        
        return results

    def embed_and_upload_to_qdrant(
        self,
        image_path: str,
        labels: List[str],
        collection_name: str = "reference_dataset_collection"
    ) -> None:
        """
        Embed faces from an image and upload to Qdrant with labels.
        
        Args:
            image_path: Path to the input image
            labels: List of labels for each detected face (in order)
            collection_name: Name of the Qdrant collection
        """
        self._validate_image_path(image_path)
        self._validate_labels(labels)
        
        logger.info(f"Embedding and uploading faces for: {image_path}")
        
        # Get face embeddings
        detector = FacialDetector(image_path)
        processed_image = detector.preprocess_image(resize=(512, 512))
        face_results = detector.facial_detection_embedding(img_array=processed_image)
        
        # Validate face count matches label count
        self._validate_face_label_count(face_results, labels, image_path)
        
        # Upload to Qdrant
        vector_db = VectorDB()
        vector_db.upload_detected_faces_to_qdrant(
            collection_name=collection_name,
            detected_faces_list=face_results,
            image_path=image_path,
            labels=labels
        )
        
        logger.info(f"Successfully uploaded {len(face_results)} faces to collection '{collection_name}'")
    
    def _validate_labels(self, labels: List[str]) -> None:
        """Validate the labels list."""
        if not isinstance(labels, list):
            raise TypeError(f"Labels must be a list, got: {type(labels)}")
        
        if not labels:
            raise ValueError("Labels list cannot be empty")
    
    def _validate_face_label_count(self, face_results: List[Dict], labels: List[str], image_path: str) -> None:
        """Validate that face count matches label count."""
        face_count = len(face_results)
        label_count = len(labels)
        
        if face_count != label_count:
            raise ValueError(
                f"Face count ({face_count}) doesn't match label count ({label_count}) "
                f"for image: {image_path}"
            )
    
    def process_multiple_images_with_labels(
        self,
        image_paths: List[str],
        all_labels: List[List[str]],
        collection_name: str = "reference_dataset_collection"
    ) -> None:
        """
        Process multiple images with their corresponding labels and upload to Qdrant.
        
        Args:
            image_paths: List of image paths
            all_labels: List of label lists (one per image)
            collection_name: Name of the Qdrant collection
        """
        if len(image_paths) != len(all_labels):
            raise ValueError("Number of images must match number of label lists")
        
        for image_path, labels in zip(image_paths, all_labels):
            try:
                self.embed_and_upload_to_qdrant(image_path, labels, collection_name)
                logger.info(f"Completed processing: {image_path}")
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                raise


# -----------------------------------------------------------------------------
# Example usage - Two-phase workflow
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Initialize the creator
    creator = ReferenceDatasetCreator()
    
    # =============================================================================
    # STEP 1: ADD YOUR IMAGES HERE
    # =============================================================================
    # To add new images to your reference dataset:
    # 1. Place your images in the data/query_images/ directory
    # 2. Add the image paths to the list below
    # 3. Run the script to extract faces (Phase 1)
    # 4. Review the cropped faces in data/reference_images_faces/
    # 5. Create label lists matching the number of faces detected
    # 6. Update the all_labels list below with your labels
    # 7. Run the script again to upload to Qdrant (Phase 2)
    
    input_image_paths = [
        "data/query_images/IMG_8916.HEIC",
        "data/query_images/query.JPG"
        # Add more images here:
        # "data/query_images/your_new_image.jpg",
        # "data/query_images/another_image.png",
    ]
    
    collection_name = "reference_dataset_collection"
    
    # =============================================================================
    # PHASE 1: FACE EXTRACTION AND CROPPING
    # =============================================================================
    # This phase:
    # - Detects faces in each image
    # - Saves cropped face images to data/reference_images_faces/
    # - Shows you exactly how many faces were detected per image
    # - You can then review the cropped faces and prepare labels
    
    logger.info("=" * 60)
    logger.info("PHASE 1: FACE EXTRACTION AND CROPPING")
    logger.info("=" * 60)
    logger.info("This phase will:")
    logger.info("1. Detect faces in each image")
    logger.info("2. Save cropped faces to data/reference_images_faces/")
    logger.info("3. Show you the number of faces detected per image")
    logger.info("4. Allow you to review faces and prepare labels")
    logger.info("")
    
    # Extract faces from all images
    face_counts = creator.process_multiple_images(input_image_paths)
    
    # Display summary for manual labeling
    logger.info("\n" + "=" * 40)
    logger.info("FACE EXTRACTION SUMMARY")
    logger.info("=" * 40)
    for image_path, count in face_counts.items():
        logger.info(f"Image: {os.path.basename(image_path)}")
        logger.info(f"  → {count} faces detected")
        logger.info(f"  → Check: {creator.output_dir}/face_*_{os.path.basename(image_path)}.jpg")
        logger.info("")
    
    logger.info("NEXT STEPS:")
    logger.info("1. Review the cropped faces in data/reference_images_faces/")
    logger.info("2. Identify each person in the cropped faces")
    logger.info("3. Create label lists matching the face count for each image")
    logger.info("4. Update the all_labels list below with your labels")
    logger.info("5. Run the script again to upload to Qdrant")
    logger.info("")
    
    # =============================================================================
    # PHASE 2: EMBEDDING AND UPLOAD TO QDRANT
    # =============================================================================
    # This phase:
    # - Takes your labeled faces and creates embeddings
    # - Uploads the embeddings to Qdrant with the labels
    # - Creates/updates the reference collection for face matching
    
    logger.info("=" * 60)
    logger.info("PHASE 2: EMBEDDING AND UPLOAD TO QDRANT")
    logger.info("=" * 60)
    logger.info("This phase will:")
    logger.info("1. Create face embeddings for each detected face")
    logger.info("2. Upload embeddings to Qdrant with your labels")
    logger.info("3. Create/update the reference collection")
    logger.info("4. Make faces available for matching in your app")
    logger.info("")
    
    # =============================================================================
    # STEP 2: UPDATE YOUR LABELS HERE
    # =============================================================================
    # IMPORTANT: The number of labels must match the number of faces detected
    # in each image. Check the face extraction summary above.
    # 
    # Format: One list of labels per image (in same order as input_image_paths)
    # Each label corresponds to one detected face in that image.
    
    all_labels = [
        # Labels for first image (IMG_8916.HEIC) - 8 faces detected
        ["Raghav","Sonali","Vinayak","Tala","Avi","Olivier","Atisha","Matene"],
        # Labels for second image (query.JPG) - 4 faces detected  
        ["Rajan Uncle", "Atisha", "Mom", "Rivaan"]
        # Add labels for more images here (if you added more images above):
        # ["Person1", "Person2", "Person3"],  # Labels for third image
    ]
    
    # Upload to Qdrant with labels
    creator.process_multiple_images_with_labels(
        image_paths=input_image_paths,
        all_labels=all_labels,
        collection_name=collection_name
    )
    
    # =============================================================================
    # COMPLETION SUMMARY
    # =============================================================================
    logger.info("\n" + "=" * 60)
    logger.info("REFERENCE DATASET CREATION COMPLETED!")
    logger.info("=" * 60)
    logger.info("Your reference dataset is now ready!")
    logger.info("")
    logger.info("What was created:")
    logger.info(f"✓ Cropped faces saved in: {creator.output_dir}")
    logger.info(f"✓ Face embeddings uploaded to Qdrant collection: {collection_name}")
    logger.info(f"✓ Total faces processed: {sum(face_counts.values())}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Check Qdrant dashboard: http://localhost:6333/dashboard")
    logger.info("2. Verify your collection and data")
    logger.info("3. Use CameraImageMatcher.process_single_image() in your Streamlit app")
    logger.info("")
    logger.info("To add more images in the future:")
    logger.info("1. Add image paths to input_image_paths list")
    logger.info("2. Run Phase 1 to extract faces")
    logger.info("3. Review cropped faces and prepare labels")
    logger.info("4. Update all_labels list with new labels")
    logger.info("5. Run Phase 2 to upload to Qdrant")