"""
Multimodal RAG Application - Main Pipeline
==========================================

This script provides a complete pipeline for:
- Part A: Building reference face database (optional)
- Part B: End-to-end image processing with face detection and multimodal description

The pipeline integrates:
1. Face detection and embedding extraction
2. Vector similarity search in Qdrant
3. Multimodal image description using OpenAI GPT-4o

Designed to be easily integrated with Streamlit frontend.
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger

# Core dependencies
import setuptools.dist as distutils
from pillow_heif import register_heif_opener
register_heif_opener()

# Import our custom modules
from .camera_image_matching import CameraImageMatcher
from .rev_multimodal_generation import MultimodalImageDescriber
from .reference_dataset_creation import ReferenceDatasetCreator

# =============================================================================
# CONFIGURATION
# =============================================================================
# Default collection name for face matching
DEFAULT_COLLECTION = "reference_dataset_collection"

# Default image paths for testing
DEFAULT_QUERY_IMAGE = "data/query_images/IMG_2577.JPG"
DEFAULT_REF_IMAGES = [
    "data/query_images/IMG_8916.HEIC",
    "data/query_images/query.JPG"
]

# =============================================================================
# PART A: REFERENCE DATABASE BUILDING (OPTIONAL)
# =============================================================================
def build_reference_database():
    """
    Build the reference face database by extracting faces and uploading to Qdrant.
    
    This function demonstrates how to:
    1. Extract faces from reference images
    2. Save cropped faces for manual labeling
    3. Upload labeled faces to Qdrant collection
    
    Instructions for use:
    1. Place your reference images in a directory (e.g., 'data/ref_images/')
    2. Update the image_paths list below with your image paths
    3. Run this function to extract faces
    4. Review the cropped faces in 'data/reference_images_faces/'
    5. Update the labels list with names for each detected face
    6. Run the upload section to add faces to Qdrant
    """
    logger.info("=" * 60)
    logger.info("PART A: BUILDING REFERENCE DATABASE")
    logger.info("=" * 60)
    
    # Initialize the reference dataset creator
    creator = ReferenceDatasetCreator()
    
    # =========================================================================
    # STEP 1: Define your reference images
    # =========================================================================
    # Add paths to your reference images here
    reference_image_paths = [
        "data/query_images/IMG_8916.HEIC",
        "data/query_images/query.JPG",
        # Add more images here:
        # "data/ref_images/your_image1.jpg",
        # "data/ref_images/your_image2.png",
    ]
    
    # =========================================================================
    # STEP 2: Extract faces from all images
    # =========================================================================
    logger.info("Extracting faces from reference images...")
    face_counts = creator.process_multiple_images(reference_image_paths)
    
    # Display summary
    logger.info("\nFace extraction summary:")
    for image_path, count in face_counts.items():
        logger.info(f"  {os.path.basename(image_path)}: {count} faces detected")
    
    # =========================================================================
    # STEP 3: Define labels for each image
    # =========================================================================
    # IMPORTANT: The number of labels must match the number of faces detected
    # in each image. Check the summary above.
    # 
    # Format: One list of labels per image (in same order as reference_image_paths)
    # Each label corresponds to one detected face in that image.
    
    all_labels = [
        # Labels for first image (IMG_8916.HEIC) - check face count above
        ["Raghav", "Sonali", "Vinayak", "Tala", "Avi", "Olivier", "Atisha", "Matene"],
        # Labels for second image (query.JPG) - check face count above
        ["Rajan Uncle", "Atisha", "Mom", "Rivaan"],
        # Add labels for more images here (if you added more images above):
        # ["Person1", "Person2", "Person3"],  # Labels for third image
    ]
    
    # =========================================================================
    # STEP 4: Upload faces to Qdrant with labels
    # =========================================================================
    logger.info("Uploading faces to Qdrant collection...")
    creator.process_multiple_images_with_labels(
        image_paths=reference_image_paths,
        all_labels=all_labels,
        collection_name=DEFAULT_COLLECTION
    )
    
    logger.info("Reference database building completed!")
    logger.info(f"Check Qdrant dashboard: http://localhost:6333/dashboard")
    logger.info(f"Collection: {DEFAULT_COLLECTION}")


# =============================================================================
# PART B: END-TO-END IMAGE PROCESSING PIPELINE
# =============================================================================
def process_image_pipeline(
    image_path: str,
    collection_name: str = DEFAULT_COLLECTION,
    description_mode: str = "humanlike"
) -> Dict:
    """
    Complete end-to-end pipeline for processing an image.
    
    This function:
    1. Detects faces in the image
    2. Matches faces against the reference database
    3. Generates a multimodal description of the image
    
    Args:
        image_path: Path to the image to process
        collection_name: Name of the Qdrant collection to search
        description_mode: Mode for description ("humanlike", "detailed", "funny")
        
    Returns:
        Dictionary containing:
        - image_path: Path to the processed image
        - faces: List of detected faces with matches
        - description: Generated description text
        - success: Boolean indicating if processing was successful
    """
    logger.info("=" * 60)
    logger.info("PART B: END-TO-END IMAGE PROCESSING")
    logger.info("=" * 60)
    logger.info(f"Processing image: {image_path}")
    logger.info(f"Collection: {collection_name}")
    logger.info(f"Description mode: {description_mode}")
    
    try:
        # =====================================================================
        # STEP 1: Initialize components
        # =====================================================================
        logger.info("Initializing components...")
        matcher = CameraImageMatcher()
        describer = MultimodalImageDescriber()
        
        # =====================================================================
        # STEP 2: Face detection and matching
        # =====================================================================
        logger.info("Detecting faces and searching for matches...")
        face_results = matcher.process_single_image(
            image_path=image_path,
            collection_name=collection_name
        )
        
        if not face_results or not face_results[0].get("faces"):
            logger.warning("No faces detected in the image")
            return {
                "image_path": image_path,
                "faces": [],
                "description": "No faces detected in the image.",
                "success": False
            }
        
        faces = face_results[0]["faces"]
        logger.info(f"Detected {len(faces)} faces")
        
        # Display face detection results
        for i, face in enumerate(faces):
            match_info = face.get('match', {'label': 'Unknown', 'score': 'N/A'})
            logger.info(f"  Face {i+1}: {match_info['label']} (confidence: {face['face_confidence']:.3f}, score: {match_info['score']:.3f})")
        
        # =====================================================================
        # STEP 3: Generate multimodal description
        # =====================================================================
        logger.info("Generating multimodal description...")
        try:
            description = describer.describe_image_with_faces(
                image_path=image_path,
                faces=faces,
                mode=description_mode
            )
            logger.info(f"Description generated successfully: {len(description)} characters")
        except Exception as e:
            logger.error(f"Error generating description: {e}")
            description = f"Error generating description: {str(e)}"
        
        # =====================================================================
        # STEP 4: Return results
        # =====================================================================
        result = {
            "image_path": image_path,
            "faces": faces,
            "description": description,
            "success": True
        }
        
        logger.info("Image processing completed successfully!")
        return result
        
    except Exception as e:
        logger.error(f"Error in image processing pipeline: {e}")
        return {
            "image_path": image_path,
            "faces": [],
            "description": f"Error processing image: {str(e)}",
            "success": False
        }


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    logger.info("Starting Multimodal RAG Application")
    
    # =========================================================================
    # PART A: Build reference database (uncomment to use)
    # =========================================================================
    # Uncomment the line below to build/update your reference database
    # build_reference_database()
    
    # =========================================================================
    # PART B: Process an image end-to-end
    # =========================================================================
    # Process a single image through the complete pipeline
    result = process_image_pipeline(
        image_path=DEFAULT_QUERY_IMAGE,
        collection_name=DEFAULT_COLLECTION,
        description_mode="humanlike"
    )
    
    # Display results
    if result["success"]:
        print("\n" + "=" * 60)
        print("PROCESSING RESULTS")
        print("=" * 60)
        print(f"Image: {result['image_path']}")
        print(f"Faces detected: {len(result['faces'])}")
        print("\nDescription:")
        print(result['description'])
    else:
        print(f"Processing failed: {result['description']}")
    
    logger.info("Application completed")