#!/usr/bin/env python3
"""
Multimodal RAG - Facial Recognition Pipeline

This is the main entry point for the facial recognition and embedding system.
It processes all images in the query_images directory, converts them to OpenCV format,
extracts face embeddings using DeepFace (RetinaFace + ArcFace), and uploads them
to a Qdrant vector database for similarity search and facial recognition.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Optional, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from facial_detection import FacialDetectionSystem
from image_converter import process_directory_images, get_image_info

# Set up logging (only if not already configured)
if not logging.getLogger().handlers:
    # Create log file handler
    log_file = 'facial_recognition_pipeline.log'
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s:%(name)s:%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler, file_handler]
    )
    
    # Test log file creation
    test_logger = logging.getLogger('test')
    test_logger.info("Log file initialized successfully")

logger = logging.getLogger(__name__)

# Reduce verbosity of third-party libraries but keep warnings visible
logging.getLogger('deepface').setLevel(logging.WARNING)  # Keep warnings visible for debugging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)
logging.getLogger('image_converter').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('requests').setLevel(logging.ERROR)

# Suppress TensorFlow and CUDA warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''


class FacialRecognitionPipeline:
    """Main pipeline class that orchestrates the entire facial recognition process."""
    
    def __init__(self, 
                 qdrant_url: str = "http://localhost:6333",
                 collection_name: str = "detected_faces_collection",
                 model_name: str = "ArcFace",
                 detector_backend: str = "retinaface"):
        """Initialize the facial recognition pipeline."""
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.facial_system = None
        
        logger.info(f"Initialized Facial Recognition Pipeline")
        logger.info(f"  - Qdrant URL: {qdrant_url}")
        logger.info(f"  - Collection: {collection_name}")
        logger.info(f"  - Model: {model_name}")
        logger.info(f"  - Detector: {detector_backend}")
    
    def initialize_system(self) -> bool:
        """Initialize the facial detection system and verify connections."""
        try:
            logger.info("🔧 Initializing facial detection system...")
            
            self.facial_system = FacialDetectionSystem(
                qdrant_url=self.qdrant_url,
                collection_name=self.collection_name,
                model_name=self.model_name,
                detector_backend=self.detector_backend
            )
            
            logger.info("✅ Facial detection system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize facial detection system: {str(e)}")
            return False
    
    def process_directory(self, directory_path: str, 
                         file_extensions: Optional[List[str]] = None) -> dict:
        """Process all images in a directory through the complete pipeline."""
        try:
            if not os.path.exists(directory_path):
                logger.error(f"❌ Directory not found: {directory_path}")
                return {"success": False, "error": "Directory not found"}
            
            logger.info(f"📁 Processing directory: {directory_path}")
            
            # Use the integrated pipeline
            results = self.facial_system.process_directory_integrated(
                directory_path, 
                file_extensions
            )
            
            # Log detailed results
            self._log_processing_results(results)
            return results
            
        except Exception as e:
            logger.error(f"❌ Error processing directory: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _log_processing_results(self, results: dict):
        """Log detailed processing results."""
        try:
            summary = results.get("summary", {})
            
            logger.info("=" * 60)
            logger.info("📊 PROCESSING RESULTS SUMMARY")
            logger.info("=" * 60)
            logger.info(f"📁 Total images found: {summary.get('total_images_found', 0)}")
            logger.info(f"✅ Images converted: {summary.get('images_converted', 0)}")
            logger.info(f"👤 Images with faces: {summary.get('images_with_faces', 0)}")
            logger.info(f"🎭 Total faces detected: {summary.get('total_faces_detected', 0)}")
            logger.info(f"❌ Conversion failures: {summary.get('conversion_failures', 0)}")
            logger.info(f"❌ Facial detection failures: {summary.get('facial_detection_failures', 0)}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Error logging results: {str(e)}")


def main():
    """Main entry point for the facial recognition pipeline."""
    parser = argparse.ArgumentParser(
        description="Facial Recognition Pipeline - Process images and extract face embeddings"
    )
    
    parser.add_argument(
        "--directory", "-d",
        type=str,
        default="data/query_images",
        help="Directory containing images to process (default: data/query_images)"
    )
    
    parser.add_argument(
        "--collection", "-c",
        type=str,
        default="detected_faces_collection",
        help="Qdrant collection name (default: detected_faces_collection)"
    )
    
    parser.add_argument(
        "--qdrant-url",
        type=str,
        default="http://localhost:6333",
        help="Qdrant server URL (default: http://localhost:6333)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="ArcFace",
        choices=["ArcFace", "Facenet", "Facenet512", "VGG-Face", "OpenFace", "DeepFace", "DeepID", "Dlib"],
        help="DeepFace embedding model (default: ArcFace)"
    )
    
    parser.add_argument(
        "--detector",
        type=str,
        default="retinaface",
        choices=["retinaface", "mtcnn", "opencv", "ssd", "dlib"],
        help="Face detector backend (default: retinaface)"
    )
    
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".heic", ".heif", ".bmp", ".tiff", ".tif", ".webp"],
        help="File extensions to process"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Log start time
    start_time = datetime.now()
    logger.info("🚀 Starting Facial Recognition Pipeline")
    logger.info(f"⏰ Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Initialize pipeline
        pipeline = FacialRecognitionPipeline(
            qdrant_url=args.qdrant_url,
            collection_name=args.collection,
            model_name=args.model,
            detector_backend=args.detector
        )
        
        # Initialize system
        if not pipeline.initialize_system():
            logger.error("❌ Failed to initialize pipeline")
            sys.exit(1)
        
        # Process directory
        results = pipeline.process_directory(args.directory, args.extensions)
        if not results.get("success", False):
            logger.error("❌ Pipeline processing failed")
            sys.exit(1)
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time = end_time - start_time
        logger.info(f"⏰ Processing completed in: {processing_time}")
        logger.info("🎉 Facial Recognition Pipeline completed successfully!")
        
        # Show log file location for debugging
        logger.info(f"📝 Detailed logs saved to: facial_recognition_pipeline.log")
        
    except KeyboardInterrupt:
        logger.info("⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Pipeline failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()