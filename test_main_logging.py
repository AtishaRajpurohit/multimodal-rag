#!/usr/bin/env python3
"""
Test the main.py logging setup.
"""

import sys
import os
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the modules to test logging
from facial_detection import FacialDetectionSystem
from image_converter import process_directory_images, get_image_info

# Set up logging exactly like main.py
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s:%(name)s:%(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('test_main_logging.log', mode='w')
        ]
    )

logger = logging.getLogger(__name__)

# Reduce verbosity of third-party libraries but keep warnings visible
logging.getLogger('deepface').setLevel(logging.WARNING)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)
logging.getLogger('image_converter').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('requests').setLevel(logging.ERROR)

def test_logging():
    """Test logging from main pipeline components."""
    logger.info("🚀 Testing main pipeline logging")
    logger.info("🔧 Testing facial detection system import")
    
    # Test importing the system (this should generate logs)
    try:
        logger.info("📁 Testing image converter import")
        logger.info("✅ All imports successful")
    except Exception as e:
        logger.error(f"❌ Import failed: {str(e)}")
    
    logger.info("✅ Logging test completed")
    print("\n📝 Check the log file:")
    print("cat test_main_logging.log")

if __name__ == "__main__":
    test_logging()
