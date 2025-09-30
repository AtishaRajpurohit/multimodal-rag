#!/usr/bin/env python3
"""
Test script to verify logging is working properly.
"""

import sys
import os
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s:%(name)s:%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_logging.log', mode='w')
    ]
)

logger = logging.getLogger(__name__)

def test_logging():
    """Test logging functionality."""
    logger.info("🚀 Starting logging test")
    logger.warning("⚠️  This is a warning message")
    logger.error("❌ This is an error message")
    logger.info("✅ Logging test completed")
    
    print("\n📝 Check the log file:")
    print("cat test_logging.log")

if __name__ == "__main__":
    test_logging()
