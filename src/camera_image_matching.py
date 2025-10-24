import cv2
import setuptools.dist as distutils
from deepface import DeepFace
from qdrant_client import QdrantClient
from loguru import logger
from typing import List, Dict, Optional


class CameraImageMatcher:
    """
    A class for camera-based image capture, face detection, and matching.
    Handles webcam capture, face embedding extraction, and vector similarity search.
    """
    
    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        """
        Initialize the CameraImageMatcher.
        
        Args:
            qdrant_url: Qdrant server URL
        """
        self.client = QdrantClient(url=qdrant_url)
        logger.info(f"Initialized CameraImageMatcher with Qdrant at {qdrant_url}")
    
    def get_face_embeddings(self, image_path: str) -> Optional[List[Dict]]:
        """
        Get face embeddings from an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of face dictionaries with embeddings, or None if error
        """
        try:
            results = DeepFace.represent(
                img_path=image_path,
                model_name="ArcFace",
                detector_backend="retinaface",
                enforce_detection=False
            )

            # Ensure results is a list
            if not isinstance(results, list):
                results = [results]

            faces = []
            for r in results:
                faces.append({
                    "embedding": r["embedding"],
                    "facial_area": r["facial_area"],
                    "face_confidence": r["face_confidence"]
                })
            
            logger.info(f"Extracted {len(faces)} faces from {image_path}")
            return faces
            
        except Exception as e:
            logger.error(f"Error extracting face embedding for {image_path}: {e}")
            return None

    def search_similar_faces(self, query_embedding: List[float], collection_name: str, top_k: int = 1) -> List[Dict]:
        """
        Search for similar faces in Qdrant collection.
        
        Args:
            query_embedding: Face embedding vector to search for
            collection_name: Name of the Qdrant collection
            top_k: Number of top matches to return
            
        Returns:
            List of matching face results
        """
        try:
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True
            )
            
            matches = []
            for res in search_results:
                # Avoid NoneType error, just set payload = True, so you can call it here directly!
                payload = res.payload or {}
                matches.append({
                    "id": res.id,
                    # Use label from payload
                    "label": payload.get("label", "Unknown"),
                    "score": res.score
                })

            for m in matches:
                logger.info(f"→ Label: {m['label']} | Score: {m['score']:.4f}")

            return matches
            
        except Exception as e:
            logger.error(f"Error searching for similar faces: {e}")
            return []

    def process_single_image(self, image_path: str, collection_name: str = "reference_dataset_collection") -> List[Dict]:
        """
        Process a single image for face detection and matching.
        
        Args:
            image_path: Path to the image file
            collection_name: Name of the Qdrant collection to search
            
        Returns:
            List of results with detected faces and matches
        """
        faces = self.get_face_embeddings(image_path)
        results = []

        if not faces:
            logger.warning(f"No faces detected in {image_path}.")
        else:
            matches = []
            for f in faces:
                embedding = f.get("embedding")
                if embedding is not None:
                    result = self.search_similar_faces(
                        query_embedding=embedding,
                        collection_name=collection_name,
                        top_k=1
                    )
                    if result:
                        # Extracting the first match.
                        f["match"] = result[0]
                    matches.append(f)
            results.append({
                "image_path": image_path,
                "faces": matches
            })

        return results

    def capture_images_from_webcam(
        self,
        window_name: str = "Python Webcam Screenshot",
        collection_name: str = "reference_dataset_collection"
    ) -> List[Dict]:
        """
        Capture images from webcam with face detection and matching.
        
        Args:
            window_name: Name of the OpenCV window
            collection_name: Name of the Qdrant collection to search
            
        Returns:
            List of captured results with face matches
        """
        # Initialize camera
        cam = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        if not cam.isOpened():
            logger.error("Could not open webcam.")
            return []

        cv2.namedWindow(window_name)
        img_counter = 0
        captured_results = []

        try:
            while True:
                ret, frame = cam.read()
                if not ret:
                    logger.error("Failed to grab frame.")
                    break

                cv2.imshow(window_name, frame)
                k = cv2.waitKey(1)

                # ESC pressed
                if k % 256 == 27:
                    logger.info("Escape key pressed. Closing...")
                    break

                # SPACE pressed
                elif k % 256 == 32:
                    img_name = f"opencv_frame_{img_counter}.png"
                    cv2.imwrite(img_name, frame)
                    logger.info(f"Captured Image: {img_name}")
                    
                    # Get facial embeddings
                    faces = self.get_face_embeddings(img_name)

                    if not faces:
                        logger.warning(f"No faces detected in {img_name}")
                    else:
                        matches = self._process_face_matches(faces, collection_name)
                        captured_results.append({
                            "image_path": img_name,
                            "matches": matches
                        })
                        img_counter += 1
                
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt detected. Stopping capture loop...")

        finally:
            # Ensure proper cleanup
            cam.release()
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            logger.info("Camera released and windows closed.")
        
        return captured_results

    def _process_face_matches(self, faces: List[Dict], collection_name: str) -> List[Dict]:
        """
        Process face matches for webcam capture.
        
        Args:
            faces: List of detected faces
            collection_name: Name of the Qdrant collection
            
        Returns:
            List of faces with match information
        """
        matches = []
        for f in faces:
            embedding = f.get("embedding")
            if embedding is not None:
                result = self.search_similar_faces(
                    query_embedding=embedding,
                    collection_name=collection_name,
                    top_k=1
                )
                if result:
                    # Adding the label to the results from Deepface! Just updating results with faces!
                    f["label"] = result[0]["label"]
                    f["score"] = result[0]["score"]
                matches.append(f)
        return matches


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Initialize the matcher
    matcher = CameraImageMatcher()
    
    try:
        # Test single image processing
        collection_name = "reference_dataset_collection"
        all_results = matcher.process_single_image(
            image_path="temp_image.png",
            collection_name=collection_name
        )
        
        if all_results and len(all_results) > 0:
            result = all_results[0]
            print(f"Image: {result['image_path']}")
            print(f"Faces detected: {len(result['faces'])}")
            
            for i, face in enumerate(result['faces']):
                print(f"\nFace {i+1}:")
                print(f"  Area: {face['facial_area']}")
                print(f"  Confidence: {face['face_confidence']:.3f}")
                
                if 'match' in face:
                    match = face['match']
                    print(f"  → {match['label']} (score: {match['score']:.3f})")
                else:
                    print("  → No match found")
        else:
            print("No faces detected or no results returned")
            
    except Exception as e:
        logger.error(f"Example failed: {e}")