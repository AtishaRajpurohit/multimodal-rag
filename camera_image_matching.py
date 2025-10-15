import cv2
import logging
from deepface import DeepFace
from qdrant_client import QdrantClient

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

#Step 1 - Image Capture, Embedding, and Uploading to Qdrant
def capture_images_from_webcam(window_name="Python Webcam Screenshot"):
    """
    Opens the webcam, displays live video, and captures an image
    each time the SPACE key is pressed.
    Press ESC to close the window safely.
    Press SPACE to capture an image, get the embedding and store it in Qdrant.
    """
    cam = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # Use CAP_DSHOW on Windows if needed
    if not cam.isOpened():
        logger.error("Could not open webcam.")
        return

    cv2.namedWindow(window_name)
    img_counter = 0
    captured_results = []  # store results here


    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                logger.error("Failed to grab frame.")
                break

            cv2.imshow(window_name, frame)
            k = cv2.waitKey(1)

            if k % 256 == 27:
                # ESC pressed
                logger.info("Escape key pressed. Closing...")
                break

            elif k % 256 == 32:
                # SPACE pressed
                img_name = f"opencv_frame_{img_counter}.png"
                cv2.imwrite(img_name, frame)
                logger.info(f"Screenshot saved: {img_name}")

                embedding = get_face_embedding(img_name)
                captured_results.append({
                    "image_path": img_name,
                    "embedding": embedding
                })
                if embedding is not None:
                    #upload_embedding_to_qdrant(embedding, img_name)
                    logger.info(f"Embedding Not found")
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

#Step 2 - Embedding
def get_face_embedding(image_path: str):
    '''Get the embedding for a face in an image'''
    try:
        result=DeepFace.represent(
            img_path=image_path,
            model_name = "ArcFace",
            detector_backend = "retinaface",
            enforce_detection = False
        )
        embedding = result[0]["embedding"]
        logger.info(f"Face embedding extracted for {image_path}")
        return embedding
    
    except Exception as e:
        logger.error(f"Error extracting face embedding for {image_path}: {e}")
        return None

#Step 3 - Qdrant
# from qdrant_client import QdrantClient, models

# client = QdrantClient("http://localhost:6333")  # or your Qdrant URL

# def ensure_collection(collection_name="face_embeddings"):
#     if collection_name not in [c.name for c in client.get_collections().collections]:
#         client.create_collection(
#             collection_name=collection_name,
#             vectors_config=models.VectorParams(size=512, distance="Cosine")
#         )
#         logger.info(f"Created Qdrant collection '{collection_name}'")

#Step 4 - Store Embedding into Qdrant
# import time

# def upload_embedding_to_qdrant(embedding, image_path, collection_name="face_embeddings"):
#     """
#     Uploads a face embedding with metadata to Qdrant.
#     """
#     point = models.PointStruct(
#         id=int(time.time() * 1000),
#         vector=embedding,
#         payload={"image_path": image_path}
#     )
#     client.upsert(collection_name=collection_name, points=[point])
#     logger.info(f"Uploaded embedding for {image_path} to Qdrant")

#Step 5 - Perform Vector Matching
def search_similar_faces(query_embedding, collection_name="face_embeddings", top_k=1):
    client = QdrantClient("http://localhost:6333")
    try:
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        matches = []
        for res in search_results:
            label = res.payload.get("label", "Unknown")
            matches.append({
                "id": res.id,
                "label": label,
                "score": res.score
        })
        for m in matches:
            print(f"→ Label: {m['label']} | Score: {m['score']:.4f}")

        return matches
    
    except Exception as e:
        logger.error(f"Error searching for similar faces: {e}")
        return []


if __name__ == "__main__":
    result = capture_images_from_webcam()
    #print(len(result[0]["embedding"]))

    search_similar_faces(result[0]["embedding"], collection_name="ref_image_test1", top_k=1)
    logger.info("Pipeline completed successfully! :) Log off!!!")