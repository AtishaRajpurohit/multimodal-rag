import cv2
import logging
from deepface import DeepFace
from qdrant_client import QdrantClient
from loguru import logger


client = QdrantClient(url="http://localhost:6333")

#Create the refrence read the reference image, upload it to a reference directory on Qdrant.

#Step 0 - Read the image aand load Qdrant. 


#Step  - Image Capture, Embedding, and Uploading to Qdrant
def capture_images_from_webcam(
    window_name="Python Webcam Screenshot",
    collection_name="reference_dataset_collection"
    ):
    """
    Opens the webcam, displays live video, and captures an image
    each time the SPACE key is pressed.
    Press ESC to close the window safely.
    Press SPACE to capture an image, get the embedding and store it in Qdrant.
    """
    # Camera backend is the bridge or driver interface that OpenCV uses to communicate with your system's camera hardware.
    cam = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  #Open the first camera "0" using the AVFoundation backend. 
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

            # ESC pressed
            if k % 256 == 27:
                logger.info("Escape key pressed. Closing...")
                break

            # SPACE pressed
            elif k % 256 == 32:
                img_name = f"opencv_frame_{img_counter}.png"
                cv2.imwrite(img_name, frame)
                logger.info(f"Captured Image: {img_name}")
                #Get facial embeddings
                faces = get_face_embeddings(img_name)

                if not faces:
                    logger.warning(f"No faces detected in {img_name}")
                else:
                    matches = []
                    #f must be a dictionary
                    for f in faces:
                        embedding = f.get("embedding")
                        if embedding is not None:
                            result = search_similar_faces(
                                query_embedding = embedding,
                                collection_name = collection_name,
                                top_k=1
                            )
                            if result:
                                #Adding the label to the results from Deepface! Just updating results with faces!
                                f["label"]=result[0]["label"]
                                f["score"]=result[0]["score"]
                            matches.append(f)



                
                    captured_results.append({
                        "image_path": img_name, #Path of the image that was captured.
                        "matches": matches #List of dictionaries with the keys: id, label, score.
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

#Step 2 - Embedding
def get_face_embeddings(image_path: str):
    '''Get the embedding for a face in an image'''
    try:
        results=DeepFace.represent(
            img_path=image_path,
            model_name = "ArcFace",
            detector_backend = "retinaface",
            enforce_detection = False
        )

        #Is this really needed?
        if not isinstance(results, list):
            results=[results]

        faces=[]
        for r in results:
            faces.append({
                "embedding": r["embedding"],
                "facial_area": r["facial_area"],
                "face_confidence": r["face_confidence"]
            })
        logger.info(f"Extracted {len(faces)} faces from {image_path}")
        return faces
        #result is a list of dictionaries, each dictionary contains the embedding for a face.
    
    except Exception as e:
        logger.error(f"Error extracting face embedding for {image_path}: {e}")
        return None

#Step 5 - Perform Vector Matching
def search_similar_faces(query_embedding, collection_name, top_k=1):
    try:
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True
        )
        
        matches = []
        for res in search_results:
            #Avoids NoneType error, just set payload = True, so you can call it here directly!
            payload = res.payload or {}
            matches.append({
                "id": res.id,
                #Use label from payload
                "label": payload.get("label", "Unknown"),
                "score": res.score
        })

        for m in matches:
            logger.info(f"→ Label: {m['label']} | Score: {m['score']:.4f}")

        return matches
    
    except Exception as e:
        logger.error(f"Error searching for similar faces: {e}")
        return []


if __name__ == "__main__":
    collection_name = "reference_dataset_collection"
    all_results = capture_images_from_webcam(collection_name=collection_name)
    for result in all_results:
        for match in result['matches']:
            logger.info(f"  - Person: {match.get('label', 'Unknown')}")
            logger.info(f"    Score: {match.get('score', 'N/A')}")
    logger.info("Pipeline completed successfully! :)")




    #search_similar_faces(result[0]["embedding"], collection_name="reference_dataset_collection", top_k=1)

    '''For tomorrow :
    So far we are able to search and get the names.
    1. Return the search results as a list of dictionaries with the keys: id, label, score.
    2. Multimodal Embedding for the user-clicked photo.
    3. Supply the search results, along with the multimodal embedding to the LLM.'''