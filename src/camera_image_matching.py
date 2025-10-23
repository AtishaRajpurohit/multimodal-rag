import cv2
from deepface import DeepFace
from qdrant_client import QdrantClient
from loguru import logger
import distutils as distutils

client = QdrantClient(url="http://localhost:6333")


#Step 1 - Main Function : Image Capture
# Helper Function : Embedding and Uploading to Qdrant
def capture_images_from_webcam(
    window_name="Python Webcam Screenshot",
    collection_name="reference_dataset_collection"
    ):
    """
    Opens the webcam, displays webcam, and captures an image.
    each time the SPACE key is pressed.
    Press ESC to close the window safely.
    Press SPACE to capture an image, get the embedding and store it in Qdrant.
    """

    # Camera backend is the bridge or driver interface that OpenCV uses to communicate with your system's camera hardware.
    # Open the first camera "0" using the AVFoundation backend. 
    cam = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  
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

# Helper: process a single image file exactly like capture_images_from_webcam would - Difference in this and main function? 
def process_single_image(image_path,collection_name="reference_dataset_collection"):
    """
    Takes an image path and runs embedding + matching,
    returning results in the same format as capture_images_from_webcam().
    """
    faces = get_face_embeddings(image_path)
    results = []

    if not faces:
        logger.warning(f"No faces detected in {image_path}.")
    else:
        matches = []
        for f in faces:
            embedding = f.get("embedding")
            if embedding is not None:
                result = search_similar_faces(
                    query_embedding=embedding,
                    collection_name=collection_name,
                    top_k=1
                )
                if result:
                    #Extracting the first match.
                    f["match"] = result[0]
                matches.append(f)
        results.append({
            "image_path": image_path,
            "faces": matches
        })

    return results



if __name__ == "__main__":
    collection_name = "reference_dataset_collection"
    all_results = process_single_image(image_path="temp_image.png",collection_name=collection_name)
    temp = ((all_results[0]["faces"][0]))
    print(temp["match"])
    
    

    #print(len(all_results[0]["faces"]))





    
    # all_results = capture_images_from_webcam(collection_name=collection_name)
    # for result in all_results:
    #     for match in result['matches']:
    #         logger.info(f"  - Person: {match.get('label', 'Unknown')}")
    #         logger.info(f"    Score: {match.get('score', 'N/A')}")
    # logger.info("Pipeline completed successfully! :)")

    # Add this at the very end of your file, after line 209

'''Code is getting stuck with the disutils error. Come back and review'''





    