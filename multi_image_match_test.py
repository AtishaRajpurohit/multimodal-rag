# Query Image Understanding Pipeline
#
# This updated script is designed for:
# - Uploading a single query image
# - Detecting how many people are in it
# - Matching each face with reference individuals
# - Generating a natural-language story of the image based on faces, scene, and metadata
#
# ---
#
# Multi-person Visual Understanding Script
# Dependencies: torch, torchvision, clip, face_recognition, transformers, PIL, piexif, geopy

import os
import torch
import clip
import numpy as np
import cv2

# Try to import face_recognition, provide fallback if not available
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    print("Warning: face_recognition library not found. Using OpenCV + CLIP for face detection instead.")
    print("Install with: pip install face_recognition (requires CMake and C++ build tools)")
    FACE_RECOGNITION_AVAILABLE = False
from PIL import Image as PILImage
from PIL.ExifTags import TAGS, GPSTAGS
from transformers import BlipProcessor, BlipForConditionalGeneration
from geopy.geocoders import Nominatim

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess_clip = clip.load("ViT-B/32", device=device)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Load and encode reference faces
def load_reference_faces(ref_dir):
    if not FACE_RECOGNITION_AVAILABLE:
        print("Using CLIP-based face matching...")
        return load_reference_faces_clip(ref_dir)
    
    known_encodings, known_names = [], []
    for file in os.listdir(ref_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = face_recognition.load_image_file(os.path.join(ref_dir, file))
            encodings = face_recognition.face_encodings(img)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(file.split(".")[0])
    return known_encodings, known_names

# CLIP-based face matching fallback
def load_reference_faces_clip(ref_dir):
    known_embeddings, known_names = [], []
    for file in os.listdir(ref_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(ref_dir, file)
            img = PILImage.open(img_path).convert("RGB")
            img_tensor = preprocess_clip(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                embedding = clip_model.encode_image(img_tensor)
                known_embeddings.append(embedding)
                known_names.append(file.split(".")[0])
    
    if known_embeddings:
        known_embeddings = torch.cat(known_embeddings)
    return known_embeddings, known_names

# Match faces in query image to known people
def match_faces(query_img_path, known_encodings, known_names, threshold=0.6):
    if not FACE_RECOGNITION_AVAILABLE:
        return match_faces_clip(query_img_path, known_encodings, known_names, threshold)
    
    query_img = face_recognition.load_image_file(query_img_path)
    locations = face_recognition.face_locations(query_img)
    encodings = face_recognition.face_encodings(query_img, locations)
    matches = []
    for encoding in encodings:
        distances = face_recognition.face_distance(known_encodings, encoding)
        best_idx = np.argmin(distances)
        if distances[best_idx] < threshold:
            matches.append(known_names[best_idx])
    return list(set(matches)), len(encodings)

# Face detection and matching using OpenCV + CLIP
def detect_and_match_faces_opencv_clip(query_img_path, known_embeddings, known_names, threshold=0.5):
    """
    Detect faces in query image using OpenCV and match them with reference faces using CLIP
    """
    # Load query image
    query_img = PILImage.open(query_img_path).convert("RGB")
    
    # Convert PIL image to OpenCV format
    opencv_img = cv2.cvtColor(np.array(query_img), cv2.COLOR_RGB2BGR)
    
    # Load OpenCV's face detection cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    print(f"OpenCV detected {len(faces)} faces")
    
    detected_faces = []
    
    for i, (x, y, w, h) in enumerate(faces):
        # Extract face region from PIL image
        face_region = query_img.crop((x, y, x+w, y+h))
        
        # Process face region with CLIP
        face_tensor = preprocess_clip(face_region).unsqueeze(0).to(device)
        
        with torch.no_grad():
            face_embedding = clip_model.encode_image(face_tensor)
        
        # Calculate similarities with reference faces
        similarities = torch.cosine_similarity(face_embedding, known_embeddings, dim=1)
        similarities = similarities.cpu().numpy()
        
        # Find best match
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        print(f"Face {i} at ({x},{y},{w},{h}) similarities: {dict(zip(known_names, similarities))}")
        
        if best_similarity > threshold:
            detected_faces.append({
                'face_id': i,
                'name': known_names[best_match_idx],
                'similarity': best_similarity,
                'coordinates': (x, y, x+w, y+h)
            })
            print(f"  -> Detected {known_names[best_match_idx]} with similarity {best_similarity:.3f}")
        else:
            print(f"  -> No match above threshold {threshold}")
    
    # Remove duplicates (same person detected multiple times)
    unique_faces = {}
    for face in detected_faces:
        name = face['name']
        if name not in unique_faces or face['similarity'] > unique_faces[name]['similarity']:
            unique_faces[name] = face
    
    detected_people = list(unique_faces.keys())
    face_count = len(detected_people)
    
    print(f"Unique people found: {detected_people}")
    print(f"Face details: {[(f['name'], f['similarity']) for f in unique_faces.values()]}")
    
    return detected_people, face_count

# Generate BLIP caption
def describe_scene(image_path):
    img = PILImage.open(image_path).convert("RGB")
    inputs = blip_processor(images=img, return_tensors="pt").to(device)
    output = blip_model.generate(**inputs)
    return blip_processor.decode(output[0], skip_special_tokens=True)

# Extract metadata
def extract_exif(image_path):
    try:
        img = PILImage.open(image_path)
        exif_data = img._getexif()
        if not exif_data:
            print(f"No EXIF data found in {image_path}")
            return {}
        
        metadata = {}
        for tag_id, val in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            metadata[tag] = val
        
        print(f"EXIF data found: {list(metadata.keys())}")
        # Print some key values for debugging
        for key in ['DateTime', 'DateTimeOriginal', 'Model', 'Make', 'GPSInfo']:
            if key in metadata:
                print(f"  {key}: {metadata[key]}")
        return metadata
    except Exception as e:
        print(f"Error extracting EXIF data: {e}")
        return {}

# Convert GPS to address
def gps_to_address(gps_info):
    def convert(value):
        d, m, s = value
        return d[0]/d[1] + m[0]/m[1]/60 + s[0]/s[1]/3600

    lat = convert(gps_info[2])
    if gps_info[1] != 'N': lat = -lat
    lon = convert(gps_info[4])
    if gps_info[3] != 'E': lon = -lon

    geolocator = Nominatim(user_agent="geoapi")
    location = geolocator.reverse((lat, lon), language='en')
    return location.address if location else None

# Generate storytelling output
def summarize(query_img_path, reference_dir):
    known_encodings, known_names = load_reference_faces(reference_dir)
    
    if FACE_RECOGNITION_AVAILABLE:
        people_found, face_count = match_faces(query_img_path, known_encodings, known_names)
    else:
        # Use OpenCV + CLIP for proper face detection and matching
        people_found, face_count = detect_and_match_faces_opencv_clip(query_img_path, known_encodings, known_names, threshold=0.4)
    
    caption = describe_scene(query_img_path)
    exif = extract_exif(query_img_path)

    # Extract date and time
    date_time = exif.get("DateTime", exif.get("DateTimeOriginal", "Unknown time"))
    
    # Extract camera model
    camera = exif.get("Model", exif.get("Make", "Unknown camera"))
    if camera != "Unknown camera" and exif.get("Make"):
        camera = f"{exif.get('Make')} {camera}"
    
    # Extract location
    location = "Unknown location"
    if "GPSInfo" in exif:
        try:
            location = gps_to_address(exif["GPSInfo"])
            if not location:
                location = "Unknown location"
        except Exception as e:
            print(f"Error processing GPS data: {e}")
            location = "Unknown location"
    
    # If no metadata is available, provide more informative defaults
    if date_time == "Unknown time":
        date_time = "recently"  # More natural than "Unknown time"
    if camera == "Unknown camera":
        camera = "a digital camera"  # More natural than "Unknown camera"
    if location == "Unknown location":
        location = "an unknown location"  # More natural than "Unknown location"
    
    print(f"Extracted metadata - Date: {date_time}, Camera: {camera}, Location: {location}")

    # Build a storytelling sentence with identified people
    if people_found:
        if len(people_found) == 1:
            names_text = people_found[0]
        elif len(people_found) == 2:
            names_text = f"{people_found[0]} and {people_found[1]}"
        else:
            names_text = ", ".join(people_found[:-1]) + f" and {people_found[-1]}"
        
        story = f"{names_text} are posing for a photo. {caption}. " \
                f"The photo was taken on {date_time} at {location} using a {camera}."
    else:
        story = f"This image features unknown individuals, with a total of {face_count} person(s) detected. " \
                f"They appear to be: {caption}. " \
                f"The photo was taken on {date_time} at {location} using a {camera}."

    print("\n--- IMAGE STORY ---")
    print(story)
    print("--------------------")

# Run
if __name__ == "__main__":
    summarize("query.JPG", "ref_images")
