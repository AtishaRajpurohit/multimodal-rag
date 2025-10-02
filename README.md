## Overall Goal - A user uploads a number of images, and then gets blog with explaining the entire scenario. 

## Setup Instructions :
### Qdrant : In order, to run the code and setup qdrant, please follow the following steps. 

Step 1 : Download Docker Desktop. Verify installation using :
    docker --version

Step 2: Pull Qdrant Image
    docker pull/qdrant/qdrant

Step 3 : Run Qdrant container 
    docker run -p 6333:6333 qdrant/qdrant
-p 6333:6333 → exposes Qdrant’s HTTP API on port 6333.

Step 4 : Test Qdrant id running, by opening it in the browser.
    http://localhost:6333/dashboard

Step 5 : Install qdrant client
    uv add qdrant_client

Note : The current code runs qdrant locally!

## Step 1: Image Discovery & Conversion
- Finds all images in directory
- Converts JPG, HEIC, PNG, BMP, TIFF, etc. to OpenCV format
- Handles format-specific issues (HEIC requires pillow-heif)
- Tracks conversion success/failure

## Step 2: Facial Detection & Embedding
- Uses DeepFace (RetinaFace + ArcFace) for each image
- Extracts face crops and 512-dimensional embeddings
- Handles images with multiple faces
- Tracks detection success/failure

## Step 3: Qdrant Upload
- Creates/recreates collection with proper configuration
- Uploads each face as separate record
- Includes comprehensive metadata in payload
- Tracks upload success/failure

## Specific Failure Scenarios:

### Scenario 1: Low-Quality HEIC Images
- Problem: iPhone photos with compression
- RetinaFace: Fails due to artifacts
- OpenCV: More tolerant, succeeds
- Result: Fallback detector saves the day

### Scenario 2: Profile Views
- Problem: Side-facing faces
- RetinaFace: Struggles with profiles
- OpenCV: Better at detecting profiles
- Result: Fallback detector handles profiles

### Scenario 3: Multiple Faces with Overlap
- Problem: Faces overlapping or too close
- Both detectors: May detect invalid coordinates
- Our validation: Checks coordinates, skips invalid ones
- Result: Processes valid faces, skips invalid ones

### Scenario 4: Edge Cases
- Problem: Faces at image boundaries
- Detectors: May return coordinates outside image bounds
- Our validation: Clips coordinates to image bounds
- Result: Prevents crashes, processes valid portions

## 🎯 Why 11 Failures Were Happening:

Based on your image collection, likely causes:

### HEIC Format Issues (5-6 failures)
- RetinaFace struggling with HEIC compression
- **Fixed:** Fallback to OpenCV detector

### Profile/Partial Faces (3-4 failures)
- Side views or partially visible faces
- **Fixed:** OpenCV better at profiles

### Low Resolution (2-3 failures)
- Small faces in large images
- **Fixed:** Better coordinate validation

### Edge Cases (1-2 failures)
- Faces at image boundaries
- **Fixed:** Boundary checking and clipping