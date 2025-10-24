<!-- FaceMatch README This README is adapted from the Metamorphosis project template. It has been tailored to describe the FaceMatch system built in this repository. The structure of this document follows the pattern of the original reference while focusing on the actual components and workflows contained here. -->
FaceMatch: AI‑Powered Facial Recognition & Multimodal Description System
Overview

FaceMatch is a modular Python application that demonstrates how to build
end‑to‑end facial recognition workflows using modern deep‑learning and
vector‑database technologies. At its core, the project provides tools to:

detect and embed faces in images,

create a labelled reference dataset of facial embeddings,

store and query these embeddings in a Qdrant
 vector
database, and

generate natural language descriptions of group photos using
OpenAI GPT‑4o
 while
preserving the privacy of individuals.

By following the steps in this repository you can construct a reusable
reference dataset, perform face matching, and obtain rich, human‑readable
captions for your images.

Core Features
🧠 Advanced Face Detection & Embedding

Flexible format support: The FacialDetector class can read JPEG,
PNG, BMP, TIFF, WebP, HEIC and HEIF images; it transparently converts
high‑efficiency formats to RGB/BGR for processing
screenshot
.

Preprocessing pipeline: images are resized and converted on the fly
ensuring consistent input to the deep‑learning models
screenshot
.

DeepFace integration: The DeepFace

library is used to detect faces and extract 512‑dimensional embeddings.

📦 Reference Dataset Creation

Two‑phase workflow: The ReferenceDatasetCreator guides you through
cropping faces from raw images (phase 1) and then uploading the
associated embeddings with labels to Qdrant (phase 2)
screenshot
.

Automatic cropping: Each detected face is saved as a separate image
in data/reference_images_faces, allowing you to review and assign
real‑world names or labels.

Labelled uploads: Once labelled, embeddings and metadata are
uploaded to a named Qdrant collection so that subsequent queries can
return meaningful matches.
screenshot

🗃️ Vector Database Integration

Qdrant client: A dedicated VectorDB class manages the connection to
your local or remote Qdrant server. It can create and delete
collections, upsert embeddings and associated payloads, and validate
collection parameters.
screenshot

Payload‑rich points: Each stored vector includes the facial bounding
box, detection confidence, image path and the user‑provided label. This
makes it possible to filter and display results in downstream
applications.

🎨 Privacy‑Preserving Multimodal Generation

Anonymized metadata: The MultimodalImageDescriber anonymizes each
detected face with placeholders (e.g., PersonA, PersonB) before
sending the request to OpenAI. A mapping between placeholders and
real names is maintained locally so that the final caption can be
de‑anonymized after the model responds
screenshot
.

Multiple description modes: Choose between humanlike, detailed or
funny prompts. Humanlike mode yields a natural description of the
scene; detailed mode emphasizes clothing and relationships; funny mode
produces light‑hearted commentary.

OpenAI GPT‑4o integration: Uses the chat.completions API of
GPT‑4o to generate coherent paragraphs that reference detected faces
without violating privacy.

🧩 Modular Design

Clean separation of concerns: Each major functionality lives in
its own class (FacialDetector, ReferenceDatasetCreator, VectorDB,
MultimodalImageDescriber), making the code base easy to test and extend.

Extensible workflows: You can plug in your own matching algorithm or
user interface on top of these components. For example, a
CameraImageMatcher module (not included here) could combine the
detector and vector‑database layers to match faces in a live feed.

Architecture

FaceMatch implements a layered architecture that divides image
processing, data management and generative AI responsibilities. The core
layers are:

Preprocessing & Detection Layer – Handles image loading,
conversion and face detection. Built on OpenCV, Pillow and DeepFace.

Dataset Creation Layer – Splits images into cropped faces,
associates them with user‑provided labels and uploads embeddings to
Qdrant. Implemented in reference_dataset_creation.py using
ReferenceDatasetCreator and VectorDB.

Vector Database Layer – Manages the Qdrant connection and
encapsulates collection operations. Implemented in vector_db.py.

Matching Layer – (optional) Matches new embeddings against the
reference collection to find the closest identities. The logic for
this layer can be written on top of VectorDB but is not provided in
this repository.

Multimodal Generation Layer – Uses OpenAI GPT‑4o to produce
textual descriptions given an image and detected face metadata. The
MultimodalImageDescriber encapsulates prompt engineering, image
encoding and post‑processing.

System Architecture

Below is a high‑level diagram describing how the components interact. It
reflects the two primary workflows: building a reference dataset and
generating a description for a new image.

graph TB
    %% Input and pre‑processing
    subgraph "User Input"
        U[User / Script]
        I[Image Files]
    end

    subgraph "Preprocessing & Detection"
        FD[FacialDetector\nDeepFace & OpenCV]
    end

    subgraph "Dataset Creation"
        Crops[Cropped Faces]
        Label[User Labeling]
        RDC[ReferenceDatasetCreator]
    end

    subgraph "Vector Storage"
        VDB[VectorDB\nQdrant Client]
        QD[Qdrant Collection]
    end

    subgraph "Multimodal Generation"
        MID[MultimodalImageDescriber\nOpenAI GPT‑4o]
    end

    %% Data flows for dataset creation
    U --> I --> FD
    FD --> Crops
    Crops --> Label
    Label --> RDC
    RDC --> VDB
    VDB --> QD

    %% Data flows for caption generation
    I -. New Image .- FD
    FD -. Embeddings .- VDB
    VDB -. Matches .- MID
    U -. Select Mode & API Key .- MID
    MID -. Generated Caption .- U

    classDef processing fill:#e8f5e9;
    classDef storage fill:#e3f2fd;
    class FD,Crops,RDC,VDB,QD processing;
    class MID storage;

Getting Started

The following steps will allow you to run FaceMatch end to end. These
instructions assume you have Python 3.9 or higher installed and that
you are familiar with running commands in a terminal.

1. Install Dependencies

Clone the repository (if you haven’t already) and create a virtual
environment. Then install the required Python packages:

python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
pip install --upgrade pip
pip install -r requirements.txt  # Includes deepface, qdrant-client, pillow-heif, loguru, openai

2. Start a Qdrant Server

Face embeddings are stored in a Qdrant vector database. You can run
Qdrant locally using Docker:

docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant


Once running, the Qdrant dashboard is available at
http://localhost:6333/dashboard where you can inspect your collections.

3. Configure OpenAI Credentials

The MultimodalImageDescriber uses OpenAI’s GPT‑4o. Create a .env
file in the project root and add your API key:

OPENAI_API_KEY=sk-your‑api‑key


You can also pass the key programmatically when instantiating
MultimodalImageDescriber(api_key="…").

4. Phase 1 – Extract and Crop Faces

Place the images you want to include in your reference dataset in
data/query_images/. Then run the reference_dataset_creation.py
script:

python reference_dataset_creation.py


During phase 1 the script will detect faces in each image, crop them and
save them to data/reference_images_faces/. It logs how many faces
were detected per image so you can prepare labels. Open the cropped
images to verify the faces and decide which name corresponds to each
crop.

5. Phase 2 – Label and Upload

After reviewing the cropped faces, edit the all_labels list in
reference_dataset_creation.py so that each entry corresponds to a
detected face. The order of labels must match the order DeepFace
returns faces (top‑to‑bottom, left‑to‑right). Then re‑run the script
again; it will embed each face and upload the vectors and labels to the
Qdrant collection specified in collection_name.

6. Perform Face Matching (Optional)

This repository does not include a ready‑made face matching module, but
you can write your own script to query Qdrant for the nearest
neighbours of a new embedding. Conceptually, the steps are:

Use FacialDetector.preprocess_image() and
FacialDetector.facial_detection_embedding() on a new image.

For each embedding, call client.search() on the Qdrant collection to
retrieve the closest stored embeddings.

Compare the distances to determine the best match. See
Qdrant’s documentation
 for
examples of vector search.

7. Generate Multimodal Descriptions

To create a natural‑language caption for an image you have processed and
labelled, use the MultimodalImageDescriber class. A typical
workflow looks like this:

from your_module.detect import FacialDetector
from your_module.vector_db import VectorDB
from your_module.rev_multimodal_generation import MultimodalImageDescriber

# Detect faces and extract embeddings from a new image
image_path = "data/new_images/group_photo.jpg"
detector = FacialDetector(image_path)
processed = detector.preprocess_image(resize=(512, 512))
faces = detector.facial_detection_embedding(img_array=processed)

# Optional: match faces against your Qdrant collection to add labels
collection_name = "reference_dataset_collection"
vector_db = VectorDB()
matches = vector_db.client.search(
    collection_name=collection_name,
    query_vector=faces[0]["embedding"],
    top=5
)
# Build a faces list with a `match.label` key for each face based on the search results

# Generate a caption
describer = MultimodalImageDescriber()
caption = describer.describe_image_with_faces(
    image_path=image_path,
    faces=faces,  # include match labels if available
    mode="humanlike"
)
print(caption)


Remember to set your OPENAI_API_KEY in the environment or pass it
directly when creating MultimodalImageDescriber. The faces
structure must include facial_area, face_confidence and a
match.label field for de‑anonymization.

Implementation Status

✅ Completed Features

Facial detection, embedding extraction and preprocessing using DeepFace,
OpenCV and Pillow.

HEIC/HEIF support via pillow‑heif and automatic format conversion.

Two‑phase reference dataset creation with face cropping and labelled
embedding upload to Qdrant.

Qdrant vector database integration for collection management and
upserts.

Privacy‑preserving multimodal caption generation using OpenAI GPT‑4o
with multiple description modes and placeholder replacement.

🚧 Future Work

Implementation of a CameraImageMatcher to match faces from live
camera feeds against the reference dataset.

Streamlit or FastAPI user interface for interactive face matching and
captioning.

Extended support for additional embedding models (e.g. ArcFace vs.
Facenet), distance metrics and threshold tuning.

Unit tests and benchmarking of the full pipeline on diverse datasets.

Technical Notes

The code adheres to good Python practices: types are annotated using
typing, logs are emitted via Loguru
,
and environment variables are loaded with
python‑dotenv
. DeepFace is
configured to align faces and normalize embeddings (ArcFace
normalization) by default. Qdrant collections use a vector size of
512 and cosine distance but these parameters can be changed when
calling create_collection() in VectorDB. The base64 encoding and
prompt templates used by MultimodalImageDescriber follow best
practices recommended by OpenAI for vision inputs.

Feel free to explore the detect.py, reference_dataset_creation.py,
vector_db.py and rev_multimodal_generation.py files to understand
the internal implementations and adapt them to your own projects.