from openai import OpenAI
from loguru import logger
import base64
from dotenv import load_dotenv
import json

#Instantiating required modules
load_dotenv()
client = OpenAI()


# -----------------------------------------------------------------------------
# Utility functions for generating image descriptions with face metadata.
#
# The GPT‑4o model has strict privacy safeguards and will not identify individuals
# by name in photos【451525781550281†L70-L92】. To respect these guidelines while
# still producing useful scene descriptions, we generate placeholder IDs
# (Person1, Person2, …) for each detected face and only send those to GPT‑4o.
# After receiving the model’s description, we replace the placeholders with the
# user‑provided labels locally. This ensures that personal information is never
# transmitted to the model while still allowing downstream use of names.
#
# A high‑level API for your application: call `describe_image_with_faces()`
# with your image path, the detected faces metadata, and a mode ("default",
# "detailed", or "funny"). The function returns a caption string with the
# placeholders substituted back to your provided names.
# -----------------------------------------------------------------------------

def _prepare_metadata(faces):
    """
    Given a list of face detection results, return two structures:

    1. safe_faces: a list of dicts with only placeholder IDs, facial_area and confidence.
       Each face will have an "id" field like "Person1" instead of the original name.
    2. mapping: a dict mapping the placeholder ID back to the original label.

    The mapping is used after the GPT call to replace placeholder names with
    actual labels in the final description. We do **not** send the original
    labels to GPT‑4o to respect its privacy policies【451525781550281†L70-L92】.
    """
    safe_faces = []
    mapping = {}
    for idx, face in enumerate(faces):
        placeholder = f"Person{idx + 1}"
        # store mapping from placeholder to original label (if present)
        label = face.get("match", {}).get("label") or placeholder
        mapping[placeholder] = label
        safe_faces.append({
            "id": placeholder,
            "facial_area": face.get("facial_area"),
            "face_confidence": face.get("face_confidence")
        })
    return safe_faces, mapping


def describe_image_with_faces(image_path, faces, mode="default"):
    """
    Generate a textual description of an image with detected faces using GPT‑4o.

    Parameters
    ----------
    image_path : str
        Local path to the image to analyze. It will be base64‑encoded and sent
        to the OpenAI API. Make sure this is the *same image* used for face
        detection so that the coordinates correspond correctly.
    faces : list of dict
        Face detection metadata. Each entry should contain at least a
        `facial_area` (dict with keys x, y, w, h) and optionally a `match` dict
        with a `label`. See the example in the README for the expected format.
    mode : str, optional
        One of {"default", "detailed", "funny"}. This controls the tone and
        level of detail in the generated caption. Unknown modes fall back to
        "default".

    Returns
    -------
    str
        A description of the image with placeholder IDs substituted back to
        the original labels. If the API call fails, the exception will be
        propagated to the caller.

    Example
    -------
    >>> faces = [
    ...     {"facial_area": {"x": 100, "y": 50, "w": 200, "h": 250},
    ...      "face_confidence": 0.95,
    ...      "match": {"label": "Atisha"}}
    ... ]
    >>> describe_image_with_faces("temp.png", faces, mode="detailed")
    "Atisha is sitting at a table …"
    """
    # Prepare the base64 image
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    # Prepare safe face metadata and mapping for post‑processing
    safe_faces, mapping = _prepare_metadata(faces)
    data = [{"faces": safe_faces}]

    # Select prompt based on mode
    mode = (mode or "default").lower()
    if mode == "detailed":
        prompt_text = (
            "Provide a detailed scene description, including clothing, actions "
            "and relationships between the individuals. Refer to people using "
            "the placeholder IDs (e.g. Person1, Person2). Do not guess their "
            "real names."
        )
    elif mode == "funny":
        prompt_text = (
            "Describe the image humorously. Inject some light‑hearted humor "
            "about what the individuals (Person1, Person2, etc.) might be "
            "thinking or doing, but keep it appropriate."
        )
    else:
        prompt_text = (
            "Describe the image concisely. Mention what each person (Person1, "
            "Person2, etc.) is doing or expressing."
        )

    # Compose the chat messages. We instruct the model not to identify people by
    # name and to respect privacy. The user message includes our prompt, the
    # sanitized JSON metadata, and the image.
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful vision‑language assistant. You analyze images "
                "using provided structured metadata. Do not identify or infer "
                "real people’s names or personal information from faces. Use "
                "only the placeholder IDs provided (e.g. Person1, Person2) to "
                "refer to individuals in the scene."
            )
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "text", "text": json.dumps(data, indent=2)},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
            ]
        }
    ]

    # Make the API call
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=400
    )

    # Extract the description
    description = response.choices[0].message.content

    # Replace placeholders with original labels
    for placeholder, label in mapping.items():
        description = description.replace(placeholder, label)

    return description


# Example usage (this block can be removed or adapted to integrate into your app)
if __name__ == "__main__":
    # Sample face detection metadata
    sample_faces = [
        {
            "embedding": ["..."],  # truncated for brevity
            "facial_area": {"x": 100, "y": 50, "w": 200, "h": 250},
            "face_confidence": 0.95,
            "match": {"id": 123, "label": "Atisha", "score": 0.87}
        }
    ]
    try:
        caption = describe_image_with_faces(
            image_path="data/ref_images/IMG_5550.PNG",
            faces=sample_faces,
            mode="default",
        )
        print(caption)
    except Exception as e:
        logger.error(f"Error generating description: {e}")
