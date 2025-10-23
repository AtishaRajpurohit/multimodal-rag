from openai import OpenAI
from loguru import logger
import base64
from dotenv import load_dotenv
import json
import string

# Load environment variables
load_dotenv()
client = OpenAI()

# -----------------------------------------------------------------------------
# Utility functions for generating image descriptions with face metadata.
# -----------------------------------------------------------------------------

def _prepare_metadata(faces):
    """
    Anonymizes faces for privacy and prepares coordinate-based metadata.
    Returns:
      - safe_faces: anonymized faces with placeholders (PersonA, PersonB, …)
      - mapping: mapping from placeholder → real label (for reinsertion)
    """
    safe_faces = []
    mapping = {}
    letters = list(string.ascii_uppercase)

    for idx, face in enumerate(faces):
        real_label = face.get("match", {}).get("label")
        placeholder = f"Person{letters[idx]}" if idx < len(letters) else f"Person{idx+1}"
        mapping[placeholder] = real_label or placeholder

        safe_faces.append({
            "id": placeholder,
            "facial_area": face.get("facial_area"),
            "face_confidence": face.get("face_confidence")
        })

    return safe_faces, mapping


def describe_image_with_faces(image_path, faces, mode="humanlike"):
    """
    Generate a humanlike textual description of an image with detected faces.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    faces : list of dict
        Detected face metadata (facial_area, face_confidence, match.label).
    mode : str, optional
        "humanlike" (default), "detailed", or "funny".
    """

    # Encode image
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    # Prepare safe anonymized metadata
    safe_faces, mapping = _prepare_metadata(faces)
    data = [{"faces": safe_faces}]

    # Prompt text (default = humanlike)
    if mode == "funny":
        prompt_text = (
            "You are a witty observer. Describe the image humorously, imagining what "
            "each anonymized person (PersonA, PersonB, etc.) might be thinking or doing. "
            "Keep it light-hearted and appropriate."
        )
    elif mode == "detailed":
        prompt_text = (
            "Provide a detailed description of the scene, including what each anonymized "
            "person (PersonA, PersonB, etc.) is wearing, doing, and how they relate to "
            "each other. Maintain order and do not rename or reorder them."
        )
    else:  # humanlike
        prompt_text = (
            "You are a vision-language assistant describing a group photo. "
            "Each detected person is anonymized (PersonA, PersonB, etc.). "
            "Use the coordinate positions (x, y, w, h) from the metadata to understand "
            "who each person is, and maintain their relative order. "
            "Do not rename or reorder them. "
            "Write a natural, coherent paragraph describing the people, their clothing, "
            "and relationships — as if you’re capturing a real moment. "
            "Group related individuals smoothly, and make it sound humanlike rather than mechanical."
        )

    # Compose GPT-4o multimodal messages
    messages = [
        {
            "role": "system",
            "content": (
                "You are a polite and observant vision assistant. "
                "You never attempt to identify real individuals. "
                "You analyze the given image and metadata to describe the scene naturally "
                "while preserving privacy."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "text", "text": json.dumps(data, indent=2)},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ],
        },
    ]

    # API call
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=600
    )

    description = response.choices[0].message.content

    # Replace anonymized placeholders with real labels (locally)
    for anon, real in mapping.items():
        if real and real != anon:
            description = description.replace(anon, real)

    return description


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sample_faces = [
        {#Back-left male, slightly behind front row; deep/royal blue outfit."
            "embedding": ["..."],
            "facial_area": {"x": 441, "y": 156, "w": 99, "h": 99},
            "face_confidence": 0.95,
            "match": {"label": "Vinayak"},
        },
        {#Front-left woman in blue patterned kurta; close to camera.
            "embedding": ["..."],
            "facial_area": {"x": 487, "y": 297, "w": 79, "h": 79},
            "face_confidence": 0.94,
            "match": {"label": "Sonali"},
        },
        {#Second woman from left in blue/white paisley.
            "embedding": ["..."],
            "facial_area": {"x": 862, "y": 278, "w": 86, "h": 86},
            "face_confidence": 0.94,
            "match": {"label": "Tala"},
        },
        {#Woman in green with gold accents/shawl.
            "embedding": ["..."],
            "facial_area": {"x": 1011, "y": 259, "w": 83, "h": 83},
            "face_confidence": 0.96,
            "match": {"label": "Atisha"},
        },
        {#Man with very short/shaved hair in white kurta.
            "embedding": ["..."],
            "facial_area": {"x": 1157, "y": 205, "w": 94, "h": 94},
            "face_confidence": 0.95,
            "match": {"label": "Matt"},
        },
        {#"Man in peach/salmon kurta.
            "embedding": ["..."],
            "facial_area": {"x": 1275, "y": 220, "w": 102, "h": 102},
            "face_confidence": 0.94,
            "match": {"label": "Avi"},
        },
        {#Far-right man in off-white/cream kurta."
            "embedding": ["..."],
            "facial_area": {"x": 1390, "y": 208, "w": 132, "h": 132},
            "face_confidence": 0.93,
            "match": {"label": "Raghav"},
        },
    ]

    try:
        caption = describe_image_with_faces(
            image_path="data/ref_images/IMG_5550.PNG",
            faces=sample_faces,
            mode="detailed",
        )
        print("\n🧾 Generated Description:\n", caption)
    except Exception as e:
        logger.error(f"Error generating description: {e}")
