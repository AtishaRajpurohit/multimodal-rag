from openai import OpenAI
from loguru import logger
import base64
from dotenv import load_dotenv
import json

#Instantiating required modules
load_dotenv()
client = OpenAI()


# import cv2
# width, height = cv2.imread("temp_image.png").shape[:2][::-1]  # [::-1] swaps width/height
# print(f"Image size: {width}x{height}")


#Opening Image
image_path = "data/ref_images/IMG_5550.PNG"
with open(image_path, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

#Temporary dataset
data = [
  {
    "image_path": "temp_image.png",
    "faces": [
      {
        "embedding": ["..."],  # you can omit or keep truncated
        "facial_area": {"x": 100, "y": 50, "w": 200, "h": 250},
        "face_confidence": 0.95,
        "match": {"id": 123, "label": "Atisha", "score": 0.87}
      }
    ]
  }
]

messages = [
    {
        "role": "system",
        "content": "You are a vision-language model that interprets images using structured metadata. "
                   "You output clear, factual scene descriptions."
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Analyze the following image. "
                    "Use the provided JSON metadata (face bounding boxes, names, and confidences) "
                    "to describe what the identified people are doing or expressing."
                )
            },
            {
                "type": "text",
                "text": json.dumps(data, indent=2)  # 👈 Pass your structured JSON directly
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"}
            }
        ]
    }
]


response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    max_tokens=400
)

print(response.choices[0].message.content)
