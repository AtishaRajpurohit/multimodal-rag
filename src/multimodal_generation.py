from openai import OpenAI
from loguru import logger
import base64
# client = OpenAI()

#Temporary
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
                "type": "input_text",
                "text": json.dumps(data, indent=2)  # 👈 Pass your structured JSON directly
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"}
            }
        ]
    }
]


#Opening Image
image_path = data[0]["image_path"]
with open(image_path, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")


response = client.chat.completions.create(
    model="gpt-4o",     # or "gpt-4-turbo" if multimodal enabled
    messages=messages,
    max_tokens=400
)

print(response.choices[0].message.content)
