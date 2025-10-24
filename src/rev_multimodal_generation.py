from openai import OpenAI
from loguru import logger
import base64
from dotenv import load_dotenv
import json
import string
from typing import List, Dict, Tuple, Optional


class MultimodalImageDescriber:
    """
    A class for generating image descriptions with face metadata using OpenAI's GPT-4o model.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the MultimodalImageDescriber.
        
        Args:
            api_key: OpenAI API key. If None, will use environment variable.
        """
        load_dotenv()
        self.client = OpenAI(api_key=api_key)
        self.letters = list(string.ascii_uppercase)
    
    def _prepare_metadata(self, faces: List[Dict]) -> Tuple[List[Dict], Dict[str, str]]:
        """
        Anonymizes faces for privacy and prepares coordinate-based metadata.
        
        Args:
            faces: List of face dictionaries with match information
            
        Returns:
            Tuple of (safe_faces, mapping) where:
            - safe_faces: anonymized faces with placeholders (PersonA, PersonB, etc.)
            - mapping: mapping from placeholder to real label
        """
        safe_faces = []
        mapping = {}
        
        for idx, face in enumerate(faces):
            real_label = face.get("match", {}).get("label")
            placeholder = f"Person{self.letters[idx]}" if idx < len(self.letters) else f"Person{idx+1}"
            mapping[placeholder] = real_label or placeholder
            
            safe_faces.append({
                "id": placeholder,
                "facial_area": face.get("facial_area"),
                "face_confidence": face.get("face_confidence")
            })
        
        return safe_faces, mapping
    
    def _get_prompt_text(self, mode: str) -> str:
        """
        Get the appropriate prompt text based on the mode.
        
        Args:
            mode: Description mode ("humanlike", "detailed", or "funny")
            
        Returns:
            Formatted prompt text
        """
        if mode == "funny":
            return (
                "You are a witty observer. Describe the image humorously, imagining what "
                "each anonymized person (PersonA, PersonB, etc.) might be thinking or doing. "
                "Keep it light-hearted and appropriate."
            )
        elif mode == "detailed":
            return (
                "Provide a detailed description of the scene, including what each anonymized "
                "person (PersonA, PersonB, etc.) is wearing, doing, and how they relate to "
                "each other. Maintain order and do not rename or reorder them."
            )
        else:  # humanlike
            return (
                "You are a vision-language assistant describing a group photo. "
                "Each detected person is anonymized (PersonA, PersonB, etc.). "
                "Use the coordinate positions (x, y, w, h) from the metadata to understand "
                "who each person is, and maintain their relative order. "
                "Do not rename or reorder them. "
                "Write a natural, coherent paragraph describing the people, their clothing, "
                "and relationships — as if you're capturing a real moment. "
                "Group related individuals smoothly, and make it sound humanlike rather than mechanical."
            )
    
    def _encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 string.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded image string
        """
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def _build_messages(self, prompt_text: str, safe_faces: List[Dict], image_b64: str) -> List[Dict]:
        """
        Build the messages array for the OpenAI API call.
        
        Args:
            prompt_text: The prompt text for the model
            safe_faces: Anonymized face metadata
            image_b64: Base64 encoded image
            
        Returns:
            Messages array for OpenAI API
        """
        data = [{"faces": safe_faces}]
        
        return [
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
    
    def _replace_placeholders(self, description: str, mapping: Dict[str, str]) -> str:
        """
        Replace anonymized placeholders with real names in the description.
        
        Args:
            description: The generated description text
            mapping: Mapping from placeholder to real name
            
        Returns:
            Description with real names restored
        """
        for anon, real in mapping.items():
            if real and real != anon:
                description = description.replace(anon, real)
        return description
    
    def describe_image_with_faces(self, image_path: str, faces: List[Dict], mode: str = "humanlike") -> str:
        """
        Generate a textual description of an image with detected faces.

        Args:
            image_path: Path to the image file
            faces: List of detected face metadata (facial_area, face_confidence, match.label)
            mode: Description mode ("humanlike", "detailed", or "funny")

        Returns:
            Generated description string with real names restored
        """
        try:
            # Encode image
            image_b64 = self._encode_image(image_path)
            
            # Prepare safe anonymized metadata
            safe_faces, mapping = self._prepare_metadata(faces)
            
            # Get prompt text
            prompt_text = self._get_prompt_text(mode)
            
            # Build messages
            messages = self._build_messages(prompt_text, safe_faces, image_b64)
            
            # Make API call
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=600
            )
            
            description = response.choices[0].message.content
            
            # Replace anonymized placeholders with real labels
            description = self._replace_placeholders(description, mapping)
            
            return description
            
        except Exception as e:
            logger.error(f"Error generating description: {e}")
            raise


# -----------------------------------------------------------------------------
# Example usage with actual face detection
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Import the face detection function
    from .camera_image_matching import CameraImageMatcher
    
    # Initialize the describer
    describer = MultimodalImageDescriber()
    
    try:
        # Get face detection results from the actual image
        image_path = "data/ref_images/IMG_5550.PNG"
        collection_name = "reference_dataset_collection"
        
        logger.info(f"Processing image: {image_path}")
        matcher = CameraImageMatcher()
        all_results = matcher.process_single_image(image_path=image_path, collection_name=collection_name)
        
        # Extract faces from the results
        if all_results and len(all_results) > 0:
            faces = all_results[0]["faces"]
            logger.info(f"Detected {len(faces)} faces")
            
            # Generate description using the detected faces
            caption = describer.describe_image_with_faces(
                image_path=image_path,
                faces=faces,
                mode="humanlike"
            )
            
            print("\nGenerated Description:")
            print(caption)
            
        else:
            logger.warning("No faces detected in the image")
            
    except Exception as e:
        logger.error(f"Error in example execution: {e}")