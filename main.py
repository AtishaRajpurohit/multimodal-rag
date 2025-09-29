
# %% [markdown]
# # Image Matching with CLIP and BLIP
# This notebook demonstrates image matching using CLIP embeddings

# %%
# Importing Libraries
import clip
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
from PIL import Image


# %%
# Loading the models
device = "cuda" if torch.cuda.is_available() else "cpu"
'''
Loading CLIP - Downloading the 'ViT-B/32' version of Vision Transformer Base with 32x32 patch size
Returns a tuple of model and preprocess
'''
model, preprocess = clip.load("ViT-B/32", device=device)
'''Loading BLIP - Loading the Bootstrapping Language Image Pre-training model
'''
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)


# %%
# Processing Reference Images
reference_embeddings = []
reference_names = []
ref_dir = "ref_images"

# Check if directory exists
if not os.path.exists(ref_dir):
    print(f"Error: Directory '{ref_dir}' not found!")
    exit(1)

for file in os.listdir(ref_dir):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        #Preprocessing every image
        img = preprocess(Image.open(os.path.join(ref_dir,file))).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(img)
        reference_embeddings.append(emb)
        reference_names.append(file)

reference_embeddings = torch.cat(reference_embeddings)
#Standardising Embeddings - Is it needed ? 
# reference_embeddings = reference_embeddings / reference_embeddings.norm(dim=-1, keepdim=True)
print(f"Embedded {len(reference_names)} reference images")
print(reference_embeddings.shape)
print(f"Embedding dimension: {reference_embeddings.shape[1]}")

# %%
# Encoding Query Image and Finding Similar Images
query_dir = "query_images"
query_file = "IMG_2747.JPG"  # Your query image

# Check if query image exists
query_path = os.path.join(query_dir, query_file)
if not os.path.exists(query_path):
    print(f"Error: Query image '{query_path}' not found!")
    exit(1)

# Process query image
print(f"Processing query image: {query_file}")
query_img = preprocess(Image.open(query_path)).unsqueeze(0).to(device)

with torch.no_grad():
    query_embedding = model.encode_image(query_img)

# Calculate similarities
similarities = torch.cosine_similarity(query_embedding, reference_embeddings, dim=1)
similarities = similarities.cpu().numpy()

# Find best matches
best_match_idx = similarities.argmax()
best_similarity = similarities[best_match_idx]

print(f"\nBest match: {reference_names[best_match_idx]}")
print(f"Similarity score: {best_similarity:.4f}")

# Show all similarities
print("\nAll similarities:")
for i, (name, sim) in enumerate(zip(reference_names, similarities)):
    print(f"{name}: {sim:.4f}") 