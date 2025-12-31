from PIL import Image
from transformers import pipeline
import torch

# Optional: reduce memory fragmentation
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Load and resize image to reduce GPU memory usage
img_path = "/home/mr_robot/Desktop/To_Learn/road_to_tune/vlm_prompt_tuning/photo1.jpg"
img = Image.open(img_path).convert("RGB")
img = img.resize((224, 224))  # smaller image

# Initialize the pipeline (FP16 to save memory)
pipe = pipeline(
    "image-text-to-text",
    model="HuggingFaceTB/SmolVLM-256M-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    model_kwargs={"low_cpu_mem_usage": True}  # optional for Kaggle
)

# Ask the model to give a complete sentence description
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "Please describe this image in a complete sentence."}
        ]
    },
]

# Increase max_new_tokens for a full sentence
out = pipe(messages, max_new_tokens=128)  # allow longer output
print(out)
