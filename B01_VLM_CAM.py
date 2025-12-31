import cv2
from PIL import Image
from transformers import pipeline
import torch
import os

# Reduce memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Load SmolVLM pipeline
pipe = pipeline(
    "image-text-to-text",
    model="HuggingFaceTB/SmolVLM-256M-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    model_kwargs={"low_cpu_mem_usage": True}
)

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

print("Press 'q' to quit, 'c' to capture and describe frame")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show webcam feed
    cv2.imshow("Webcam", frame)
    frame = cv2.flip(frame, flipCode=1)    # Mirror image

    key = cv2.waitKey(1) & 0xFF

    # Press 'c' to capture frame and run VLM
    if key == ord("c"):
        # Convert OpenCV (BGR) → PIL (RGB)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((224, 224))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": "what do you see"
                    }
                ]
            }
        ]

        output = pipe(messages, max_new_tokens=64)
        print("VLM:", output[0]["generated_text"])

    # Quit
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
