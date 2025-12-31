import cv2
from PIL import Image
from transformers import pipeline
import torch
import os
import threading

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
frame_lock = threading.Lock()
current_frame = None

print("Type a question and press Enter to describe the current frame. Type 'quit' to exit.")

# Thread to continuously read webcam frames
def capture_frames():
    global current_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, flipCode=1)  # Mirror image
        with frame_lock:
            current_frame = frame.copy()
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Start the webcam thread
threading.Thread(target=capture_frames, daemon=True).start()

# Main loop for user input
while True:
    user_question = input("Your question: ")
    if user_question.lower() in ["quit", "q", "exit"]:
        break

    with frame_lock:
        if current_frame is None:
            print("No frame available yet!")
            continue
        image = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((224, 224))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_question}
            ]
        }
    ]

    output = pipe(messages, max_new_tokens=64)
    print("VLM:", output[0]["generated_text"])

cap.release()
cv2.destroyAllWindows()
