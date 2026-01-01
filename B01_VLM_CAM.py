import cv2
import time
from PIL import Image
from transformers import pipeline
import torch
import os
import threading

# ----------------------------
# Device & performance config
# ----------------------------
use_gpu = torch.cuda.is_available()
device = 0 if use_gpu else -1
dtype = torch.float16 if use_gpu else torch.float32

if not use_gpu:
    torch.set_num_threads(4)   # IMPORTANT for CPU

print(f"Running on: {'GPU' if use_gpu else 'CPU'}")

# Reduce memory fragmentation (GPU only)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ----------------------------
# Load SmolVLM
# ----------------------------
pipe = pipeline(
    "image-text-to-text",
    model="HuggingFaceTB/SmolVLM-256M-Instruct",
    torch_dtype=dtype,
    device=device,
    model_kwargs={"low_cpu_mem_usage": True}
)

# ----------------------------
# Webcam
# ----------------------------
cap = cv2.VideoCapture(0)
frame_lock = threading.Lock()
current_frame = None

# ----------------------------
# Webcam thread with FPS
# ----------------------------
def capture_frames():
    global current_frame

    prev_time = 0
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # FPS calculation
        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        with frame_lock:
            current_frame = frame.copy()

        cv2.imshow("Webcam", frame)
        cv2.waitKey(1)

# Start webcam thread
threading.Thread(target=capture_frames, daemon=True).start()

print("Type a question and press Enter (type 'q' to quit)")

# ----------------------------
# Main loop (VLM inference)
# ----------------------------
while True:
    user_question = input("Your question: ")
    if user_question.lower() in ["q", "quit", "exit"]:
        break

    with frame_lock:
        if current_frame is None:
            print("No frame available yet")
            continue

        image = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        # Smaller image for CPU
        image = image.resize((224, 224) if not use_gpu else (224, 224))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        "Describe the image briefly with useful information.\n"
                        "Use one short sentence.\n"
                        "No extra explanation.\n\n"
                        f"{user_question}"
                    )
                }
            ]
        }
    ]

    start = time.time()
   
    output = pipe(
        messages,
        max_new_tokens=24,
        do_sample=False
    )

    response_time = time.time() - start

    infer_time = time.time() - start

    gen = output[0]["generated_text"]
    answer = gen[-1]["content"] if isinstance(gen, list) else gen
    print(answer.strip())
    print(f"\nResponse time: {response_time:.2f}s\n")

# ----------------------------
# Cleanup
# ----------------------------
cap.release()
cv2.destroyAllWindows()
# ----------------------------