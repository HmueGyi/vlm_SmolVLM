import cv2
import base64
import time
import requests
import threading

# ----------------------------
# Configuration
# ----------------------------
BASE_URL = "http://localhost:8080"
INSTRUCTION = "which object do you see?"
INTERVAL_MS = 500   # same as dropdown (100, 250, 500, 1000, 2000)

RUNNING = False
LAST_RESPONSE = ""

# ----------------------------
# Send request to server
# ----------------------------
def send_chat_completion(instruction, image_base64_url):
    url = f"{BASE_URL}/v1/chat/completions"

    payload = {
        "max_tokens": 100,
        "temperature": 0.6,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_base64_url
                        }
                    }
                ]
            }
        ]
    }

    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {e}"

# ----------------------------
# Worker thread (interval loop)
# ----------------------------
# ...existing code...
def process_loop(cap):
    global RUNNING, LAST_RESPONSE

    # target size to send (choose 1280x720 or 1920x1080 to reduce bandwidth)
    TARGET_W, TARGET_H = 1280, 720

    while RUNNING:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame")
            time.sleep(1)
            continue

        # resize to target (2K -> smaller for faster encoding & network)
        try:
            frame_send = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        except Exception:
            frame_send = frame

        # Encode frame as JPEG (quality 80 is a good default; lower => smaller)
        _, buffer = cv2.imencode(".jpg", frame_send, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        image_base64 = base64.b64encode(buffer).decode("utf-8")
        image_base64_url = f"data:image/jpeg;base64,{image_base64}"

        # Send to server
        response = send_chat_completion(INSTRUCTION, image_base64_url)
        LAST_RESPONSE = response
        print("🤖 Response:", response)

        time.sleep(INTERVAL_MS / 1000)

# ----------------------------
# Main
# ----------------------------
def main():
    global RUNNING

    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    print("📷 Camera ready")
    print("▶ Press 's' to START")
    print("⏹ Press 'q' to STOP & EXIT")

    worker = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Show webcam feed
        display = frame.copy()

        # Show last response on frame
        if LAST_RESPONSE:
            cv2.putText(
                display,
                LAST_RESPONSE[:60],
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.imshow("Camera Interaction App (Python)", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and not RUNNING:
            RUNNING = True
            print("▶ Processing started")
            worker = threading.Thread(target=process_loop, args=(cap,), daemon=True)
            worker.start()

        elif key == ord('q'):
            RUNNING = False
            print("⏹ Stopping...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
