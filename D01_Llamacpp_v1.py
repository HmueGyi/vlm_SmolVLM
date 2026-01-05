import cv2
import base64
import requests

# ----------------------------
# Configuration
# ----------------------------
BASE_URL = "http://localhost:8080"

# ----------------------------
# Send request to server
# ----------------------------
def send_chat_completion(instruction, image_base64_url):
    url = f"{BASE_URL}/v1/chat/completions"

    payload = {
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": image_base64_url}}
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
# Main
# ----------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    print("📷 Camera ready")
    print("💬 Type your question and press Enter (or 'q' to quit)")

    while True:
        instruction = input("❓ Your question: ").strip()
        if instruction.lower() == 'q':
            print("⏹ Exiting...")
            break
        if not instruction:
            continue

        # Capture one frame
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame")
            continue

        # Encode frame as JPEG
        _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        image_base64 = base64.b64encode(buffer).decode("utf-8")
        image_base64_url = f"data:image/jpeg;base64,{image_base64}"

        # Send to server
        response = send_chat_completion(instruction, image_base64_url)
        print("🤖 Response:", response)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
