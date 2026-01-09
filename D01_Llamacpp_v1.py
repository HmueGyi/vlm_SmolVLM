import cv2
import base64
import requests
import threading

import json
import time
import websocket

# Global websocket app reference (will be set in main)
WS_APP = None

# ----------------------------
# Configuration
# ----------------------------
BASE_URL = "http://localhost:8080"
LAST_FRAME = None
LOCK = threading.Lock()

# ----------------------------
# GPU support detection
# ----------------------------
USE_CUDA = False
try:
    if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        USE_CUDA = True
        try:
            cv2.cuda.setDevice(0)
        except Exception:
            pass
except Exception:
    USE_CUDA = False

if USE_CUDA:
    print("✅ OpenCV CUDA available — using GPU for image ops")
else:
    print("⚠ OpenCV CUDA not available — running on CPU")

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
                    {
                        "type": "image_url",
                        "image_url": {"url": image_base64_url}
                    }
                ]
            }
        ]
    }

    r = requests.post(url, json=payload, timeout=20)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# ----------------------------
# Camera preview loop (ALWAYS ON)
# ----------------------------
def camera_loop():
    global LAST_FRAME

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        if USE_CUDA:
            try:
                # upload to GPU
                gpu = cv2.cuda_GpuMat()
                gpu.upload(frame)
                # optional: GPU resize to reduce CPU <-> GPU transfers
                # Example: if width > 1280, downscale to 1280
                try:
                    h, w = frame.shape[:2]
                    max_w = 1280
                    if w > max_w:
                        scale = max_w / float(w)
                        new_w = max_w
                        new_h = int(h * scale)
                        gpu = cv2.cuda.resize(gpu, (new_w, new_h))
                except Exception:
                    pass

                with LOCK:
                    LAST_FRAME = gpu  # store GpuMat directly
                # Download only for display
                disp = gpu.download()
            except Exception:
                # Fallback to CPU path on any CUDA error
                with LOCK:
                    LAST_FRAME = frame.copy()
                disp = frame
        else:
            with LOCK:
                LAST_FRAME = frame.copy()
            disp = frame

        cv2.imshow("Live Camera", disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ----------------------------
# Terminal question handler (NO LOOP inference)
# ----------------------------
def question_loop():
    global LAST_FRAME

    print(" Ask question : ")
    print("🛑 Stop program with Ctrl + C")

    while True:
        question = input(" Ask : ").strip()
        if not question:
            continue

        with LOCK:
            if LAST_FRAME is None:
                print("⚠ No frame yet")
                continue
            # If GPU mat was stored, download before encoding
            if USE_CUDA and isinstance(LAST_FRAME, cv2.cuda_GpuMat):
                try:
                    frame = LAST_FRAME.download().copy()
                except Exception:
                    # fallback to CPU copy if download fails
                    frame = None
            else:
                frame = LAST_FRAME.copy()

        if frame is None:
            print("⚠ Failed to retrieve frame from GPU — skipping")
            continue

        _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        image_base64 = base64.b64encode(buffer).decode()
        image_base64_url = f"data:image/jpeg;base64,{image_base64}"

        answer = send_chat_completion(question, image_base64_url)
        print(f" Bot : {answer}")

        # act on server/model answer: yes -> send stop, no -> send rotate
        try:
            a_lower = answer.strip().lower() if isinstance(answer, str) else str(answer).lower()
        except Exception:
            a_lower = str(answer).lower()

        if a_lower in ("yes", "y"):
            # ask rosbridge to run 'stop' task
            send_service_call("stop")
        elif a_lower in ("no", "n"):
            send_service_call("left")

def on_open(ws,message):
    print("[WS] Connected to rosbridge")

    request_id = f"which_tasks_req_{int(time.time() * 1000)}"

    service_call = {
        "op": "call_service",
        "service": "/which_tasks",
        "type": "rom_interfaces/srv/WhichTasks",
        "args": {
            "request_string": "voice_command",
            "request_settings": message,
            "str_reserve2": "",
            "float_reserve1": 0.0,
            "float_reserve2": 0.0
        },
        "id": request_id
    }

    ws.send(json.dumps(service_call))
    print("[WS] Sent service call JSON")
    print(json.dumps(service_call, indent=2))

def build_service_call(request_settings):
    """Return the service_call dict used to call /which_tasks on rosbridge.

    This centralizes JSON construction so both on_open and send_service_call
    reuse the exact same payload structure.
    """
    request_id = f"which_tasks_req_{int(time.time() * 1000)}"
    return {
        "op": "call_service",
        "service": "/which_tasks",
        "type": "rom_interfaces/srv/WhichTasks",
        "args": {
            "request_string": "voice_command",
            "request_settings": request_settings,
            "str_reserve2": "",
            "float_reserve1": 0.0,
            "float_reserve2": 0.0
        },
        "id": request_id
    }

def send_service_call(request_settings):
    """Helper to send the same service_call used in on_open using the global WS_APP."""
    global WS_APP
    if WS_APP is None:
        print("[WS] No websocket available to send service call")
        return
    service_call = build_service_call(request_settings)
    try:
        WS_APP.send(json.dumps(service_call))
        print("[WS] Sent service call JSON")
        print(json.dumps(service_call, indent=2))
    except Exception as e:
        print(f"[WS] Failed to send service call: {e}")
# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":

    # Create websocket app and run it in a background thread so camera + question loops run
    ws_app = websocket.WebSocketApp(
        "ws://192.168.1.18:9090",
        on_open=on_open,
    )

    # expose globally so send_service_call can use it
    WS_APP = ws_app

    try:
        # run websocket in background
        threading.Thread(target=ws_app.run_forever, daemon=True).start()

        # start camera and question loop
        threading.Thread(target=camera_loop, daemon=True).start()
        question_loop()
    except KeyboardInterrupt:
        print("\n🛑 Exiting...")
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
    
