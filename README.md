# vlm_SmolVLM

A minimal Visual-Language Model (VLM) demo. This repository contains two example scripts:

- `A01_test_VLM.py` — single-image VLM inference (example uses `photo1.jpg`).
- `B01_VLM_CAM.py` — interactive webcam demo: captures frames and lets you type questions in the terminal to describe the current frame.

Files and purpose

- `A01_test_VLM.py`
  - Loads an image, resizes it to reduce memory use, and runs the SmolVLM `image-text-to-text` pipeline.
  - Note: the script contains an absolute image path. Either update `img_path` to point to `./photo1.jpg` in this repo or edit it to your image location.
  - Example output: printed pipeline result (JSON-like list).

- `B01_VLM_CAM.py`
  - Opens the default webcam, shows a live window, and runs a background thread to capture frames.
  - In the terminal you can type a question (for example: "What is in the image?") and the script will send the latest frame + text prompt to the VLM and print the response.
  - Press `q` in the webcam window to stop the camera, or type `quit` / `q` / `exit` in the terminal to exit.

Quickstart

1. Create and activate a Python virtual environment (recommended):

   python -m venv .venv
   source .venv/bin/activate

2. Install dependencies:

   pip install -r requirements.txt

3. (Optional) Install a PyTorch wheel matching your CUDA/CPU setup if needed.

Running the examples

- Run the single-image script (make sure `img_path` points to an existing image):

  python A01_test_VLM.py

- Run the webcam demo (requires a working webcam and OpenCV):

  python B01_VLM_CAM.py

Headless / terminal-only usage

- A01: run and save output to a file without opening any GUI windows:

  python A01_test_VLM.py > results.txt

  If the script tries to open windows, remove or comment out `cv2.imshow` / `cv2.waitKey` in the script.

- B01 on a headless server:
  - Use a virtual X server: `xvfb-run -s "-screen 0 1400x900x24" python B01_VLM_CAM.py`.
  - Or modify `B01_VLM_CAM.py` to skip `cv2.imshow` and instead save frames or only print model outputs.

Device selection and memory tips

- Scripts attempt to use CUDA if available (via `device_map="auto"`) and fall back to CPU.
- To force CPU, set the environment variable before running:

  export CUDA_VISIBLE_DEVICES=""

- Reduce memory usage: scripts set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and use `torch_dtype=torch.float16`. Further reduce size by resizing images and lowering `max_new_tokens`.

Troubleshooting

- If `A01_test_VLM.py` fails to find the image, update `img_path` to the correct path or place `photo1.jpg` next to the script and set `img_path = "./photo1.jpg"`.
- If the webcam is not seen, try a different device index in `cv2.VideoCapture(0)` (e.g. `1`) or verify permissions.
- For CUDA / PyTorch errors, install the correct PyTorch build for your CUDA version from https://pytorch.org.

License

MIT (change as needed).