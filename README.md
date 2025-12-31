# vlm_SmolVLM

A tiny Visual-Language Model (VLM) demo repository with two example scripts:

- `A01_test_VLM.py` — run model inference on a single image (`photo1.jpg` by default).
- `B01_VLM_CAM.py` — run a webcam demo that captures frames and runs the VLM in real time.

Prerequisites

- Python 3.8 or newer
- A suitable PyTorch wheel for your platform (CUDA-enabled if you have an NVIDIA GPU, or CPU-only)

Installation

1. Create and activate a virtual environment (recommended):

   python -m venv .venv && source .venv/bin/activate

2. Install the dependencies listed in `requirements.txt`.

   pip install -r requirements.txt

Note about PyTorch

PyTorch should be installed according to your CUDA/CPU setup. For example, for CUDA 11.8:

   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

If you don't have a GPU, install the CPU-only build (follow instructions on https://pytorch.org).

Usage

- Run inference on the example image:

  python A01_test_VLM.py

- Run the webcam demo (requires a working webcam and OpenCV):

  python B01_VLM_CAM.py

Files

- `A01_test_VLM.py` — single-image VLM inference example.
- `B01_VLM_CAM.py` — webcam capture + VLM inference example.
- `photo1.jpg` — example image used by `A01_test_VLM.py`.
- `requirements.txt` — Python dependencies for the project.

Notes

- Some advanced features (8-bit quantization) are optional and require additional packages (see `requirements.txt` comments).
- This project is intended as a small demo and starting point — adjust model paths and device selection in the scripts as needed.

License

- MIT (adjust as needed).