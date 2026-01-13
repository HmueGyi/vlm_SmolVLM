# SmolVLM Demo Project

Small demos for running SmolVLM variants and integrations. This README gives a short, easy-to-scan overview and quickstart steps.

Quick links
- Examples: A01/B01/C01/D01/D02
- Myanmar translation: [README_myanmar.md](README_myanmar.md)

Prerequisites
- Python 3.10+ (use a venv)
- Install a matching torch wheel for your machine (see requirements.txt comments)

Quickstart (Linux)
1. Create and activate a venv
   python3 -m venv venv
   source venv/bin/activate

2. Install dependencies
   - First install the appropriate torch/torchvision wheel for your CUDA (or CPU) build (see comments at top of requirements.txt)
   - Then:
     pip install --upgrade pip
     pip install -r requirements.txt

3. (Optional) If you need ONNX runtime for CPU inference:
   pip install onnxruntime

Important files / folders
- requirements.txt — dependency notes and packages

Scripts (short descriptions)
- A01_test_VLM.py
  Purpose: Run a single-image pipeline and print a text description.
  Usage: adjust img_path then run `python A01_test_VLM.py`.

- B01_VLM_CAM.py
  Purpose: Display webcam frames and run SmolVLM on the latest frame when you type a question at the terminal prompt.
  Usage: `python B01_VLM_CAM.py` → type your question at the prompt and press Enter. Type 'q' or 'quit' to exit.
  Notes: The webcam window is for display only; inference is triggered from the terminal prompt.

- C01_ONNX_VLM.py
  Purpose: Run exported ONNX vision/embed/decoder sessions via onnxruntime (CPU). Script is under active development.
  Usage: `python C01_ONNX_VLM.py` and follow prompts.

- D01_Llamacpp_v1.py
  Purpose: Capture one webcam frame and send it with a text instruction to a local llama.cpp HTTP server (`/v1/chat/completions`).
  Usage: Start the llama.cpp server, then run `python D01_Llamacpp_v1.py`.

- D02_Llamacpp_v2.py
  Purpose: Continuously capture frames and periodically send them to a llama.cpp server; shows the latest response on the webcam window.
  Usage: Start the server, then run `python D02_Llamacpp_v2.py`.

llama.cpp HTTP server (brief)
- The D01/D02 clients expect a llama.cpp-compatible HTTP server supporting `/v1/chat/completions` (release b5394 tested).
- Start the server and set BASE_URL in the Python scripts (default: http://localhost:8080).
- Example start command (server-side):
1. Install [llama.cpp](https://github.com/ggml-org/llama.cpp)
2. In cmd ` cd build/bin` and then 
   Run `./llama-server -hf ggml-org/Qwen3-VL-2B-Instruct-GGUF --host 0.0.0.0 --port 8080 -ngl 99`  
   Note: you may need to add `-ngl 99` to enable GPU (if you are using NVidia/AMD/Intel GPU)  
   Note (2): You can also try other models [here](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)


Run examples (commands)
- python A01_test_VLM.py
- python B01_VLM_CAM.py
- python C01_ONNX_VLM.py
- Start server -> python D01_Llamacpp_v1.py
- Start server -> python D02_Llamacpp_v2.py

Troubleshooting (common checks)
- Camera: ensure OpenCV can access /dev/video0 and you have permissions.
- Model download/auth: if model downloads fail, run `huggingface-cli login` or use local paths.
- Memory/OOM: reduce image sizes or use CPU/FP16 where appropriate.
- ONNX shape errors: check printed decoder input names in C01_ONNX_VLM.py and adapt feed keys.
- Server connection: verify `curl http://localhost:8080/health` and adjust BASE_URL.
