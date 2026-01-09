# SmolVLM Demo Project (မြန်မာ)

SmolVLM ကို စမ်းသပ်ရန်နှင့် အသုံးပြုရန် သက်သာလွယ်ကူစေသော README (မြန်မာဘာသာ)

## စာချုပ်
ဤrepo သည် SmolVLM မော်ဒယ်များနှင့် အမျိုးမျိုးသော demo scripts များကို စမ်းသပ်ရန် အတွက် ရည်ရည်ရွယ်ထားသည်။ အောက်တွင် scripts များ၏ အကျဉ်းချုပ်၊ စတင်အသုံးပြုရန် လမ်းညွှန်ချက်များနှင့် အသုံးပြုမှုအကြံပြုချက်များ ပါရှိသည်။

## Scripts အကျဉ်းချုပ်

- A01_test_VLM.py — တစ်ပုံတည်း ဖြင့် စမ်းသပ်ခြင်း
  - ရည်ရွယ်ချက်: ပုံတစ်ပုံကို load လုပ်၍ SmolVLM Hugging Face pipeline ဖြင့် စာရှင်းလင်းချက် (caption/response) ထုတ်ရန်။
  - အသုံးပြုနည်း: img_path ကို ပြင်ပြီး `python A01_test_VLM.py` ကို run ပါ။
  - မှတ်ချက်: single-shot request ဖြစ်ပြီး webcam မလိုအပ်ပါ။ Hugging Face credentials ထည့်ရန် လိုအပ်နိုင်သည်။ GPU ရှိပါက အမြန်နှုန်းပိုကောင်းသည်။

- B01_VLM_CAM.py — Webcam ဖြင့် စမ်းသပ်ခြင်း (အပြန်အလှန်)
  - ရည်ရွယ်ချက်: Webcam ထံမှ frame များကို ဖမ်းယူပြီး SmolVLM မော်ဒယ်နှင့် အပြန်အလှန် ဆက်သွယ်ရန်။
  - အသုံးပြုနည်း: `python B01_VLM_CAM.py` ကို run ပါ၊ prompt တွင် မေးခွန်းများ ရိုက်ထည့်နိုင်သည်။
  - မှတ်ချက်: OpenCV ဖြင့် camera access လုပ်ရန် permission လိုအပ်သည်။ latency ကို လျော့ချရန် GPU အသုံးပြုရန် အကြံပြုသည်။

- C01_ONNX_VLM.py — ONNX Runtime (CPU) (အဆင့်မြှင့်နေဆဲ)
  - ရည်ရွယ်ချက်: ONNX export ပြီးသော sessions (vision, embed, decoder) ကို onnxruntime ဖြင့် CPU ပေါ်တွင် အသုံးပြုရန်။ decoder ကို စစ်ဆေးနေဆဲ ဖြစ်တယ်။ out က မှား နေတယ်။
  - အသုံးပြုနည်း: `python C01_ONNX_VLM.py` ကို run ပြီး prompt တွင် မေးခွန်းများ ထည့်ပါ။
  - မှတ်ချက်: ဖိုင်များနှင့် input/output shapes ကို သေချာစစ်ပါ။

- D01_Llamacpp_v1.py — llama.cpp သို့ single request/response
  - ရည်ရွယ်ချက်: webcam frame တစ်ခုကို စာသားညွှန်ကြားချက်နှင့်အတူ local llama.cpp HTTP server ရဲ့ `/v1/chat/completions` endpoint သို့ ပို့ရန်။
  - အသုံးပြုနည်း: server ကို စတင်ထားပြီး `python D01_Llamacpp_v1.py` ကို run ၍ prompt တွင် မေးခွန်း ထည့်ပါ။

- D02_Llamacpp_v2.py — Looping requests/responses to llama.cpp
  - ရည်ရွယ်ချက်: frame များကို ဆက်တိုက် ဖမ်းယူ၍ တိတိကျကျ အချိန်အကွာအဝေး (interval) အလိုက် server သို့ ပို့ပြီး latest response ကို webcam window ပေါ်တွင် ပြရန်။
  - အသုံးပြုနည်း: server ကို စတင်ထားပြီး `python D02_Llamacpp_v2.py` မှာ 's' ကို ဖိ၍ loop စတင်၊ 'q' ကို ဖိ၍ ရပ်ပါ။

## လိုအပ်သောဖိုင်များ
- requirements.txt — Python dependency များ

## Linux ပေါ်တွင် အမြန်စတင်ရန်
1) virtual environment ဖန်တီး၍ ဖွင့်ပါ

   python3 -m venv venv
   source venv/bin/activate

2) pip ကို အပ်ဒိတ်လုပ်ပြီး dependency များ install လုပ်ပါ

   pip install --upgrade pip
   pip install -r requirements.txt

3) ONNX Runtime (CPU) အသုံးပြုရန်

   pip install onnxruntime

4) (Optional) NVIDIA GPU ရှိပါက CUDA-enabled PyTorch ကို အတည်ပြုရန် — https://pytorch.org/get-started/locally/

## llama.cpp HTTP server (အကျဉ်း)
- D01 နှင့် D02 scripts များအတွက် `/v1/chat/completions` ကို support ပြုသော local HTTP server တစ်ခု လိုအပ်သည် (llama.cpp-compatible server, release b5394 ကို ညွှန်ပြထားသည်)
- GGUF model ဖိုင်များကို `llamacpp_model/` ထဲသို့ ထည့်ပြီး server README အတိုင်း စတင်ပါ။

### ဥပမာ server စတင်မှု (llama.cpp, tag b5394)
1. llama.cpp ကို install ပြုလုပ်ပြီး `b5394` tag ကို checkout လုပ်ပါ
2. SmolVLM GGUF model သို့မဟုတ် သင်၏ model ဖြင့် server ကို run ပါ

   - [llama.cpp](https://github.com/ggml-org/llama.cpp)
   - Run `./llama-server -hf ggml-org/Qwen2-VL-2B-Instruct-GGUF`  
   Note: you may need to add `-ngl 99` to enable GPU (if you are using NVidia/AMD/Intel GPU)  
   Note (2): You can also try other models [here](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)

- GPU အသုံးပြုမည့်အချိန်တွင် `-ngl 99` option ကို ထည့်ရန် လိုအပ်နိုင်သည်။

## Scripts run နည်း (ဥပမာ)
- A01: `python A01_test_VLM.py`
- B01: `python B01_VLM_CAM.py`
- C01: `python C01_ONNX_VLM.py`
- D01: server စတင်ပြီး `python D01_Llamacpp_v1.py`
- D02: server စတင်ပြီး `python D02_Llamacpp_v2.py`

## ပြဿနာဖြေရှင်းချက် (Troubleshooting)
- Camera access: OpenCV သည် /dev/video0 သို့ access ရနိုင်ကြောင်း permission စစ်ပါ။
- Model download/auth: Hugging Face ကိစ္စများ ဖြစ်ပါက `huggingface-cli login` ဖြင့် authenticate သို့မဟုတ် local model path ကို အသုံးပြုပါ။
- Memory/OOM: ပုံအရွယ်အစား (image size) ကို သေးငယ်စေပါ၊ CPU inference သို့ FP16 ကို အသုံးပြုပါ။
- ONNX shape errors: C01_ONNX_VLM.py ထဲမှ decoder input names နှင့် feed keys များကို စစ်ဆေးပါ။
- Server connection: server အလုပ်လုပ်နေမရှိမစစ်ရန် `curl http://localhost:8080/health` သို့ သွားစစ်ပါ၊ BASE_URL ကို ကိုက်ညီစေပါ။

လိုအပ်ပါက ဤဖိုင်ကို repo ထဲသို့ ထည့်၍ README_myanmar.md အဖြစ်သုံးပါ။