# vlm_SmolVLM — မြန်မာဘာသာ README

ဒီ Repository မှာ SmolVLM သုံးပြီး Visual–Language Model (VLM) ကို စမ်းသပ်နိုင်ဖို့ အမည်တပ်ထားတဲ့ example scripts ၂ ခု ပါဝင်ပါတယ်။

ဖိုင်များနှင့် ရည်ရွယ်ချက်

- `A01_test_VLM.py` — တစ်ပုံကို ဖတ်ပြီး VLM နဲ့ inference လုပ်တဲ့ စမ်းသပ်ကိရိယာ။
  - ပုံရွေးရန် `img_path` မှာ absolute path ထည့်ထားပါသည်။ `photo1.jpg` ကို repository ထဲမှာ သို့မဟုတ် `img_path = "./photo1.jpg"` အဖြစ် ပြင်လိုက်ပါ။
  - ပုံကို resize လုပ်ပြီး GPU memory သက်သာစေသည်။

- `B01_VLM_CAM.py` — Webcam သုံး၍ အချိန်နှင့်တပြေးညီ frame များကို ခံယူပြီး user က terminal မှာ ထည့်သွင်းသော မေးခွန်းအရ ရုပ်ပုံအကြောင်း ပြန်စာရင်းထုတ်ပေးသည်။
  - Webcam ကို ဖွင့်ထားပြီး `q` ကီးနှိပ်၍ camera window ကို ပိတ်နိုင်သည်။ Terminal မှာ `quit`/`q`/`exit` ထည့်၍ script ကို နောက်ထပ် ရပ်ပေးနိုင်သည်။

အမြန်စမ်းသပ်ရန်

1. Python virtual environment ဖန်တီးပြီး အသုံးပြုရန် (အကြံပြု):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. လိုအပ်သော libraries 설치:

   ```bash
   pip install -r requirements.txt
   ```

3. (လိုလျှင်) PyTorch ကို သင့် CUDA အတွက် ကိုက်ညီအောင် တပ်ဆင်ပါ။

Examples ကို ပြေးရန်

- တစ်ပုံ စမ်းသပ်ရန် (img_path ကို သေချာစွာ ပြင်ထားပါ):

  ```bash
  python A01_test_VLM.py
  ```

- Webcam demo ကို chạyရန် (OpenCV နှင့် webcam လိုအပ်):

  ```bash
  python B01_VLM_CAM.py
  ```

Headless / server-only သုံးချင်လျှင်

- `A01_test_VLM.py` ကို GUI မဖွင့်ဘဲ လုပ်ချင်ရင် stdout ကို ဖိုင်ထဲသို့ redirect လုပ်ပါ။

  ```bash
  python A01_test_VLM.py > results.txt
  ```

- `B01_VLM_CAM.py` ကို headless server ပေါ်တွင် chạy ရင် `cv2.imshow`/`cv2.waitKey` ကို ဖျက်ပစ်ရန် သို့မဟုတ် `xvfb-run` နဲ့ virtual X server သုံးနိုင်သည်။

  ```bash
  xvfb-run -s "-screen 0 1400x900x24" python B01_VLM_CAM.py
  ```

Device ရွေးချယ်မှုနှင့် memory အကြံပြုချက်

- ဖြစ်နိုင်သလောက် CUDA ကို အသုံးပြုရန် `device_map="auto"` သတ်မှတ်ထားသည်။ CPU ကို မဖြစ်မနေ သုံးချင်လျှင်:

  ```bash
  export CUDA_VISIBLE_DEVICES=""
  ```

- Memory လျော့ချရန် အကြံ
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` ထည့်ထားသည်။
  - မူရင်း script များတွင် `torch_dtype=torch.float16` အသုံးပြုထားသည်။
  - ပုံကို resize လုပ်ပြီး `max_new_tokens` ကို လျော့ချပါ။

ပြဿနာဖြေရှင်းနည်း

- ပုံမတွေ့ပါက: `img_path` ကို စစ်ဆေး၍၊ `photo1.jpg` ကို script နှင့် တူ directory ထဲတွင် ထားပါ။
- Webcam မသိရှိပါက: `cv2.VideoCapture(0)` အစား `1` သို့မဟုတ် အခြား device index သတ်မှတ်ပြီး စမ်းပါ။
- CUDA / PyTorch ပြဿနာများရှိပါက https://pytorch.org သို့ သွားပြီး သင့် CUDA version နှင့် ကိုက်ညီသော PyTorch build ကို ထည့်သွင်းပါ။

License

MIT (လိုအပ်လျှင် ပြောင်းလဲနိုင်သည်)
