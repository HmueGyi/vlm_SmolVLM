import cv2
import time
from PIL import Image
import numpy as np
import onnxruntime as ort
import threading
from transformers import AutoTokenizer

# ----------------------------
# Paths to FP32 ONNX models
# ----------------------------
model_path = "./model/onnx"
vision_model_file = f"{model_path}/vision_encoder.onnx"
embed_model_file = f"{model_path}/embed_tokens.onnx"
decoder_model_file = f"{model_path}/decoder_model_merged.onnx"
providers = ["CPUExecutionProvider"]
print("Running on CPU with FP32 models")

# ----------------------------
# Load ONNX sessions
# ----------------------------
vision_sess = ort.InferenceSession(vision_model_file, providers=providers)
embed_sess = ort.InferenceSession(embed_model_file, providers=providers)
decoder_sess = ort.InferenceSession(decoder_model_file, providers=providers)

# Debug: print decoder session input names/shapes/types to identify expected vision/embedding input names
print("Decoder ONNX inputs:")
for inp in decoder_sess.get_inputs():
    try:
        print("  ", inp.name, inp.shape, inp.type)
    except Exception:
        print("  ", inp.name)

# Load tokenizer for decoder output
tokenizer = AutoTokenizer.from_pretrained("./model")

# ----------------------------
# Webcam setup
# ----------------------------
cap = cv2.VideoCapture(0)
frame_lock = threading.Lock()
current_frame = None

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

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        with frame_lock:
            current_frame = frame.copy()

        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

threading.Thread(target=capture_frames, daemon=True).start()

print("Type a question and press Enter (type 'q' to quit)")

# ----------------------------
# Helper functions
# ----------------------------
def preprocess_image(image: Image, size=512):
    # Ensure 3 channels and correct size
    image = image.convert("RGB")
    image = image.resize((size, size))
    arr = np.array(image).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    arr = np.expand_dims(arr, axis=0)  # batch dim -> (1, C, H, W)
    arr = np.expand_dims(arr, axis=1)  # add extra dimension for ONNX 5D -> (1,1,C,H,W)
    return arr

def run_vlm(image: Image, question: str):
    vision_input = preprocess_image(image)
    # Build boolean attention mask with shape (batch, channels, H, W) matching ONNX vision encoder expected rank 4
    pixel_attention_mask = np.ones((vision_input.shape[0], vision_input.shape[1], vision_input.shape[-2], vision_input.shape[-1]), dtype=np.bool_)

    vision_feats = vision_sess.run(
        None,
        {
            "pixel_values": vision_input,
            "pixel_attention_mask": pixel_attention_mask
        }
    )[0]

    input_ids = tokenizer(question, return_tensors="np")["input_ids"]
    token_embeds = embed_sess.run(None, {"input_ids": input_ids})[0]

    # Build decoder input feed matching decoder_sess inputs. The decoder expects many inputs (inputs_embeds, attention_mask, position_ids, past_key_values.*)
    batch = int(input_ids.shape[0])
    seq_len = int(input_ids.shape[1])

    feed = {}

    # Find actual input names in the decoder for vision features / embeddings / input_ids
    vision_input_name = None
    embed_input_name = None
    input_ids_name = None

    # Collect decoder input names lowercased for matching
    decoder_input_names = [inp.name for inp in decoder_sess.get_inputs()]
    decoder_input_names_l = [n.lower() for n in decoder_input_names]

    # Common candidate names for encoder/vision outputs in different models
    vision_candidates = [
        'vision_feats', 'vision_embeds', 'vision_embeddings', 'visual_embeds', 'visual_features',
        'image_embeds', 'image_embeddings', 'image_features', 'pixel_values', 'encoder_hidden_states',
        'encoder_last_hidden_state', 'encoder_outputs', 'encoder_output', 'img_embeddings', 'image', 'visual'
    ]

    for cand in vision_candidates:
        for orig_name, lname in zip(decoder_input_names, decoder_input_names_l):
            if cand in lname:
                vision_input_name = orig_name
                break
        if vision_input_name:
            break

    # Embedding input (inputs_embeds) detection
    for inp in decoder_sess.get_inputs():
        lname = inp.name.lower()
        if 'inputs_embeds' == inp.name or 'inputs_embeds' in lname or 'input_embeds' in lname:
            embed_input_name = inp.name
            break

    # input_ids detection
    for inp in decoder_sess.get_inputs():
        lname = inp.name.lower()
        if inp.name == 'input_ids' or 'input_ids' in lname:
            input_ids_name = inp.name
            break

    # Map computed tensors to the decoder's expected input names
    # Build inputs_embeds by prepending visual embeddings to token embeddings if decoder does not take vision input directly
    embed_name_to_use = embed_input_name if embed_input_name else ( 'inputs_embeds' if 'inputs_embeds' in decoder_input_names_l else None )

    # Prepare vision_feats to match embedding dim
    if embed_name_to_use is not None:
        embed_dim = int(token_embeds.shape[-1])
        vf = vision_feats
        # normalize vision_feats shape to (batch, n_vis, embed_dim)
        if isinstance(vf, np.ndarray):
            if vf.ndim == 3 and vf.shape[-1] == embed_dim:
                pass
            elif vf.ndim >= 3 and vf.shape[-1] == embed_dim:
                vf = vf.reshape(vf.shape[0], -1, embed_dim)
            elif vf.ndim == 2 and vf.shape[-1] == embed_dim:
                vf = vf.reshape(vf.shape[0], 1, embed_dim)
            else:
                # try to move/embed channels to last dim if possible
                try:
                    vf = vf.reshape(vf.shape[0], -1, embed_dim)
                except Exception:
                    # fallback to zeros
                    vf = np.zeros((batch, 0, embed_dim), dtype=np.float32)
        else:
            vf = np.zeros((batch, 0, embed_dim), dtype=np.float32)

        # Concatenate visual tokens (may be empty) and token embeddings
        try:
            combined_embeds = np.concatenate([vf, token_embeds], axis=1)
        except Exception:
            # fallback: use token_embeds only
            combined_embeds = token_embeds

        # Assign to the decoder's expected embedding input name
        if embed_name_to_use in [inp.name for inp in decoder_sess.get_inputs()]:
            feed[embed_name_to_use] = combined_embeds
        else:
            feed['inputs_embeds'] = combined_embeds

        # Attention mask covers visual + text tokens; decoder expects attention_mask with total_sequence_length
        total_len = int(combined_embeds.shape[1])
        feed['attention_mask'] = np.ones((batch, total_len), dtype=np.int64)

        # Use total_len so position_ids align with inputs_embeds length (visual + text)
        feed['position_ids'] = np.arange(total_len, dtype=np.int64).reshape(1, total_len)
        if batch > 1:
            feed['position_ids'] = np.repeat(feed['position_ids'], batch, axis=0)

    else:
        # No embedding input expected by decoder; do not add unknown keys
        print("Decoder does not accept inputs_embeds; available inputs:", decoder_input_names)

    # Only add input_ids if decoder explicitly expects it
    # Do not pass input_ids; decoder expects inputs_embeds. If needed for other models, enable below:
    # if input_ids_name:
    #     feed[input_ids_name] = input_ids

    # Helper to create dummy arrays for other inputs using the input metadata shape and type
    def make_dummy_from_meta(node_arg):
        # node_arg.type like 'tensor(float)'
        dtype = node_arg.type
        # node_arg.shape may contain None or strings for dynamic dims
        raw_shape = None
        try:
            raw_shape = node_arg.shape
        except Exception:
            raw_shape = None
        shape = []
        if raw_shape:
            for d in raw_shape:
                # if ONNX provides an int dimension, use it
                if isinstance(d, int):
                    shape.append(d)
                else:
                    # For symbolic dims like 'batch_size', 'sequence_length', 'past_sequence_length'
                    ds = str(d).lower() if d is not None else ''
                    if 'batch' in ds:
                        shape.append(batch)
                    elif 'past' in ds:
                        # no past keys initially: use 1 as safe default
                        shape.append(1)
                    elif 'sequence' in ds or 'total' in ds:
                        # align sequence-like dims to total_len when available, otherwise seq_len
                        try:
                            shape.append(total_len)
                        except Exception:
                            shape.append(seq_len)
                    else:
                        # unknown symbolic dim -> 1
                        shape.append(1)
        else:
            shape = [1]

        # Replace -1 or 0 with 1
        shape = [1 if (not isinstance(d, int) or d <= 0) else d for d in shape]

        # If tensor likely depends on sequence length, try to set second dim to seq_len when shape length >=2
        if len(shape) >= 2:
            # try to heuristically set seq dimension
            # If first dim is batch and second is seq, and second is 1, set to seq_len
            if shape[0] == 1 and shape[1] == 1:
                shape[1] = seq_len
            elif shape[0] == batch and shape[1] == 1:
                shape[1] = seq_len

        if 'float' in dtype:
            return np.zeros(tuple(shape), dtype=np.float32)
        if 'int64' in dtype or 'int' in dtype:
            return np.zeros(tuple(shape), dtype=np.int64)
        if 'bool' in dtype:
            return np.zeros(tuple(shape), dtype=np.bool_)
        # fallback
        return np.zeros(tuple(shape), dtype=np.float32)

    # Inspect decoder inputs and add any missing ones with sensible defaults
    for inp in decoder_sess.get_inputs():
        name = inp.name
        if name in feed:
            continue
        if name == 'attention_mask':
            # use total_len if available (when inputs_embeds were provided), otherwise seq_len
            llen = total_len if 'total_len' in locals() else seq_len
            feed[name] = np.ones((batch, llen), dtype=np.int64)
            continue
        if name == 'position_ids':
            llen = total_len if 'total_len' in locals() else seq_len
            feed[name] = np.arange(llen, dtype=np.int64).reshape(1, llen)
            if batch > 1:
                feed[name] = np.repeat(feed[name], batch, axis=0)
            continue
        # For past_key_values.* and other tensors, create zeros using metadata
        feed[name] = make_dummy_from_meta(inp)

    # Run decoder
    try:
        decoder_out = decoder_sess.run(None, feed)[0]
    except Exception as e:
        print("Decoder run failed:", e)
        print("Provided feed keys:", list(feed.keys()))
        print("Decoder expected inputs:", [inp.name for inp in decoder_sess.get_inputs()])
        raise

    # If decoder_out are token ids, decode; otherwise handle accordingly
    # Normalize decoder output to token ids (batch, seq_len)
    token_ids = None
    try:
        out = decoder_out
        # If we got a numpy array
        if isinstance(out, np.ndarray):
            if out.ndim == 3:
                # logits: take argmax over vocab dim
                token_ids = out.argmax(axis=-1).astype(np.int64)
            elif out.ndim == 2:
                # already (batch, seq_len)
                token_ids = out.astype(np.int64)
            elif out.ndim == 1:
                token_ids = out.reshape(1, -1).astype(np.int64)
            else:
                # unexpected rank -> flatten per-batch
                token_ids = out.reshape(out.shape[0], -1).astype(np.int64)
        elif isinstance(out, (list, tuple)):
            # try to convert first element to array
            try:
                a = np.array(out)
                if a.ndim == 1:
                    token_ids = a.reshape(1, -1).astype(np.int64)
                else:
                    token_ids = a.astype(np.int64)
            except Exception:
                # last resort: take first element
                token_ids = np.array(out[0]).astype(np.int64)
        else:
            token_ids = np.array([int(out)])

        # Ensure token_ids is 2D (batch, seq_len)
        if token_ids.ndim == 1:
            token_ids = token_ids.reshape(1, -1)

        # Decode first batch element
        text_out = tokenizer.decode(token_ids[0].tolist(), skip_special_tokens=True)
    except Exception as e:
        print("Decoding failed, decoder_out type/shape:", type(decoder_out), getattr(decoder_out, 'shape', None))
        raise
    return text_out

# ----------------------------
# Main loop
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

    start_time = time.time()
    answer = run_vlm(image, user_question)
    response_time = time.time() - start_time

    print(answer)
    print(f"Response time: {response_time:.2f}s\n")

cap.release()
cv2.destroyAllWindows()
