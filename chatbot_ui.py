import streamlit as st
import requests
import json
import random
from datetime import datetime
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import tempfile
import os
import traceback

st.set_page_config(page_title="Gemma 3 Chatbot + Robust OCR", layout="centered")
st.title("🤖 Gemma 3 Chatbot + Robust PaddleOCR")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

@st.cache_resource
def load_ocr():
    # Only include very safe args so it works across versions
    return PaddleOCR(use_angle_cls=True, lang="en")

ocr_model = load_ocr()

st.sidebar.title("Controls")
if st.sidebar.button("Clear Chat"):
    st.session_state["messages"] = []
    st.rerun()
if st.sidebar.button("Reset App"):
    st.session_state.clear()
    st.rerun()

st.subheader("Upload Text / CSV file")
uploaded_file = st.file_uploader("Upload file", type=["txt", "csv"])
if uploaded_file:
    try:
        content = uploaded_file.read().decode("utf-8")
        preview = content[:500]
        st.text_area("Preview", preview, height=150)
        st.session_state["messages"].append({"role":"assistant","content":f"File uploaded. Preview:\n{preview}..."})
    except Exception as e:
        st.error(f"Error reading file: {e}")

st.subheader("OCR — Upload image for text extraction")
uploaded_image = st.file_uploader("Upload image (jpg/png)", type=["jpg","jpeg","png"])

def safe_call_ocr(model, img_arg, is_path=False):
    """
    Try multiple safe ways to call PaddleOCR:
     - model.ocr(img_array)  (preferred)
     - model.predict(img_array) (fallback)
     - model.ocr(path) (fallback)
    Return (result, method_name, exception_if_any)
    """
    # 1) try ocr(img_np)
    try:
        if is_path:
            res = model.ocr(img_arg)  # some versions accept path
        else:
            res = model.ocr(img_arg)
        return res, "ocr(img)", None
    except Exception as e1:
        # 2) try predict(img_np) if available
        try:
            if hasattr(model, "predict"):
                res = model.predict(img_arg)  # no extra kwargs
                return res, "predict(img)", None
        except Exception as e2:
            # 3) try calling ocr with path (if we passed array before)
            if not is_path:
                try:
                    # if img_arg is numpy array -> write to temp file and pass path
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                        tmp_path = tmp.name
                        Image.fromarray(img_arg).save(tmp_path)
                    try:
                        res = model.ocr(tmp_path)
                        return res, "ocr(path)", None
                    finally:
                        try:
                            os.remove(tmp_path)
                        except:
                            pass
                except Exception as e3:
                    return None, "failed_all", (e1, e2, e3)
            return None, "failed_all", (e1, e2)

def parse_paddle_result(raw):
    """
    Normalize various PaddleOCR return shapes into list of text lines.
    """
    lines = []
    if raw is None:
        return lines
    # Some versions return list of [ [box, (text,score)], ... ] blocks
    try:
        if isinstance(raw, list):
            for block in raw:
                # If block is dict-style (newer versions), try keys
                if isinstance(block, dict):
                    # modern style: dict with rec_texts
                    if "rec_texts" in block:
                        for t in block["rec_texts"]:
                            if t and str(t).strip():
                                lines.append(str(t).strip())
                    else:
                        # fallback: iterate values
                        continue
                elif isinstance(block, list):
                    for item in block:
                        try:
                            # item usually [box, (text, score)]
                            text = item[1][0] if isinstance(item[1], (list,tuple)) else item[1]
                            if text and str(text).strip():
                                lines.append(str(text).strip())
                        except Exception:
                            # fallback if item is different shape
                            try:
                                # maybe item is (text, score)
                                if isinstance(item, (list, tuple)) and len(item) >= 2:
                                    lines.append(str(item[0]))
                            except:
                                pass
                else:
                    # unknown block type — try to stringify
                    try:
                        txt = str(block)
                        if txt.strip():
                            lines.append(txt.strip())
                    except:
                        pass
        return lines
    except Exception:
        # last resort: return stringified raw
        return [str(raw)]

if uploaded_image:
    pil = Image.open(uploaded_image).convert("RGB")
    st.image(pil, caption="Uploaded image", use_container_width=True)

    if st.button("Extract Text"):
        with st.spinner("Running OCR..."):
            img_np = np.array(pil)

            result, method, err = safe_call_ocr(ocr_model, img_np, is_path=False)

            if result is None:
                st.error("OCR failed to run with available call patterns.")
                if err:
                    st.write("Exceptions (first, second, third):")
                    for exc in (err if isinstance(err, (list,tuple)) else [err]):
                        st.write(repr(exc))
                    st.write("Full traceback for debugging:")
                    st.text(traceback.format_exc())
            else:
                texts = parse_paddle_result(result)
                if texts:
                    out = "\n".join(texts)
                    st.text_area("OCR Output", out, height=250)
                    st.session_state["messages"].append({"role":"assistant","content":f"OCR Result:\n{out}"})
                else:
                    st.warning("No readable text detected. Raw OCR result shown below for debugging:")
                    st.write(result)
                    st.session_state["messages"].append({"role":"assistant","content":"OCR Result: [no readable text]"})
 
responses = {
    "hello": "Hello! How can I assist you?",
    "hi": "Hi there! What can I do for you?",
    "how are you": "I'm just a program, but I'm doing great!",
    "your name": "I'm Gemma 3 Chatbot, created using Streamlit!",
    "time": f"The current time is {datetime.now().strftime('%H:%M:%S')}.",
    "date": f"Today's date is {datetime.now().strftime('%Y-%m-%d')}.",
    "bye": "Goodbye! Have a nice day.",
    "thanks": "You're welcome!"
}

for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f"👤 *You:* {msg['content']}")
    else:
        st.markdown(f"🤖 *Bot:* {msg['content']}")

user_input = st.text_input("You: ", "")

def ask_ollama(prompt):
    try:
        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={"model": "gemma3:latest", "prompt": prompt, "stream": False},
            timeout=120
        )
        if response.status_code == 200:
            return response.json().get("response", "")
        return f"Error: {response.status_code} {response.text}"
    except Exception as e:
        return f"Ollama error: {e}"

if st.button("Send"):
    if user_input.strip():
        st.session_state["messages"].append({"role":"user","content":user_input})
        text = user_input.lower().strip()
        reply = None
        for k,v in responses.items():
            if k in text:
                reply = v
                break
        if reply is None and any(ch.isdigit() for ch in text):
            try:
                reply = str(eval(user_input))
            except:
                reply = "Couldn't evaluate expression."
        if reply is None:
            reply = ask_ollama(user_input)
        st.session_state["messages"].append({"role":"assistant","content":reply})
        st.rerun()