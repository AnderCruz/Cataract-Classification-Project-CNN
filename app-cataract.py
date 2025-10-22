import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import gdown
import os

st.set_page_config(page_title="👁️ Cataract Classifier", layout="centered")
st.title("👁️ Cataract Classifier")
st.write("Classifier for **Cataract (Mature / Immature)** using a TensorFlow Lite model (single sigmoid output).")

MODEL_PATH = "cataract_model_fp16.tflite"

# ==============================
# MODEL DOWNLOAD
# ==============================
if not os.path.exists(MODEL_PATH):
    st.info("📥 Downloading cataract classification model...")
    url = "https://drive.google.com/uc?id=10_lZPWX-Ig5zOmAt3SiSdho8-Pwu7TW-"
    gdown.download(url, MODEL_PATH, quiet=False)
    st.success("✅ Download completed!")

# ==============================
# LOAD MODEL
# ==============================
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    st.success("✅ Model successfully loaded!")
except Exception as e:
    st.error(f"❌ Error loading the model: {e}")
    st.stop()

# ==============================
# IMAGE UPLOAD
# ==============================
uploaded_file = st.file_uploader("📷 Upload an eye image for analysis", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("📐 Resizing image to 416x416")

    img = image.resize((416, 416))
    img_array = np.array(img)

    # ==============================
    # ADJUST INPUT TYPE
    # ==============================
    input_dtype = input_details[0]['dtype']

    if input_dtype == np.float32:
        input_data = np.expand_dims(img_array / 255.0, axis=0).astype(np.float32)
    elif input_dtype == np.uint8:
        input_data = np.expand_dims(img_array, axis=0).astype(np.uint8)
    else:
        st.error(f"Unexpected input type: {input_dtype}")
        st.stop()

    st.success(f"✅ Image ready for inference: {input_data.shape}, dtype {input_dtype}")

    # ==============================
    # INFERENCE
    # ==============================
    try:
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Single sigmoid output value (probability)
        prob_mature = float(output_data[0][0])
        prob_immature = 1.0 - prob_mature

        # Organized probabilities
        probabilities = {
            "Immature Cataract": prob_immature * 100,
            "Mature Cataract": prob_mature * 100
        }

        # Final prediction
        predicted_class = "Mature Cataract" if prob_mature >= 0.5 else "Immature Cataract"
        confidence = probabilities[predicted_class]

        # ==============================
        # RESULTS
        # ==============================
        st.subheader("🔍 Analysis Result")
        st.write("**Estimated probabilities:**")
        for k, v in probabilities.items():
            st.write(f"- {k}: {v:.2f}%")

        if predicted_class == "Mature Cataract":
            st.success(f"👁️ Diagnosis: **{predicted_class} ({confidence:.2f}%)**")
        else:
            st.info(f"👁️ Diagnosis: **{predicted_class} ({confidence:.2f}%)**")

        st.progress(int(confidence))

        # ==============================
        # CLINICAL INTERPRETATION (SIMPLIFIED)
        # ==============================
        st.markdown("---")
        st.markdown("### 🧾 Interpretation")
        if predicted_class == "Mature Cataract":
            st.write("🔹 The model detected a high level of lens opacity, suggesting a **mature stage of cataract**.")
        else:
            st.write("🔹 The model indicates an **immature stage**, where some lens transparency is still preserved.")

        if confidence < 70:
            st.warning("⚠️ Low confidence — consider retaking the image under better lighting or focus conditions.")
        else:
            st.success("✅ Result has good statistical confidence.")

    except Exception as e:
        st.error(f"❌ Error during inference: {e}")

else:
    st.warning("⏳ Waiting for image upload...")

