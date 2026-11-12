import json
import os

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

# ------------------------------------------------------------------
# Konfigurasi
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_model_path():
    """Cari file .keras di folder app.py (nama bebas)."""
    preferred = os.path.join(BASE_DIR, "oral_diseases_mobilenetv2_final.keras")
    if os.path.exists(preferred):
        return preferred
    for nama in sorted(os.listdir(BASE_DIR)):
        if nama.lower().endswith(".keras"):
            return os.path.join(BASE_DIR, nama)
    return preferred


MODEL_PATH = find_model_path()
CLASS_PATH = os.path.join(BASE_DIR, "class_names.json")
IMG_SIZE = (224, 224)

# Urutan ini sama dengan class_names di notebook (alfabetis)
DEFAULT_CLASSES = [
    "Calculus",
    "Caries",
    "Gingivitis",
    "Hypodontia",
    "Mouth_Ulcer",
    "Tooth_Discoloration",
]

INFO = {
    "Calculus": "Karang gigi: plak yang mengeras dan menempel pada permukaan gigi.",
    "Caries": "Karies: gigi berlubang akibat kerusakan email oleh bakteri.",
    "Gingivitis": "Gingivitis: peradangan gusi, biasanya gusi merah, bengkak, dan mudah berdarah.",
    "Hypodontia": "Hypodontia: kondisi kurangnya jumlah gigi yang tumbuh dibanding normal.",
    "Mouth_Ulcer": "Sariawan: luka terbuka kecil di dalam mulut.",
    "Tooth_Discoloration": "Perubahan warna gigi: gigi tampak kuning, cokelat, atau keabu-abuan.",
}

st.set_page_config(page_title="Deteksi Penyakit Gigi & Mulut", page_icon="🦷", layout="centered")


# ------------------------------------------------------------------
# Load model (di-cache supaya hanya dimuat sekali)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model...")
def load_model_and_classes():
    from tensorflow import keras

    model = keras.models.load_model(MODEL_PATH, compile=False)

    if os.path.exists(CLASS_PATH):
        with open(CLASS_PATH, "r") as f:
            class_names = json.load(f)
    else:
        class_names = DEFAULT_CLASSES
    return model, class_names


def preprocess(image: Image.Image) -> np.ndarray:
    """Model sudah punya layer preprocess_input di dalamnya,
    jadi cukup kirim piksel 0-255 ukuran 224x224 RGB."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize(IMG_SIZE, Image.BILINEAR)
    arr = np.asarray(image, dtype="float32")
    return np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)


# ------------------------------------------------------------------
# Tampilan
# ------------------------------------------------------------------
st.title("🦷 Deteksi Penyakit Gigi & Mulut")
st.write(
    "Unggah gambar gigi atau rongga mulut, lalu model **MobileNetV2** akan "
    "memprediksi kategori kondisinya."
)

st.warning(
    "Aplikasi ini hanya untuk tujuan belajar/penelitian dan **bukan pengganti "
    "diagnosis dokter gigi**. Akurasi model pada data test sekitar 80%."
)

# ===== INPUT GAMBAR (selalu tampil) =====
file = st.file_uploader(
    "📤 Masukkan gambar gigi atau mulut (JPG, JPEG, PNG, BMP, WEBP)",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

# ===== Muat model =====
model, class_names = None, DEFAULT_CLASSES
if os.path.exists(MODEL_PATH):
    model, class_names = load_model_and_classes()
else:
    st.error(
        "File model `.keras` belum ditemukan, jadi prediksi belum bisa dijalankan. "
        "Letakkan `oral_diseases_mobilenetv2_final.keras` di folder yang sama dengan `app.py`."
    )
    with st.expander("Lihat folder yang dibaca aplikasi"):
        st.code(BASE_DIR)
        st.code("\n".join(sorted(os.listdir(BASE_DIR))))

# Opsional: tombol gambar acak dari folder "sample_images" (taruh beberapa gambar di sana)
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_images")
sample_files = []
if os.path.isdir(SAMPLE_DIR):
    sample_files = [
        os.path.join(SAMPLE_DIR, f)
        for f in os.listdir(SAMPLE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ]

if sample_files and st.button("🎲 Pilih gambar acak dari sample_images"):
    import random

    st.session_state["random_image"] = random.choice(sample_files)

if file is not None:
    st.session_state.pop("random_image", None)  # upload manual menggantikan gambar acak
    image = Image.open(file)
    caption = "Gambar input"
elif "random_image" in st.session_state:
    image = Image.open(st.session_state["random_image"])
    caption = f"Gambar acak: {os.path.basename(st.session_state['random_image'])}"
else:
    image = None

if image is not None:
    st.image(image, caption=caption, use_container_width=True)

    if st.button("Prediksi", type="primary", disabled=model is None):
        with st.spinner("Menganalisis gambar..."):
            x = preprocess(image)
            probs = model.predict(x, verbose=0)[0]

        top = int(np.argmax(probs))
        label = class_names[top]

        st.subheader("Hasil prediksi")
        st.success(f"**{label.replace('_', ' ')}** — keyakinan {probs[top] * 100:.2f}%")
        st.info(INFO.get(label, ""))

        if probs[top] < 0.5:
            st.caption(
                "⚠️ Keyakinan model rendah. Gambar mungkin kurang jelas atau bukan "
                "foto gigi/mulut. Model hanya mengenal 6 kelas dan akan selalu "
                "memilih salah satunya, walaupun gambarnya tidak relevan."
            )

        df = pd.DataFrame(
            {"Kelas": [c.replace("_", " ") for c in class_names], "Probabilitas (%)": probs * 100}
        ).sort_values("Probabilitas (%)", ascending=False)

        st.bar_chart(df.set_index("Kelas"))
        st.dataframe(df.style.format({"Probabilitas (%)": "{:.2f}"}), hide_index=True)

with st.expander("Tentang model"):
    st.markdown(
        """
- Arsitektur: MobileNetV2 (transfer learning dari ImageNet + fine-tuning)
- Input: gambar RGB 224×224
- Kelas: Calculus, Caries, Gingivitis, Hypodontia, Mouth Ulcer, Tooth Discoloration
- Akurasi test: **80,45%**
- Catatan: recall kelas *Caries* masih rendah (~50%), jadi hasilnya bisa meleset.
"""
    )
