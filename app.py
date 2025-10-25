# ==========================================================
# 🦁 STREAMLIT DASHBOARD: Animal Habitat Detection
# ==========================================================

import streamlit as st
import numpy as np
import torch
from keras.models import load_model
from ultralytics import YOLO
from PIL import Image

# ==========================================================
# 🌸 1. KONFIGURASI DASAR
# ==========================================================
st.set_page_config(
    page_title="🦁 AnimalAI Dashboard",
    page_icon="🐾",
    layout="wide"
)

# ==========================================================
# 🌈 CUSTOM THEME
# ==========================================================
def add_custom_style():
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #E9C46A 0%, #F4A261 50%, #E76F51 100%);
            background-attachment: fixed;
            color: #2B2118;
        }

        section[data-testid="stSidebar"] {
            background-color: #FFF8E1 !important;
            border-right: 2px solid #E76F51;
            color: #2B2118 !important;
        }

        h1, h2, h3, h4, h5, h6 {
            color: #2B2118 !important;
            font-weight: 800 !important;
        }

        p, li, span, label, div {
            color: #3B2F2F !important;
        }

        .stSuccess, .stInfo {
            background-color: rgba(255,255,255,0.85) !important;
            color: #2B2118 !important;
            border-radius: 12px;
            padding: 1em;
        }

        button[kind="primary"] {
            background-color: #E76F51 !important;
            color: white !important;
            border-radius: 10px !important;
            font-weight: bold;
        }
        button[kind="primary"]:hover {
            background-color: #F4A261 !important;
            color: black !important;
        }

        div[data-testid="stFileUploader"] > div {
            background-color: rgba(255,255,255,0.9);
            border: 2px dashed #E76F51;
            border-radius: 10px;
        }

        div[data-testid="stFileUploader"] div[role="button"] {
            background-color: #E76F51 !important;
            color: #fff !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
        }

        div[data-testid="stFileUploader"] div[role="button"]:hover {
            background-color: #F4A261 !important;
            color: #2B2118 !important;
        }

        div[data-testid="stFileUploader"] * {
            color: #3B2F2F !important;
            font-weight: 600 !important;
        }

        div[data-testid="stFileUploader"] svg {
            fill: #E76F51 !important;
            opacity: 1 !important;
        }

        img[alt="Sandrina Nur Amanda"] {
            border-radius: 18px !important;
            box-shadow: 0 4px 20px rgba(0,0,0,0.25) !important;
            transition: transform 0.3s ease;
        }

        img[alt="Sandrina Nur Amanda"]:hover {
            transform: scale(1.05);
        }
        </style>
    """, unsafe_allow_html=True)

add_custom_style()

# ==========================================================
# 🌳 2. SIDEBAR
# ==========================================================
st.sidebar.title("📂 Navigasi")
page = st.sidebar.radio(
    "Pilih Halaman:",
    ["🏠 Dashboard", "📘 Tentang Dashboard", "👩‍💻 Tentang Pembuat"]
)

if page == "🏠 Dashboard":
    st.sidebar.markdown("---")
    st.sidebar.header("⚙ Pengaturan Model")
    model_option = st.sidebar.radio(
        "Pilih jenis model yang ingin digunakan:",
        ["Klasifikasi Habitat", "Deteksi Objek"]
    )

st.sidebar.markdown("---")
st.sidebar.write("👨‍💻 *Dibuat oleh:* Sandrina Nur Amanda")
st.sidebar.markdown("[🌐 Repository GitHub](https://github.com/Sandrina-Nur-Amanda)")
st.sidebar.markdown("📅 *Tugas UTS Big Data 2025*")

# ==========================================================
# 🧩 3. HALAMAN DASHBOARD
# ==========================================================
if page == "🏠 Dashboard":
    st.title("🦁 Animal Classification & Habitat Dashboard")
    st.markdown("""
    Selamat datang di *AnimalAI Dashboard*.  
    Unggah gambar hewan buas (Lion, Tiger, Leopard, Cheetah) untuk mengetahui:
    - 🧠 *Klasifikasi Gambar & Habitat Alami*
    - 🎯 *Deteksi Objek*
    """)

    @st.cache_resource
    def load_models():
        klasifikasi_model = load_model("model/klasifikasi.h5")
        deteksi_model = YOLO("model/deteksi.pt")
        return klasifikasi_model, deteksi_model

    klasifikasi_model, deteksi_model = load_models()

    # 🖼 Upload
    st.markdown("#### 📤 Tarik atau unggah gambar hewan buas ke sini:")
    uploaded_file = st.file_uploader("", type=["jpg", "png"])

    # 📘 Deskripsi Spesies
    animal_descriptions = {
        "Tiger": """
        🐯 *Harimau (Tiger)*  
        *Habitat:* Hutan tropis, hutan hujan, hingga padang rumput di Asia (terutama India, Sumatra, dan Siberia).  
        *Ciri khas:* Tubuh besar berotot dengan belang oranye-hitam yang khas.  
        *Keunikan:* Satu-satunya kucing besar yang suka air; sering berenang untuk berburu.  
        *Identik dengan:* Kekuatan dan keberanian — simbol pelindung alam di budaya Asia.
        """,
        "Lion": """
        🦁 *Singa (Lion)*  
        *Habitat:* Padang rumput dan sabana Afrika, serta sebagian kecil Asia (India barat).  
        *Ciri khas:* Jantan memiliki surai tebal di sekitar kepala, betina tidak.  
        *Keunikan:* Hidup berkelompok dalam kawanan (pride).  
        *Identik dengan:* Kepemimpinan dan keagungan — “Raja Hutan”.
        """,
        "Leopard": """
        🐈‍⬛ *Macan Tutul (Leopard)*  
        *Habitat:* Hutan hujan tropis, padang rumput, dan pegunungan di Afrika serta Asia Selatan.  
        *Ciri khas:* Bulu keemasan dengan bintik hitam berbentuk roset.  
        *Keunikan:* Ahli memanjat pohon; sering menyimpan mangsa di dahan.  
        *Identik dengan:* Ketenangan dan kecerdikan — pemburu sabar dan taktis.
        """,
        "Cheetah": """
        🐆 *Cheetah (Cheetah)*  
        *Habitat:* Savana dan padang rumput terbuka di Afrika serta sebagian Timur Tengah.  
        *Ciri khas:* Tubuh ramping, kaki panjang, dan bercak hitam kecil di seluruh tubuh.  
        *Keunikan:* Hewan darat tercepat di dunia (hingga 112 km/jam).  
        *Identik dengan:* Kecepatan dan kelincahan — simbol fokus dan semangat mengejar tujuan.
        """
    }

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📸 Gambar yang diunggah", use_container_width=True)

        st.markdown("### 🔎 Hasil Prediksi:")
        if st.button("Jalankan Model"):
            with st.spinner("⏳ Sedang memproses gambar..."):
                if model_option == "Klasifikasi Habitat":
                    img = image.resize((224, 224))
                    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
                    pred = klasifikasi_model.predict(img_array)
                    labels = ["Cheetah", "Leopard", "Lion", "Tiger"]
                    hasil = labels[np.argmax(pred)]
                    confidence = np.max(pred) * 100

                    st.success(f"✅ Teridentifikasi sebagai *{hasil}* ({confidence:.2f}%)")
                    st.markdown(animal_descriptions[hasil])

                elif model_option == "Deteksi Objek":
                    results = deteksi_model(image)
                    st.image(results[0].plot(), caption="🎯 Hasil Deteksi Objek", use_container_width=True)
                    st.success("✅ Deteksi objek selesai!")  # tanpa habitat
    else:
        st.info("⬆ Silakan unggah gambar terlebih dahulu.")

# ==========================================================
# 🧩 4. HALAMAN TENTANG DASHBOARD (Keterangan Spesies)
# ==========================================================
elif page == "📘 Tentang Dashboard":
    st.title("📘 Tentang AnimalAI Dashboard")

    # 🌈 CSS tambahan untuk font agar tidak terlalu besar di bagian ini
    st.markdown("""
        <style>
        .dashboard-info h3, .dashboard-info h4, .dashboard-info h5, .dashboard-info p, .dashboard-info li {
            font-size: 17px !important;
            line-height: 1.5em !important;
            font-weight: 500 !important;
        }
        .dashboard-info strong {
            color: #2B2118 !important;
        }
        hr {
            border: 1px solid #E76F51 !important;
            opacity: 0.7;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="dashboard-info">
    <p><b>AnimalAI Dashboard</b> adalah sistem kecerdasan buatan yang mampu mengenali empat spesies kucing besar berikut:</p>
    <ul>
        <li>🦁 <b>Singa (Lion)</b></li>
        <li>🐯 <b>Harimau (Tiger)</b></li>
        <li>🐆 <b>Cheetah</b></li>
        <li>🐈‍⬛ <b>Macan Tutul (Leopard)</b></li>
    </ul>

    <hr>

    <h4>🐯 Harimau (Tiger)</h4>
    <p><b>Habitat:</b> Hutan tropis dan padang rumput Asia.<br>
    <b>Ciri khas:</b> Belang oranye-hitam tegas, tubuh besar.<br>
    <b>Keunikan:</b> Suka berenang, pemburu soliter.<br>
    <b>Makna simbolis:</b> Kekuatan dan keberanian.</p>

    <hr>

    <h4>🦁 Singa (Lion)</h4>
    <p><b>Habitat:</b> Sabana Afrika dan sebagian India.<br>
    <b>Ciri khas:</b> Surai tebal pada jantan.<br>
    <b>Keunikan:</b> Hidup berkelompok (pride).<br>
    <b>Makna simbolis:</b> Kepemimpinan dan kebangsawanan.</p>

    <hr>

    <h4>🐆 Cheetah (Cheetah)</h4>
    <p><b>Habitat:</b> Savana Afrika dan Timur Tengah.<br>
    <b>Ciri khas:</b> Tubuh ramping dan cepat.<br>
    <b>Keunikan:</b> Hewan tercepat di darat.<br>
    <b>Makna simbolis:</b> Fokus dan ketangkasan.</p>

    <hr>

    <h4>🐈‍⬛ Macan Tutul (Leopard)</h4>
    <p><b>Habitat:</b> Hutan tropis dan pegunungan Afrika–Asia.<br>
    <b>Ciri khas:</b> Bintik roset di bulu emas.<br>
    <b>Keunikan:</b> Ahli memanjat.<br>
    <b>Makna simbolis:</b> Ketenangan dan strategi.</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# 🧩 5. HALAMAN TENTANG PEMBUAT
# ==========================================================
elif page == "👩‍💻 Tentang Pembuat":
    st.title("👩‍💻 Tentang Pembuat")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        *Nama:* Sandrina Nur Amanda
                    
        *Jurusan:* Statistika – Universitas Syiah Kuala  

        *Deskripsi:*  
        Sandrina Nur Amanda tertarik dalam bidang *Artificial Intelligence*,  
        terutama pada penerapan *Machine Learning, **Deep Learning, dan **Computer Vision*  
        untuk menyelesaikan berbagai permasalahan nyata.  

        Dashboard ini dibuat untuk *UTS Big Data 2025*  
        guna memperkenalkan konsep deteksi dan klasifikasi berbasis AI.  

        *Kontak:*  
        - 📧 Email: sandrinaamanda65@gmail.com  
        - 🌐 GitHub: [github.com/Sandrina-Nur-Amanda](https://github.com/Sandrina-Nur-Amanda)  
        - 📸 Instagram: [@amandaaamnd_](https://www.instagram.com/amandaaamnd_/)
        """)
    with col2:
        st.markdown("<div style='display:flex;flex-direction:column;align-items:center;margin-top:50px;'>", unsafe_allow_html=True)
        st.image("assets/Amanda.jpg", use_container_width=True)
        st.markdown("""
        <style>
        [data-testid="stImage"] img {
            border: 5px solid white !important;
            border-radius: 20px !important;
            box-shadow: 0 6px 20px rgba(0,0,0,0.3) !important;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex;justify-content:center;align-items:center;flex-direction:column;margin-top:-10px;padding-left:35px;">
            <h4 style="font-size:22px;color:#2B2118;font-weight:800;margin-top:15px;margin-bottom:0;text-align:center;">
                Sandrina Nur Amanda
            </h4>
        </div>
        """, unsafe_allow_html=True)