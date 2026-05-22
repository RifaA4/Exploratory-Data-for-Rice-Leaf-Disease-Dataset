import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os
import glob
import zipfile

# ─────────────────────────────────────────────
# DOWNLOAD DATASET DARI GOOGLE DRIVE
# ─────────────────────────────────────────────
@st.cache_resource
def download_dataset():
    import gdown
    ZIP_ID   = "1RsW4hC-pI67eMPFnVKX6wFTbtQn7TjAE"
    ZIP_PATH = "dataset.zip"
    BASE_PATH = "data/split"

    if os.path.exists(BASE_PATH):
        return BASE_PATH

    # Download zip dari Google Drive
    gdown.download(id=ZIP_ID, output=ZIP_PATH, quiet=False)

    # Extract zip
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(".")

    # Hapus zip setelah extract
    os.remove(ZIP_PATH)

    return BASE_PATH

# Jalankan download saat app pertama kali dibuka
with st.spinner("⏳ Memuat dataset..."):
    BASE_PATH = download_dataset()

# kelas_folder = nama folder asli di disk
# kelas_label  = nama tampilan di dashboard
kelas_folder = ["Blast", "BrownSpot", "Healthy", "Tungro"]
kelas_label  = ["Blast", "Brown Spot", "Healthy", "Tungro"]
folder_to_label = dict(zip(kelas_folder, kelas_label))
label_to_folder = dict(zip(kelas_label,  kelas_folder))

# ─────────────────────────────────────────────
# HITUNG JUMLAH DATA ASLI DARI FOLDER
# ─────────────────────────────────────────────
@st.cache_data
def hitung_data(base_path):
    data_train = {}
    data_test  = {}
    for folder, label in folder_to_label.items():
        path_train = os.path.join(base_path, "train", folder)
        path_test  = os.path.join(base_path, "test",  folder)
        data_train[label] = len(glob.glob(os.path.join(path_train, "*.jpg"))) if os.path.exists(path_train) else 0
        data_test[label]  = len(glob.glob(os.path.join(path_test,  "*.jpg"))) if os.path.exists(path_test)  else 0
    return data_train, data_test

data_train, data_test = hitung_data(BASE_PATH)

# ─────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RiceCare AI - EDA Dashboard",
    page_icon="🌾",
    layout="wide"
)

st.markdown("""
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: bold;
            color: #2e7d32;
            text-align: center;
        }
        .sub-title {
            font-size: 1rem;
            color: #555;
            text-align: center;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">🌾 RiceCare AI — EDA Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Exploratory Data Analysis | Coding Camp 2026 × DBS Foundation | Tim CC26-PSU169</div>', unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────────
# SIDEBAR NAVIGASI
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/7d/Rice_p1160005.jpg/320px-Rice_p1160005.jpg", use_column_width=True)
    st.markdown("## 🌾 RiceCare AI")
    st.markdown("**EDA Dashboard**")
    st.divider()
    halaman = st.radio(
        "Pilih Halaman:",
        ["📋 Overview & Data Dictionary", "📊 Visualisasi EDA", "🖼️ Galeri Sampel Gambar", "🔬 Analisis Piksel & Warna"]
    )
    st.divider()
    st.caption("Tim Data Scientist: Rifa Agnia & Nisa Nuraini")

# ─────────────────────────────────────────────
# HALAMAN 1: OVERVIEW & DATA DICTIONARY
# ─────────────────────────────────────────────
if halaman == "📋 Overview & Data Dictionary":
    st.subheader("📋 Overview Proyek")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Kelas", "4")
    col2.metric("Total Data Train", sum(data_train.values()))
    col3.metric("Total Data Test",  sum(data_test.values()))
    col4.metric("Target Akurasi", "≥ 85%")

    st.divider()
    st.subheader("📖 Data Dictionary")
    dict_data = {
        "Kolom / Fitur": ["image", "label", "split", "width", "height", "color_mode"],
        "Tipe Data": ["Image (JPG/PNG)", "String (Kategori)", "String", "Integer", "Integer", "String"],
        "Deskripsi": [
            "File foto daun padi yang digunakan sebagai input model",
            "Kelas penyakit: Blast, Brown Spot, Tungro, atau Healthy",
            "Pembagian data: Train atau Test",
            "Lebar gambar dalam piksel (standar: 224px setelah resize)",
            "Tinggi gambar dalam piksel (standar: 224px setelah resize)",
            "Mode warna gambar (RGB)"
        ],
        "Contoh Nilai": ["Blast_0001.jpg", "Blast", "Train", "224", "224", "RGB"]
    }
    st.dataframe(pd.DataFrame(dict_data), use_container_width=True)

    st.divider()
    st.subheader("🦠 Penjelasan Kelas Penyakit")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**🟤 Brown Spot**")
        st.write("Bercak coklat pada daun akibat jamur Helminthosporium oryzae. Menyebabkan biji padi tidak berisi penuh.")
    with col2:
        st.markdown("**🟡 Blast**")
        st.write("Penyakit jamur Magnaporthe oryzae yang menyerang leher malai dan daun. Salah satu penyakit paling merusak.")
    with col3:
        st.markdown("**🟠 Tungro**")
        st.write("Disebabkan oleh virus yang ditularkan wereng hijau. Daun menguning dan pertumbuhan terhambat.")
    with col4:
        st.markdown("**🟢 Healthy**")
        st.write("Daun padi dalam kondisi sehat, tidak menunjukkan gejala penyakit apapun.")

# ─────────────────────────────────────────────
# HALAMAN 2: VISUALISASI EDA
# ─────────────────────────────────────────────
elif halaman == "📊 Visualisasi EDA":
    st.subheader("📊 Visualisasi Distribusi Data")

    pilihan_split = st.radio("Pilih Dataset:", ["Data Train", "Data Test"], horizontal=True)

    data_dipilih = data_train if pilihan_split == "Data Train" else data_test
    df = pd.DataFrame(list(data_dipilih.items()), columns=["Kelas", "Jumlah"])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Pie Chart Distribusi {pilihan_split}**")
        fig_pie = px.pie(
            df, values="Jumlah", names="Kelas",
            color_discrete_sequence=["#ef5350", "#ff7043", "#ffa726", "#66bb6a"],
            hole=0.3
        )
        fig_pie.update_traces(textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.markdown(f"**Bar Chart Distribusi {pilihan_split}**")
        fig_bar = px.bar(
            df, x="Kelas", y="Jumlah",
            color="Kelas",
            color_discrete_sequence=["#ef5350", "#ff7043", "#ffa726", "#66bb6a"],
            text="Jumlah"
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(showlegend=False, yaxis_title="Jumlah Gambar")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()
    st.markdown("**📊 Perbandingan Jumlah Data Train vs Test per Kelas**")
    df_compare = pd.DataFrame({
        "Kelas": kelas_label,
        "Train": list(data_train.values()),
        "Test":  list(data_test.values())
    })
    fig_group = px.bar(
        df_compare.melt(id_vars="Kelas", var_name="Split", value_name="Jumlah"),
        x="Kelas", y="Jumlah", color="Split", barmode="group",
        color_discrete_sequence=["#42a5f5", "#ef5350"],
        text="Jumlah"
    )
    fig_group.update_traces(textposition="outside")
    st.plotly_chart(fig_group, use_container_width=True)

    st.divider()
    st.markdown("**⚖️ Analisis Keseimbangan Data (Imbalance Check)**")
    vals = [v for v in data_train.values() if v > 0]
    if vals:
        rasio = round(max(vals) / min(vals), 2)
        if rasio > 1.5:
            st.warning(f"⚠️ Rasio kelas terbesar vs terkecil: **{rasio}x** — Data kemungkinan **tidak seimbang**. Disarankan melakukan oversampling pada kelas minoritas.")
        else:
            st.success(f"✅ Rasio kelas terbesar vs terkecil: **{rasio}x** — Data relatif **seimbang**.")

# ─────────────────────────────────────────────
# HALAMAN 3: GALERI SAMPEL GAMBAR
# ─────────────────────────────────────────────
elif halaman == "🖼️ Galeri Sampel Gambar":
    st.subheader("🖼️ Galeri Sampel Gambar Per Kelas")

    kelas_dipilih  = st.selectbox("Pilih Kelas Penyakit:", kelas_label)
    jumlah_gambar  = st.slider("Jumlah sampel yang ingin ditampilkan:", 1, 10, 4)

    # Konversi label tampilan -> nama folder asli (BrownSpot, bukan Brown Spot)
    folder_dipilih = label_to_folder[kelas_dipilih]
    folder_path    = os.path.join(BASE_PATH, "train", folder_dipilih)

    if os.path.exists(folder_path):
        semua_gambar = glob.glob(os.path.join(folder_path, "*.jpg")) + \
                       glob.glob(os.path.join(folder_path, "*.JPG")) + \
                       glob.glob(os.path.join(folder_path, "*.png"))
        if semua_gambar:
            import random
            sampel = random.sample(semua_gambar, min(jumlah_gambar, len(semua_gambar)))
            cols = st.columns(min(jumlah_gambar, 4))
            for i, path_gambar in enumerate(sampel):
                with cols[i % 4]:
                    img = Image.open(path_gambar)
                    st.image(img, caption=os.path.basename(path_gambar), use_column_width=True)
        else:
            st.warning("Tidak ada gambar ditemukan di folder ini.")
    else:
        st.error(f"Folder tidak ditemukan: `{folder_path}`")

# ─────────────────────────────────────────────
# HALAMAN 4: ANALISIS PIKSEL & WARNA
# ─────────────────────────────────────────────
elif halaman == "🔬 Analisis Piksel & Warna":
    st.subheader("🔬 Analisis Piksel & Warna (Advanced EDA)")

    tab1, tab2 = st.tabs(["📊 Mean Image per Kelas", "🔍 Interactive Pixel Inspector"])

    # ── TAB 1: MEAN IMAGE ──────────────────────
    with tab1:
        st.markdown("**Rata-rata Piksel (Mean Image) per Kelas Penyakit**")
        st.write("Visualisasi ini menunjukkan karakteristik warna dan tekstur rata-rata dari setiap kelas penyakit.")

        @st.cache_data
        def hitung_mean_image(base_path):
            hasil = {}
            hasil_rgb = {}
            for folder, label in folder_to_label.items():
                folder_path  = os.path.join(base_path, "train", folder)
                semua_gambar = glob.glob(os.path.join(folder_path, "*.jpg"))[:50]
                if semua_gambar:
                    arrays = []
                    for p in semua_gambar:
                        img = Image.open(p).resize((224, 224)).convert("RGB")
                        arrays.append(np.array(img))
                    mean_arr = np.mean(arrays, axis=0).astype(np.uint8)
                    hasil[label]     = mean_arr
                    hasil_rgb[label] = {
                        "R": float(mean_arr[:,:,0].mean()),
                        "G": float(mean_arr[:,:,1].mean()),
                        "B": float(mean_arr[:,:,2].mean())
                    }
                else:
                    placeholder = np.random.randint(80, 180, (224, 224, 3), dtype=np.uint8)
                    hasil[label]     = placeholder
                    hasil_rgb[label] = {"R": 128.0, "G": 128.0, "B": 128.0}
            return hasil, hasil_rgb

        with st.spinner("Menghitung mean image dari dataset..."):
            mean_images, rgb_per_kelas = hitung_mean_image(BASE_PATH)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        warna_judul = ["#ef5350", "#ff7043", "#66bb6a", "#ffa726"]
        for idx, label in enumerate(kelas_label):
            axes[idx].imshow(mean_images[label])
            axes[idx].set_title(label, color=warna_judul[idx], fontweight="bold")
            axes[idx].axis("off")
        plt.suptitle("Mean Image per Kelas", fontsize=14, fontweight="bold")
        st.pyplot(fig)

        with st.expander("💡 Lihat Analisis Potensi Bias Model (PENTING — Baca Ini!)"):
            st.warning("""
            **⚠️ Temuan Kritis: Bias Background pada Kelas Healthy**

            Dari visualisasi Mean Image di atas, ditemukan bahwa kelas **Healthy** memiliki 
            karakteristik background yang berbeda signifikan dibanding 3 kelas penyakit lainnya 
            (cenderung lebih terang/putih).

            **Kenapa ini berbahaya?**
            Model MobileNetV2 bisa "malas" belajar — alih-alih mengenali tekstur daun, 
            model hanya menghafal warna latar belakang. Akibatnya, di lapangan ketika petani 
            memfoto daun sehat dengan latar belakang tanah gelap, model bisa salah memprediksi 
            sebagai Blast atau Brown Spot.
            """)
            st.info("""
            **✅ Rekomendasi untuk Tim AI Engineer (Fransiskus & Wega):**

            Disarankan menggunakan teknik Data Augmentation yang lebih agresif seperti 
            Random Brightness & Contrast, atau melakukan Background Segmentation (crop daun) 
            agar model terpaksa fokus pada objek daun, bukan warna latar belakangnya.
            """)

        st.divider()
        st.markdown("**📊 Distribusi Rata-rata Nilai RGB per Kelas (dari data asli)**")
        df_rgb = pd.DataFrame({
            "Kelas":    kelas_label,
            "R (Red)":  [rgb_per_kelas[l]["R"] for l in kelas_label],
            "G (Green)":[rgb_per_kelas[l]["G"] for l in kelas_label],
            "B (Blue)": [rgb_per_kelas[l]["B"] for l in kelas_label],
        })
        fig_rgb = go.Figure()
        fig_rgb.add_trace(go.Bar(name="R", x=df_rgb["Kelas"], y=df_rgb["R (Red)"],    marker_color="#ef5350"))
        fig_rgb.add_trace(go.Bar(name="G", x=df_rgb["Kelas"], y=df_rgb["G (Green)"],  marker_color="#66bb6a"))
        fig_rgb.add_trace(go.Bar(name="B", x=df_rgb["Kelas"], y=df_rgb["B (Blue)"],   marker_color="#42a5f5"))
        fig_rgb.update_layout(barmode="group", title="Rata-rata Nilai RGB per Kelas (data asli)", yaxis_title="Nilai Piksel (0-255)")
        st.plotly_chart(fig_rgb, use_container_width=True)

    # ── TAB 2: PIXEL INSPECTOR ─────────────────
    with tab2:
        st.markdown("**🔍 Interactive Image Pixel Inspector**")
        st.write("Upload foto daun padi kamu sendiri untuk melihat informasi piksel dan distribusi warnanya.")

        uploaded_file = st.file_uploader("Upload foto daun padi (JPG/PNG):", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            img       = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(img)

            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="Gambar Original", use_column_width=True)
                st.markdown(f"""
                - **Ukuran:** {img.size[0]} x {img.size[1]} px  
                - **Mode Warna:** {img.mode}  
                - **Total Piksel:** {img.size[0] * img.size[1]:,}
                """)
            with col2:
                tampilan = st.radio("Pilih tampilan:", ["Original", "Grayscale"], horizontal=True)
                if tampilan == "Grayscale":
                    st.image(img.convert("L"), caption="Grayscale", use_column_width=True)
                else:
                    st.image(img, caption="Original", use_column_width=True)

            st.divider()
            st.markdown("**📊 Histogram Distribusi Warna (R, G, B)**")
            fig_hist, ax = plt.subplots(figsize=(10, 3))
            for i, (nama, warna) in enumerate([("R", "#ef5350"), ("G", "#66bb6a"), ("B", "#42a5f5")]):
                ax.hist(img_array[:, :, i].ravel(), bins=64, alpha=0.6, color=warna, label=nama)
            ax.set_xlabel("Nilai Piksel (0-255)")
            ax.set_ylabel("Frekuensi")
            ax.legend()
            ax.set_title("Distribusi Histogram RGB")
            st.pyplot(fig_hist)

            st.divider()
            st.markdown("**📋 Statistik Piksel**")
            col1, col2, col3 = st.columns(3)
            for i, (ch, col) in enumerate(zip(["Red", "Green", "Blue"], [col1, col2, col3])):
                with col:
                    st.metric(f"Mean {ch}", f"{img_array[:,:,i].mean():.1f}")
                    st.metric(f"Std {ch}",  f"{img_array[:,:,i].std():.1f}")
        else:
            st.info("👆 Silakan upload gambar daun padi di atas untuk memulai analisis.")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.caption("🌾 RiceCare AI EDA Dashboard — Tim CC26-PSU169 | Coding Camp 2026 powered by DBS Foundation")
