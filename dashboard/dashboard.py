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
    ZIP_ID    = "1RsW4hC-pI67eMPFnVKX6wFTbtQn7TjAE"
    ZIP_PATH  = "dataset.zip"
    EXTRACT_TO = "dataset_rice"

    if os.path.exists(EXTRACT_TO):
        return EXTRACT_TO

    gdown.download(id=ZIP_ID, output=ZIP_PATH, quiet=False)

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_TO)

    os.remove(ZIP_PATH)
    return EXTRACT_TO

with st.spinner("⏳ Memuat dataset... (hanya sekali saat pertama buka)"):
    BASE_PATH = download_dataset()

# ─────────────────────────────────────────────
# KONFIGURASI KELAS
# ─────────────────────────────────────────────
kelas_folder = ["Blast", "BrownSpot", "Healthy", "Tungro"]
kelas_label  = ["Blast", "Brown Spot", "Healthy", "Tungro"]
folder_to_label = dict(zip(kelas_folder, kelas_label))
label_to_folder = dict(zip(kelas_label, kelas_folder))

# ─────────────────────────────────────────────
# DATA HARDCODED (dari hasil split & augmentasi)
# ─────────────────────────────────────────────
# Data asli sebelum augmentasi
data_train_before = {
    "Blast":      1559,
    "Brown Spot": 1714,
    "Healthy":    1804,
    "Tungro":     1783,
}
data_test_before = {
    "Blast":      390,
    "Brown Spot": 429,
    "Healthy":    452,
    "Tungro":     446,
}
# Data setelah augmentasi (train saja yang diaugmentasi → 2000 per kelas)
data_train_after = {
    "Blast":      2000,
    "Brown Spot": 2000,
    "Healthy":    2000,
    "Tungro":     2000,
}
data_test_after = data_test_before  # test tidak diaugmentasi

# Untuk hitung_data fallback (jika folder tersedia)
@st.cache_data
def hitung_data(base_path):
    dt = {}
    dte = {}
    for folder, label in folder_to_label.items():
        pt = os.path.join(base_path, "train", folder)
        pe = os.path.join(base_path, "test",  folder)
        dt[label]  = len(glob.glob(os.path.join(pt, "*.jpg"))) if os.path.exists(pt) else data_train_before[label]
        dte[label] = len(glob.glob(os.path.join(pe, "*.jpg"))) if os.path.exists(pe) else data_test_before[label]
    return dt, dte

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

st.markdown('<div class="main-title">🌾 RiceCare AI — EDA Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Exploratory Data Analysis | Coding Camp 2026 × DBS Foundation | Tim CC26-PSU169</div>', unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/7d/Rice_p1160005.jpg/320px-Rice_p1160005.jpg", use_column_width=True)
    st.markdown("## 🌾 RiceCare AI")
    st.markdown("**EDA Dashboard**")
    st.divider()
    halaman = st.radio(
        "Pilih Halaman:",
        ["📋 Overview & Data Dictionary", "📊 Visualisasi EDA", "🖼️ Galeri Sampel Gambar"]
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
    col2.metric("Total Data Train (setelah aug)", sum(data_train_after.values()))
    col3.metric("Total Data Test",  sum(data_test_before.values()))
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

    # ── SECTION 1: DONUT CHART TRAIN & TEST ──
    st.markdown("### 🍩 Distribusi Kelas (Train & Test)")
    col1, col2 = st.columns(2)

    df_train = pd.DataFrame(list(data_train_before.items()), columns=["Kelas", "Jumlah"])
    df_test  = pd.DataFrame(list(data_test_before.items()),  columns=["Kelas", "Jumlah"])
    warna_kelas = ["#ef5350", "#ff7043", "#66bb6a", "#ffa726"]

    with col1:
        st.markdown("**Data Train (Sebelum Augmentasi)**")
        fig_train = px.pie(
            df_train, values="Jumlah", names="Kelas",
            color_discrete_sequence=warna_kelas,
            hole=0.45
        )
        fig_train.update_traces(textinfo="percent+label")
        fig_train.update_layout(showlegend=True, margin=dict(t=30, b=10))
        st.plotly_chart(fig_train, use_container_width=True)

    with col2:
        st.markdown("**Data Test**")
        fig_test = px.pie(
            df_test, values="Jumlah", names="Kelas",
            color_discrete_sequence=warna_kelas,
            hole=0.45
        )
        fig_test.update_traces(textinfo="percent+label")
        fig_test.update_layout(showlegend=True, margin=dict(t=30, b=10))
        st.plotly_chart(fig_test, use_container_width=True)

    st.divider()

    # ── SECTION 2: BAR CHART PERBANDINGAN TRAIN VS TEST ──
    st.markdown("### 📊 Perbandingan Jumlah Data Train vs Test per Kelas")
    df_compare = pd.DataFrame({
        "Kelas": kelas_label,
        "Train": [data_train_before[l] for l in kelas_label],
        "Test":  [data_test_before[l]  for l in kelas_label]
    })
    fig_group = px.bar(
        df_compare.melt(id_vars="Kelas", var_name="Split", value_name="Jumlah"),
        x="Kelas", y="Jumlah", color="Split", barmode="group",
        color_discrete_sequence=["#42a5f5", "#ef5350"],
        text="Jumlah"
    )
    fig_group.update_traces(textposition="outside")
    fig_group.update_layout(yaxis_title="Jumlah Gambar", legend_title="Split")
    st.plotly_chart(fig_group, use_container_width=True)

    st.divider()

    # ── SECTION 3: BEFORE vs AFTER AUGMENTASI ──
    st.markdown("### ⚖️ Distribusi Data Train: Sebelum vs Sesudah Augmentasi")
    st.write("Augmentasi dilakukan **hanya pada data train** untuk menyeimbangkan jumlah tiap kelas menjadi **2.000 gambar per kelas**.")

    df_before = pd.DataFrame({
        "Kelas": kelas_label,
        "Jumlah": [data_train_before[l] for l in kelas_label]
    })
    df_after = pd.DataFrame({
        "Kelas": kelas_label,
        "Jumlah": [data_train_after[l] for l in kelas_label]
    })

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Sebelum Augmentasi**")
        fig_before = px.bar(
            df_before, x="Kelas", y="Jumlah",
            color="Kelas",
            color_discrete_sequence=warna_kelas,
            text="Jumlah"
        )
        fig_before.update_traces(textposition="outside")
        fig_before.update_layout(
            showlegend=False,
            yaxis_title="Jumlah Gambar",
            yaxis_range=[0, 2300]
        )
        st.plotly_chart(fig_before, use_container_width=True)

    with col2:
        st.markdown("**Sesudah Augmentasi**")
        fig_after = px.bar(
            df_after, x="Kelas", y="Jumlah",
            color="Kelas",
            color_discrete_sequence=warna_kelas,
            text="Jumlah"
        )
        fig_after.update_traces(textposition="outside")
        fig_after.update_layout(
            showlegend=False,
            yaxis_title="Jumlah Gambar",
            yaxis_range=[0, 2300]
        )
        st.plotly_chart(fig_after, use_container_width=True)

    # Tabel ringkasan delta
    df_delta = pd.DataFrame({
        "Kelas":           kelas_label,
        "Sebelum (Train)": [data_train_before[l] for l in kelas_label],
        "Sesudah (Train)": [data_train_after[l]  for l in kelas_label],
        "Penambahan":      [data_train_after[l] - data_train_before[l] for l in kelas_label],
    })
    st.dataframe(df_delta, use_container_width=True, hide_index=True)

    with st.expander("💡 Mengapa augmentasi diperlukan?"):
        st.info("""
        **Distribusi data train sebelum augmentasi tidak seimbang:**
        - Kelas terbanyak (Healthy): **1.804** gambar
        - Kelas tersedikit (Blast): **1.559** gambar
        - Rasio: ±1.16x — masih moderat, namun tetap bisa menyebabkan bias

        **Setelah augmentasi**, semua kelas train menjadi **2.000 gambar** (balanced).
        Teknik augmentasi yang digunakan: rotasi, flip horizontal/vertikal, brightness & contrast adjustment, dan zoom.
        
        Data test **tidak diaugmentasi** agar tetap merepresentasikan kondisi nyata di lapangan.
        """)

    st.divider()

    # ── SECTION 4: MEAN IMAGE PER KELAS ──
    st.markdown("### 🖼️ Rata-rata Piksel (Mean Image) per Kelas")
    st.write("Visualisasi karakteristik warna dan tekstur rata-rata tiap kelas penyakit dari 50 sampel gambar train.")

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
                hasil[label] = mean_arr
                hasil_rgb[label] = {
                    "R": float(mean_arr[:,:,0].mean()),
                    "G": float(mean_arr[:,:,1].mean()),
                    "B": float(mean_arr[:,:,2].mean())
                }
            else:
                placeholder = np.random.randint(80, 180, (224, 224, 3), dtype=np.uint8)
                hasil[label] = placeholder
                hasil_rgb[label] = {"R": 128.0, "G": 128.0, "B": 128.0}
        return hasil, hasil_rgb

    with st.spinner("Menghitung mean image dari dataset..."):
        mean_images, rgb_per_kelas = hitung_mean_image(BASE_PATH)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    warna_judul = ["#ef5350", "#ff7043", "#66bb6a", "#ffa726"]
    for idx, label in enumerate(kelas_label):
        axes[idx].imshow(mean_images[label])
        axes[idx].set_title(label, color=warna_judul[idx], fontweight="bold", fontsize=13)
        axes[idx].axis("off")
    plt.suptitle("Mean Image per Kelas (50 sampel train)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)

    with st.expander("⚠️ Analisis Potensi Bias Model — Baca Ini!"):
        st.warning("""
        **Temuan Kritis: Bias Background pada Kelas Healthy**

        Dari mean image di atas, kelas **Healthy** memiliki karakteristik background yang berbeda 
        signifikan dibanding 3 kelas penyakit lainnya (cenderung lebih terang/putih).

        Model MobileNetV2 bisa "malas" belajar — alih-alih mengenali tekstur daun, 
        model hanya menghafal warna latar belakang. Di lapangan, ketika petani memfoto daun sehat 
        dengan latar belakang tanah gelap, model bisa salah memprediksi sebagai Blast atau Brown Spot.
        """)
        st.info("""
        **✅ Rekomendasi untuk Tim AI Engineer:**

        Gunakan teknik Data Augmentation yang lebih agresif (Random Brightness & Contrast), 
        atau lakukan Background Segmentation (crop daun) agar model fokus pada tekstur daun, 
        bukan warna latar belakangnya.
        """)

    st.divider()

    # ── SECTION 5: DISTRIBUSI RGB ──
    st.markdown("### 🎨 Distribusi Rata-rata Nilai RGB per Kelas")
    df_rgb = pd.DataFrame({
        "Kelas":     kelas_label,
        "R (Red)":   [rgb_per_kelas[l]["R"] for l in kelas_label],
        "G (Green)": [rgb_per_kelas[l]["G"] for l in kelas_label],
        "B (Blue)":  [rgb_per_kelas[l]["B"] for l in kelas_label],
    })
    fig_rgb = go.Figure()
    fig_rgb.add_trace(go.Bar(name="R", x=df_rgb["Kelas"], y=df_rgb["R (Red)"],    marker_color="#ef5350"))
    fig_rgb.add_trace(go.Bar(name="G", x=df_rgb["Kelas"], y=df_rgb["G (Green)"],  marker_color="#66bb6a"))
    fig_rgb.add_trace(go.Bar(name="B", x=df_rgb["Kelas"], y=df_rgb["B (Blue)"],   marker_color="#42a5f5"))
    fig_rgb.update_layout(
        barmode="group",
        title="Rata-rata Nilai RGB per Kelas (dari 50 sampel train per kelas)",
        yaxis_title="Nilai Piksel (0–255)",
        legend_title="Channel",
    )
    st.plotly_chart(fig_rgb, use_container_width=True)

    st.caption("💡 Nilai G (Green) yang tinggi pada kelas Healthy mengindikasikan daun yang lebih hijau dan segar dibanding kelas penyakit.")

# ─────────────────────────────────────────────
# HALAMAN 3: GALERI SAMPEL GAMBAR
# ─────────────────────────────────────────────
elif halaman == "🖼️ Galeri Sampel Gambar":
    st.subheader("🖼️ Galeri Sampel Gambar Per Kelas")

    col_sel, col_jml = st.columns([2, 1])
    with col_sel:
        kelas_dipilih = st.selectbox("Pilih Kelas Penyakit:", kelas_label)
    with col_jml:
        pilihan_jumlah = st.selectbox(
            "Jumlah gambar:",
            options=list(range(1, 11)),
            index=3,   # default 4
            format_func=lambda x: f"{x} gambar"
        )

    folder_dipilih = label_to_folder[kelas_dipilih]
    folder_path    = os.path.join(BASE_PATH, "train", folder_dipilih)

    if os.path.exists(folder_path):
        semua_gambar = (
            glob.glob(os.path.join(folder_path, "*.jpg")) +
            glob.glob(os.path.join(folder_path, "*.JPG")) +
            glob.glob(os.path.join(folder_path, "*.png"))
        )
        if semua_gambar:
            import random
            sampel = random.sample(semua_gambar, min(pilihan_jumlah, len(semua_gambar)))
            # Tampilkan maksimal 4 kolom per baris
            n_cols = min(pilihan_jumlah, 4)
            cols   = st.columns(n_cols)
            for i, path_gambar in enumerate(sampel):
                with cols[i % n_cols]:
                    img = Image.open(path_gambar)
                    st.image(img, caption=os.path.basename(path_gambar), use_column_width=True)
        else:
            st.warning("Tidak ada gambar ditemukan di folder ini.")
    else:
        st.error(f"Folder tidak ditemukan: `{folder_path}`")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.caption("🌾 RiceCare AI EDA Dashboard — Tim CC26-PSU169 | Coding Camp 2026 powered by DBS Foundation")
