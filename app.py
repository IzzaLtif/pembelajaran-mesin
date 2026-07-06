"""
app.py
=========================================================
Implementasi YOLO11 untuk Monitoring Real-Time Kepatuhan
Alat Pelindung Diri (APD) pada Area Konstruksi
Melalui Simulasi CCTV
=========================================================

Cara jalankan:
    streamlit run app.py

Lihat README.md untuk panduan instalasi lengkap.
"""

import streamlit as st
import numpy as np
import cv2
import tempfile
import time
import os
from PIL import Image

from utils.detector import PPEDetector
from utils.helpers import new_log_entry, log_to_dataframe, compute_kpis, df_to_csv_bytes

# ----------------------------------------------------------------------------
# KONFIGURASI HALAMAN
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="APD Monitoring | YOLO11",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#F2A007"     # safety orange/yellow
DARK = "#0B0D10"        # darker charcoal
DANGER = "#E63946"      # merah pelanggaran
SAFE = "#2A9D8F"        # hijau/teal patuh

CUSTOM_CSS = f"""
<style>
  :root {{ --primary: {PRIMARY}; --danger: {DANGER}; --safe: {SAFE}; --dark: {DARK}; }}
  body, .stApp {{ background-color: var(--dark); color: #E6EEF3; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; }}
  h1, h2, h3 {{ color: #FFFFFF !important; }}
  .kpi-card {{ background: #0F1418; border-radius: 12px; padding: 14px; border: 1px solid #1F262A; box-shadow: 0 4px 14px rgba(0,0,0,0.4); }}
  .kpi-value {{ font-size: 28px; font-weight: 700; color: var(--primary); }}
  .kpi-label {{ font-size: 12px; color: #9AA0AC; text-transform: uppercase; letter-spacing: 0.04em; }}
  .alert-violation {{ background: rgba(230,57,70,0.12); border: 1px solid var(--danger); color: #FFD1D6; padding: 10px 16px; border-radius: 10px; font-weight: 700; }}
  .alert-safe {{ background: rgba(42,157,143,0.08); border: 1px solid var(--safe); color: #BFF3E8; padding: 10px 16px; border-radius: 10px; font-weight: 700; }}
  section[data-testid="stSidebar"] {{ background-color: #071018; padding-top: 12px; }}
  /* make uploads and buttons easier to hit */
  .stFileUploader, .stButton {{ border-radius: 10px; }}
  /* accessible focus outlines */
  .stButton button:focus, .stFileUploader input:focus {{ outline: 3px solid rgba(242,160,7,0.25); outline-offset: 2px; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------------
if "log" not in st.session_state:
    st.session_state.log = []
if "running" not in st.session_state:
    st.session_state.running = False
if "detector" not in st.session_state:
    st.session_state.detector = None

# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
st.sidebar.markdown("## 🦺 APD Monitoring")
st.sidebar.caption("YOLO11 · Simulasi CCTV Konstruksi")

page = st.sidebar.radio(
    "Navigasi",
    ["📊 Dashboard", "🎥 Simulasi CCTV (Video)", "🖼️ Deteksi Gambar", "📁 Riwayat & Statistik", "ℹ️ Tentang"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Konfigurasi Model")

model_source = st.sidebar.radio(
    "Sumber model", ["Model default (yolo11n.pt)", "Unggah model kustom (.pt)"],
    help="Untuk hasil akurat, latih model YOLO11 sendiri di dataset APD (Roboflow) lalu unggah file best.pt di sini.",
)

model_path = "yolo11n.pt"
if model_source == "Unggah model kustom (.pt)":
    uploaded_model = st.sidebar.file_uploader("Unggah file model (.pt)", type=["pt"])
    if uploaded_model is not None:
        tmp_model_path = os.path.join(tempfile.gettempdir(), uploaded_model.name)
        with open(tmp_model_path, "wb") as f:
            f.write(uploaded_model.read())
        model_path = tmp_model_path

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05)

if st.sidebar.button("🔄 Muat / Perbarui Model", use_container_width=True):
    with st.spinner("Memuat model YOLO11..."):
        st.session_state.detector = PPEDetector(model_path=model_path, conf_threshold=conf_threshold)
    st.sidebar.success("Model berhasil dimuat!")

if st.session_state.detector is not None:
    st.session_state.detector.set_confidence(conf_threshold)
    st.sidebar.info(f"Kelas model: {list(st.session_state.detector.class_names.values())}")

st.sidebar.markdown("---")
st.sidebar.caption("Kelompok: Bintang S. B. Marchvindo, dkk — Implementasi YOLO11 Monitoring APD")


def ensure_detector():
    if st.session_state.detector is None:
        try:
            with st.spinner("Memuat model default..."):
                st.session_state.detector = PPEDetector(model_path=model_path, conf_threshold=conf_threshold)
        except Exception as e:
            st.sidebar.error(f"Gagal memuat model: {e}")
            raise
    return st.session_state.detector


# ----------------------------------------------------------------------------
# PAGE: DASHBOARD
# ----------------------------------------------------------------------------
if page == "📊 Dashboard":
    st.title("📊 Dashboard Monitoring Kepatuhan APD")
    st.write(
        "Ringkasan kepatuhan penggunaan **helm** dan **rompi keselamatan** "
        "pada area konstruksi berdasarkan simulasi CCTV yang telah diproses."
    )

    df = log_to_dataframe(st.session_state.log)
    kpis = compute_kpis(df)

    c1, c2, c3, c4 = st.columns(4)
    kpi_items = [
        ("Frame Diproses", kpis["total_frame"], ""),
        ("Total Orang Terdeteksi", kpis["total_orang_terdeteksi"], ""),
        ("Frame Pelanggaran", kpis["total_pelanggaran"], ""),
        ("Tingkat Kepatuhan", f"{kpis['tingkat_kepatuhan']}%", ""),
    ]
    for col, (label, value, _) in zip((c1, c2, c3, c4), kpi_items):
        with col:
            st.markdown(
                f"""<div class="kpi-card" role="region" aria-label="{label}">
                        <div class="kpi-label">{label}</div>
                        <div class="kpi-value">{value}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("### 📈 Tren Kepatuhan per Frame")
    if not df.empty:
        chart_df = df.copy()
        chart_df["melanggar_flag"] = (chart_df["status"] == "MELANGGAR").astype(int)
        st.line_chart(chart_df["melanggar_flag"], height=250)

        st.markdown("### 🔎 Distribusi Status Deteksi")
        status_counts = df["status"].value_counts()
        st.bar_chart(status_counts)
    else:
        st.info("Belum ada data. Jalankan deteksi di menu **Simulasi CCTV** atau **Deteksi Gambar** terlebih dahulu.")

# ----------------------------------------------------------------------------
# PAGE: SIMULASI CCTV (VIDEO)
# ----------------------------------------------------------------------------
elif page == "🎥 Simulasi CCTV (Video)":
    st.title("🎥 Simulasi CCTV — Monitoring Real-Time")
    st.write("Unggah video rekaman area konstruksi untuk disimulasikan sebagai feed CCTV.")

    video_file = st.file_uploader("Unggah video (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])
    frame_skip = st.slider("Proses setiap N frame (mempercepat simulasi)", 1, 10, 2, help="Semakin besar nilainya, semakin sedikit frame yang diproses.")

    col_a, col_b = st.columns([1,1])
    start_btn = col_a.button("▶️ Mulai Simulasi", use_container_width=True, type="primary")
    stop_btn = col_b.button("⏹️ Hentikan", use_container_width=True)

    if stop_btn:
        st.session_state.running = False

    video_placeholder = st.empty()
    alert_placeholder = st.empty()
    metric_placeholder = st.empty()

    if start_btn and video_file is not None:
        try:
            detector = ensure_detector()
        except Exception:
            st.error("Tidak dapat memuat model. Periksa log di sidebar.")
        else:
            st.session_state.running = True

            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(video_file.read())
            cap = cv2.VideoCapture(tfile.name)

            frame_idx = 0
            processed = 0
            while cap.isOpened() and st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if frame_idx % frame_skip != 0:
                    continue

                try:
                    detections, annotated, infer_ms = detector.predict(frame)
                    summary = detector.summarize(detections)
                except Exception as e:
                    st.error(f"Error saat inferensi: {e}")
                    break

                entry = new_log_entry(video_file.name, summary)
                st.session_state.log.append(entry)

                if annotated is not None:
                    try:
                        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)
                    except Exception:
                        video_placeholder.text("(Preview tidak tersedia)")

                if summary["violation_items"] > 0:
                    alert_placeholder.markdown(
                        f"""<div class="alert-violation" role="alert">🚨 PELANGGARAN TERDETEKSI — {", ".join(summary['violation_labels'])}</div>""",
                        unsafe_allow_html=True,
                    )
                else:
                    alert_placeholder.markdown(
                        """<div class="alert-safe" role="status">✅ Kondisi aman — seluruh APD terpakai sesuai standar</div>""",
                        unsafe_allow_html=True,
                    )

                m1, m2, m3 = metric_placeholder.columns(3)
                m1.metric("Orang Terdeteksi", summary["persons"])
                m2.metric("Item Melanggar", summary["violation_items"])
                m3.metric("Waktu Inferensi", f"{infer_ms:.1f} ms")

                processed += 1
                time.sleep(0.02)

            cap.release()
            st.session_state.running = False
            st.success(f"Simulasi selesai. {len(st.session_state.log)} entri tercatat di log. (Diproses: {processed} frame)")

    elif start_btn and video_file is None:
        st.warning("Silakan unggah video terlebih dahulu.")

# ----------------------------------------------------------------------------
# PAGE: DETEKSI GAMBAR
# ----------------------------------------------------------------------------
elif page == "🖼️ Deteksi Gambar":
    st.title("🖼️ Deteksi APD pada Gambar")
    st.write("Unggah satu foto area konstruksi untuk diperiksa kepatuhan APD-nya.")

    img_file = st.file_uploader("Unggah gambar (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

    if img_file is not None:
        try:
            detector = ensure_detector()
        except Exception:
            st.error("Tidak dapat memuat model untuk deteksi gambar.")
        else:
            image = Image.open(img_file).convert("RGB")
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            with st.spinner("Menjalankan deteksi..."):
                try:
                    detections, annotated, infer_ms = detector.predict(frame)
                    summary = detector.summarize(detections)
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat inferensi: {e}")
                    detections, annotated, infer_ms, summary = [], None, 0.0, {"persons":0, "violation_items":0, "violation_labels":[]}

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Gambar Asli**")
                st.image(image, use_container_width=True)
            with col2:
                st.markdown("**Hasil Deteksi**")
                if annotated is not None:
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                else:
                    st.text("(Tidak ada preview hasil deteksi)")

            if summary["violation_items"] > 0:
                st.markdown(
                    f"""<div class="alert-violation">🚨 Ditemukan {summary['violation_items']} pelanggaran: {", ".join(summary['violation_labels'])}</div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """<div class="alert-safe">✅ Semua APD terpakai sesuai standar</div>""",
                    unsafe_allow_html=True,
                )

            entry = new_log_entry(img_file.name, summary)
            st.session_state.log.append(entry)
            st.caption(f"Waktu inferensi: {infer_ms:.1f} ms | Orang terdeteksi: {summary['persons']}")

# ----------------------------------------------------------------------------
# PAGE: RIWAYAT & STATISTIK
# ----------------------------------------------------------------------------
elif page == "📁 Riwayat & Statistik":
    st.title("📁 Riwayat & Statistik Deteksi")

    df = log_to_dataframe(st.session_state.log)
    if df.empty:
        st.info("Belum ada riwayat deteksi.")
    else:
        st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True, height=400)

        csv_bytes = df_to_csv_bytes(df)
        st.download_button(
            "⬇️ Unduh Log (CSV)", data=csv_bytes, file_name="log_kepatuhan_apd.csv", mime="text/csv"
        )

        if st.button("🗑️ Hapus Semua Riwayat"):
            st.session_state.log = []
            st.rerun()

# ----------------------------------------------------------------------------
# PAGE: TENTANG
# ----------------------------------------------------------------------------
elif page == "ℹ️ Tentang":
    st.title("ℹ️ Tentang Proyek")
    st.markdown(
        """
        ### Implementasi YOLO11 untuk Monitoring Real-Time Kepatuhan APD
        pada Area Konstruksi Melalui Simulasi CCTV

        **Latar Belakang:** Sektor konstruksi menyumbang sekitar 32% dari total
        kasus kecelakaan kerja setiap tahunnya. Metode pengawasan manual tidak
        konsisten dan sulit dilakukan secara real-time, sehingga dibutuhkan
        solusi *computer vision* berbasis YOLO11 untuk deteksi otomatis.

        **Fokus Deteksi:** Helm (*helmet*) dan rompi keselamatan (*vest*).

        **Metodologi:** Data Collection (Roboflow) → Data Labelling → Data
        Balancing → Split Dataset → Modeling YOLO11 → Model Training →
        Evaluasi (Loss & mAP) → Uji Coba Prototipe.

        ---
        ### Anggota Kelompok
        - Ketua: **Bintang Suriya Bagus Marchvindo** (A11.2024.16077)
        - Reva Juni Pratama (A11.2024.16063)
        - Petrus Damianus Sigit Dwi S. (A11.2022.14764)
        - Muhammad Ramadhan Badali (A11.2023.15450)
        - Rahmad Izza Nur Latif (A11.2024.15843)
        """
    )
