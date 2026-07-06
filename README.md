# 🦺 APD Monitoring — YOLO11 Simulasi CCTV

Aplikasi web (Streamlit) untuk memonitor kepatuhan Alat Pelindung Diri (APD) —
helm & rompi keselamatan — pada area konstruksi, menggunakan model **YOLO11**
dan simulasi feed CCTV dari video/gambar yang diunggah.

---

## 1. Struktur Folder

```
apd_monitoring_app/
├── app.py                  # Aplikasi utama Streamlit
├── requirements.txt        # Daftar library yang dibutuhkan
├── README.md                # Dokumen ini
├── utils/
│   ├── detector.py         # Wrapper model YOLO11
│   └── helpers.py          # Logging & statistik
├── models/                  # (opsional) letakkan file best.pt hasil training di sini
└── sample_data/              # (opsional) video/gambar contoh untuk uji coba
```

---

## 2. Kebutuhan Sistem (Bahan-bahan)

| Kebutuhan | Keterangan |
|---|---|
| Python | versi 3.9 – 3.11 (disarankan 3.10) |
| pip | untuk instalasi library |
| GPU (opsional) | mempercepat inferensi, tidak wajib. CPU tetap bisa jalan |
| File model `.pt` | model YOLO11 hasil training (custom) untuk deteksi helm/rompi, atau pakai model default `yolo11n.pt` (generic, belum dilatih khusus APD) |
| Video/gambar contoh | untuk mensimulasikan feed CCTV |

**Catatan penting:** Model default `yolo11n.pt` dari Ultralytics **belum
dilatih khusus untuk kelas APD** (helm/rompi). Untuk hasil sesuai skripsi
(deteksi helm & vest + status pelanggaran), kamu perlu **melatih model
sendiri** di Google Colab menggunakan dataset dari Roboflow (sesuai
metodologi di proposal: Data Collection → Labelling → Balancing → Split →
Training → Evaluasi), lalu unggah file `best.pt` hasil training melalui
sidebar aplikasi ("Unggah model kustom").

---

## 3. Langkah Instalasi

### a. Install Python
Unduh & install Python 3.10 dari https://www.python.org/downloads/
Pastikan saat instalasi mencentang "Add Python to PATH".

### b. Buat folder proyek & masuk ke dalamnya
```bash
mkdir apd_monitoring_app
cd apd_monitoring_app
```
Salin semua file yang diberikan (app.py, requirements.txt, folder utils/, dll)
ke dalam folder ini.

### c. Buat virtual environment (disarankan)
```bash
python -m venv venv
```
Aktifkan:
- Windows:
  ```bash
  venv\Scripts\activate
  ```
- macOS/Linux:
  ```bash
  source venv/bin/activate
  ```

### d. Install semua library yang dibutuhkan
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Proses ini akan menginstall:
- **streamlit** — framework web app
- **ultralytics** — library resmi YOLO11
- **opencv-python-headless** — pengolahan video/gambar
- **torch & torchvision** — backend deep learning
- **pandas, numpy, pillow** — pengolahan data & gambar

> Instalasi pertama kali bisa memakan waktu 5–15 menit karena `torch`
> berukuran besar. Pastikan koneksi internet stabil.

### e. Jalankan aplikasi
```bash
streamlit run app.py
```

Setelah itu, terminal akan menampilkan alamat seperti:
```
Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```
Buka alamat **Local URL** tersebut di browser (Chrome/Edge/Firefox) — aplikasi
akan langsung terbuka.

---

## 4. Cara Menggunakan Aplikasi

1. **Sidebar → Konfigurasi Model**
   - Pilih model default, atau unggah file `best.pt` hasil training sendiri.
   - Atur *confidence threshold* (ambang kepercayaan deteksi).
   - Klik **"Muat / Perbarui Model"**.

2. **Menu 📊 Dashboard**
   Menampilkan ringkasan KPI: total frame diproses, jumlah orang terdeteksi,
   jumlah pelanggaran, dan tingkat kepatuhan (%) beserta grafik tren.

3. **Menu 🎥 Simulasi CCTV (Video)**
   - Unggah video area konstruksi (.mp4/.avi/.mov).
   - Klik **"Mulai Simulasi"** — video akan diproses frame demi frame seperti
     feed CCTV, lengkap dengan bounding box, label, dan notifikasi
     peringatan real-time saat ada pelanggaran (tidak pakai helm/rompi).
   - Klik **"Hentikan"** untuk menghentikan simulasi kapan saja.

4. **Menu 🖼️ Deteksi Gambar**
   - Unggah satu foto untuk diperiksa kepatuhan APD-nya secara instan.

5. **Menu 📁 Riwayat & Statistik**
   - Melihat seluruh log deteksi dalam bentuk tabel.
   - Mengunduh log sebagai file CSV.
   - Menghapus riwayat jika perlu memulai sesi baru.

6. **Menu ℹ️ Tentang**
   - Informasi proyek dan anggota kelompok.

---

## 5. Cara Melatih Model Sendiri (opsional, sesuai metodologi skripsi)

1. Kumpulkan/gunakan dataset dari **Roboflow** (contoh: dataset "Construction
   Site Safety" berisi kelas Hardhat/NO-Hardhat/Safety Vest/NO-Safety
   Vest/Person).
2. Export dataset dalam format **YOLOv11 (Ultralytics)**.
3. Latih model di **Google Colab** (gratis, sudah ada GPU):
   ```python
   !pip install ultralytics
   from ultralytics import YOLO

   model = YOLO("yolo11n.pt")  # atau yolo11s.pt untuk akurasi lebih tinggi
   model.train(data="data.yaml", epochs=50, imgsz=640)
   ```
4. Setelah training selesai, ambil file `runs/detect/train/weights/best.pt`.
5. Unggah file `best.pt` tersebut ke aplikasi Streamlit lewat sidebar.

**Penting:** Sistem ini otomatis menganggap kelas yang namanya diawali
`NO-` / `No-` (misalnya `NO-Hardhat`, `NO-Safety Vest`) sebagai **pelanggaran**.
Pastikan penamaan kelas saat labelling di Roboflow konsisten dengan format ini,
atau sesuaikan logika di `utils/detector.py` bagian `is_violation`.

---

## 6. Troubleshooting

| Masalah | Solusi |
|---|---|
| `ModuleNotFoundError: No module named 'cv2'` | Jalankan ulang `pip install -r requirements.txt` |
| Aplikasi lambat saat proses video | Naikkan nilai "Proses setiap N frame" di menu Simulasi CCTV |
| Model gagal dimuat | Pastikan file `.pt` valid hasil export/training Ultralytics YOLO11 |
| Video tidak tampil di browser | Gunakan format `.mp4` (H.264) untuk kompatibilitas terbaik |
| Ingin deploy online | Gunakan [Streamlit Community Cloud](https://streamlit.io/cloud), Hugging Face Spaces, atau server VPS dengan `streamlit run app.py --server.port 8501` |

---

## 7. Kredit

Dikembangkan berdasarkan proposal skripsi:
**"Implementasi YOLO11 untuk Monitoring Real-Time Kepatuhan Alat Pelindung
Diri (APD) pada Area Konstruksi Melalui Simulasi CCTV"**

Kelompok:
- Bintang Suriya Bagus Marchvindo (A11.2024.16077) — Ketua
- Reva Juni Pratama (A11.2024.16063)
- Petrus Damianus Sigit Dwi S. (A11.2022.14764)
- Muhammad Ramadhan Badali (A11.2023.15450)
- Rahmad Izza Nur Latif (A11.2024.15843)
