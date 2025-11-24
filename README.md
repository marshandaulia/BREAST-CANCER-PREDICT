# 🎗️ Sistem Klasifikasi Kanker Payudara Multi-Kelas Berbasis Web Menggunakan CBAM dan LLM

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Flask](https://img.shields.io/badge/Framework-Flask-green)
![Gemini AI](https://img.shields.io/badge/LLM-Google%20Gemini-blueviolet)

> **Final Project - Artificial Intelligence**
> Sistem deteksi dini dan klasifikasi histopatologi kanker payudara yang tidak hanya memprediksi kelas kanker, tetapi juga memberikan penjelasan medis yang mudah dipahami menggunakan Generative AI.

---

## 📋 Daftar Isi
- [Latar Belakang](#-latar-belakang)
- [Fitur Utama](#-fitur-utama)
- [Arsitektur Sistem](#-arsitektur-sistem)
- [Dataset](#-dataset)
- [Struktur Folder](#-struktur-folder)
- [Instalasi & Cara Menjalankan](#-instalasi--cara-menjalankan)
- [Tim Pengembang](#-anggota-kelompok)

---

## 🧐 Latar Belakang
Diagnosis kanker payudara melalui citra histopatologi seringkali menantang karena kompleksitas visual dan kontras gambar yang rendah. Proyek ini bertujuan untuk membangun sistem pendukung keputusan (*decision support system*) yang membantu mengklasifikasikan jenis kanker payudara ke dalam **8 kelas spesifik** (seperti Ductal Carcinoma, Adenosis, dll) dengan akurasi tinggi menggunakan *Deep Learning* dan memberikan konteks penjelasan menggunakan *Large Language Model* (LLM).

---

## 🚀 Fitur Utama

1.  **Multi-Class Classification:** Mampu mendeteksi 8 jenis histopatologi kanker payudara (Benign & Malignant).
2.  **Advanced Preprocessing (CLAHE):** Menggunakan *Contrast Limited Adaptive Histogram Equalization* untuk mempertajam detail citra medis sebelum diproses.
3.  **Attention Mechanism (CBAM):** Mengintegrasikan *Convolutional Block Attention Module* agar model fokus pada area sel yang paling relevan.
4.  **Generative AI Explanation:** Terintegrasi dengan **Google Gemini AI** untuk memberikan penjelasan hasil prediksi yang informatif dan ramah pengguna.
5.  **User-Friendly Web Interface:** Antarmuka berbasis web yang sederhana dan responsif menggunakan Flask.

---

## 🏗️ Arsitektur Sistem

Sistem ini menggunakan pendekatan *Hybrid Architecture*:

1.  **Backbone:** `InceptionResNetV2` (Transfer Learning dari ImageNet).
2.  **Attention Module:** `CBAM` (Channel & Spatial Attention) untuk meningkatkan fokus fitur.
3.  **Preprocessing:** OpenCV dengan metode CLAHE.
4.  **Explanation Engine:** Google Gemini API (LLM).

---

## 📊 Dataset

Dataset yang digunakan adalah **BreakHis (Breast Cancer Histopathological Database)**.
* **Sumber:** [Kaggle Dataset](https://www.kaggle.com/code/manofnoego/multi-class-breast-cancer-classification-with-cbam)
* **Jumlah Kelas:** 8 Kelas
    * **Jinak (Benign):** Adenosis, Fibroadenoma, Phyllodes Tumor, Tubular Adenoma.
    * **Ganas (Malignant):** Ductal Carcinoma, Lobular Carcinoma, Mucinous Carcinoma, Papillary Carcinoma.

---

## 📂 Struktur Folder

Pastikan struktur folder proyek Anda terlihat seperti ini agar aplikasi berjalan lancar:

```text
BREAST-CANCER-PREDICT/
├── app.py                 # File utama aplikasi Flask
├── requirements.txt       # Daftar pustaka yang dibutuhkan
├── .env                   # File konfigurasi API Key (Buat Manual)
├── static/
│   ├── uploads/
│   ├── images/    
│   └── css
│   │   └── styles.css
│   └── js
│       └── app.js
│       └── upload-handler.js
├── templates/
│   ├── index.html
│   ├── predict.html
│   └── about.html
└── models/
    └── model_with_cbam.h5 # FILE MODEL WAJIB ADA DI SINI
```
## 💻 Instalasi & Cara Menjalankan

Ikuti langkah-langkah ini secara berurutan untuk menjalankan proyek di komputer lokal Anda:

### 1. Clone Repositori
Buka terminal (Command Prompt/PowerShell) dan jalankan perintah berikut:
```bash
git clone [https://github.com/marshandaulia/BREAST-CANCER-PREDICT.git](https://github.com/marshandaulia/BREAST-CANCER-PREDICT.git)
cd BREAST-CANCER-PREDICT
```
### 2. Buat Virtual Environment (Wajib)
```bash
python -m venv venv
venv\Scripts\activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Konfigurasi Model (PENTING)
```text
File model AI (model_with_cbam.h5) berukuran besar (>100MB) sehingga tidak disertakan langsung di dalam GitHub ini.
1. Unduh file model_with_cbam.h5 dari penyimpanan tim (Google Drive/Kaggle Output).
2. Salin/Copy file tersebut.
3. Tempel/Paste ke dalam folder models/ di dalam direktori proyek ini.
```
### 5. Konfigurasi API Key
### 6. Jalankan Aplikasi 
```bash
python app.py
```
### 7. Akses Website
```text
Buka browser (Chrome/Edge/Firefox) dan kunjungi alamat: http://127.0.0.1:5000/
```

## 👥 Anggota Kelompok
```text
Ketua: Marshanda Aulia Saroinsong
Anggota:
- Monica Giselle Sumual
- Chelsea Virsty Juliayarnes, Rantung
- Timo Baware Meres
- Quiland Mark Nico Wenas


