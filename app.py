import os
import numpy as np
import cv2
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Setup Gemini
# Read API key from environment first. If it's missing, avoid calling
# genai.configure(...) with a None key and disable LLM calls safely.
# NOTE: The API key has been embedded here per your request. This is
# convenient for quick testing but insecure for production. Prefer
# storing secrets in environment variables or a .env file.
GEMINI_API_KEY = 'AIzaSyDr0Cq4k7vqTmOCtbwCh6ckb3vvujnPv1o'
genai.configure(api_key=GEMINI_API_KEY)
gemini_llm = genai.GenerativeModel('gemini-2.5-flash')
print("Gemini LLM initialized successfully.")

# Import Keras/TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

# Import untuk CBAM dan Arsitektur Model
from tensorflow.keras.layers import (
    Layer, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Reshape, 
    Multiply, Conv2D, Concatenate, Activation, Add,
    Input, Flatten, BatchNormalization, Dropout
)
from tensorflow.keras.applications import InceptionResNetV2

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Konfigurasi path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model_with_cbam.h5')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Pastikan folder uploads ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def get_explanation(prediction_class, confidence):
    """
    Mendapatkan penjelasan dari Gemini tentang hasil klasifikasi
    """
    try:
        # If the Gemini client wasn't initialized because the API key
        # wasn't provided, return a helpful message instead of attempting
        # to call the library and raising the ADC/No API key error.
        if gemini_llm is None:
            return (
                "Penjelasan LLM tidak tersedia karena kunci API Gemini belum dikonfigurasi. "
                "Silakan atur variabel lingkungan GEMINI_API_KEY atau konsultasikan hasil ini dengan dokter Anda."
            )
        prompt = f"""Sebagai seorang ahli patologi, berikan penjelasan singkat dalam bahasa Indonesia yang mudah dipahami tentang hasil klasifikasi histopatologi kanker payudara ini:

        HASIL DETEKSI:
        {prediction_class}
        (Tingkat keyakinan: {confidence:.2f}%)

        Tolong jelaskan dalam format berikut:
        1️⃣ Karakteristik: [jelaskan ciri khas dari tipe kanker ini]
        2️⃣ Tingkat Keparahan: [jelaskan seberapa serius kondisi ini]
        3️⃣ Saran Tindakan: [berikan 2-3 saran tindakan lanjutan]

        Gunakan bahasa yang sederhana dan empatik. Hindari istilah medis yang terlalu teknis."""

        # Use the dedicated Gemini client (gemini_llm) so we don't call
        # methods on the TensorFlow model by mistake.
        response = gemini_llm.generate_content(prompt,
                                       generation_config={
                                           'temperature': 0.3,
                                           'top_k': 32,
                                           'max_output_tokens': 150
                                       })
        return response.text
    except Exception as e:
        print(f"Error saat berkomunikasi dengan Gemini: {str(e)}")
        return "Maaf, terjadi kesalahan saat menghasilkan penjelasan. Silakan konsultasikan hasil ini dengan dokter Anda."

# ----------------------------------------------------------------------------
# 1. Fungsi CLAHE (dari notebook)
# ----------------------------------------------------------------------------
def apply_clahe_to_color_image(image):
    """Menerapkan CLAHE ke gambar warna BGR (dari cv2.imread)."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    ca = clahe.apply(a)
    cb = clahe.apply(b)
    lab = cv2.merge((cl, ca, cb))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# ----------------------------------------------------------------------------
# 2. Definisi Class CBAM (dari notebook)
# ----------------------------------------------------------------------------
class CBAM(Layer):
    def __init__(self, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channel = input_shape[-1]
        self.shared_dense_one = Dense(channel // self.reduction_ratio,
                                      activation='relu',
                                      kernel_initializer='he_normal',
                                      use_bias=True,
                                      bias_initializer='zeros')
        self.shared_dense_two = Dense(channel,
                                      kernel_initializer='he_normal',
                                      use_bias=True,
                                      bias_initializer='zeros')
        self.conv2d = Conv2D(filters=1,
                             kernel_size=7,
                             strides=1,
                             padding='same',
                             activation='sigmoid',
                             kernel_initializer='he_normal',
                             use_bias=False)

    def call(self, input_feature):
        # Channel Attention Module
        avg_pool = GlobalAveragePooling2D()(input_feature)
        avg_pool = Reshape((1, 1, input_feature.shape[-1]))(avg_pool)
        avg_pool = self.shared_dense_one(avg_pool)
        avg_pool = self.shared_dense_two(avg_pool)

        max_pool = GlobalMaxPooling2D()(input_feature)
        max_pool = Reshape((1, 1, input_feature.shape[-1]))(max_pool)
        max_pool = self.shared_dense_one(max_pool)
        max_pool = self.shared_dense_two(max_pool)

        channel_attention = Activation('sigmoid')(Add()([avg_pool, max_pool]))
        channel_refined = Multiply()([input_feature, channel_attention])

        # Spatial Attention Module
        avg_pool = tf.reduce_mean(channel_refined, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(channel_refined, axis=-1, keepdims=True)
        concat = Concatenate(axis=-1)([avg_pool, max_pool])
        spatial_attention = self.conv2d(concat)
        refined_feature = Multiply()([channel_refined, spatial_attention])

        return refined_feature
    
    def get_config(self):
        """Agar model bisa di-save dan di-load."""
        config = super(CBAM, self).get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config

# ----------------------------------------------------------------------------
# 3. Fungsi Arsitektur Model (dari notebook) - DENGAN PERBAIKAN
# ----------------------------------------------------------------------------
def build_model_with_cbam(input_shape=(224, 224, 3), num_classes=8):
    base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # --- PERBAIKAN DI SINI ---
    # Harus 'True' agar arsitekturnya sama persis dengan notebook saat menyimpan
    base_model.trainable = True 
    # -------------------------

    inputs = Input(shape=input_shape)
    # 'training=False' di sini memastikan model berjalan dalam mode inferensi
    # (misalnya, BatchNormalization menggunakan statistik yang tersimpan)
    x = base_model(inputs, training=False) 
    x = CBAM()(x)
    x = Flatten()(x)
    
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)

    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)

# ----------------------------------------------------------------------------
# 4. Daftar Nama Kelas
# ----------------------------------------------------------------------------
CLASS_NAMES = [
    'Adenosis',
    'Ductal Carcinoma',
    'Fibroadenoma',
    'Lobular Carcinoma',
    'Mucinous Carcinoma',
    'Papillary Carcinoma',
    'Phyllodes Tumor',
    'Tubular Adenoma'
]

# ----------------------------------------------------------------------------
# 5. Muat Model (Cara yang Diperbaiki)
# ----------------------------------------------------------------------------
model = None
try:
    print("Membangun arsitektur model...")
    # Panggil fungsi untuk membangun arsitektur (jumlah kelas = 8)
    model = build_model_with_cbam(input_shape=(224, 224, 3), num_classes=8)
    
    print(f"Mencoba memuat bobot (weights) dari: {MODEL_PATH}")
    # Muat HANYA bobot (weights) ke arsitektur yang ada
    model.load_weights(MODEL_PATH)
    
    print("Model dan bobot (weights) berhasil dimuat.")

except Exception as e:
    print(f"GAGAL MEMUAT MODEL/BOBOT: {e}")
    print("Pastikan file 'model_with_cbam.h5' ada di folder 'models' dan arsitektur model di app.py sama dengan di notebook.")
    model = None

# ----------------------------------------------------------------------------
# 6. Fungsi Preprocessing dan Prediksi
# ----------------------------------------------------------------------------
def model_predict(img_path, model):
    """
    Melakukan preprocessing gambar sesuai notebook (CLAHE, resize, normalize)
    dan mengembalikan prediksi model.
    """
    if model is None:
        print("Model tidak dimuat, prediksi dibatalkan.")
        return None

    try:
        # 1. Baca gambar menggunakan cv2
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Gagal membaca gambar dari {img_path}")
            return None
        
        # 2. Terapkan CLAHE
        img = apply_clahe_to_color_image(img)
        
        # 3. Resize ke (224, 224)
        img_resized = cv2.resize(img, (224, 224)) 
        
        # 4. Normalisasi
        x = img_resized.astype('float32') / 255.0 
        
        # 5. Tambahkan dimensi batch (1, 224, 224, 3)
        x = np.expand_dims(x, axis=0) 

        # 6. Lakukan prediksi
        preds = model.predict(x)
        return preds
        
    except Exception as e:
        print(f"Error saat prediksi: {e}")
        return None

# ----------------------------------------------------------------------------
# 7. Route Flask
# ----------------------------------------------------------------------------

# Route untuk halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Route untuk halaman 'About'
@app.route('/about')
def about():
    return render_template('about.html')

# Route untuk prediksi
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if model is None:
            return render_template('predict.html', prediction_text="Error: Model tidak berhasil dimuat. Periksa log server.", image_path=None)

        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            preds = model_predict(file_path, model)
            
            if preds is not None:
                try:
                    print(f"[DEBUG] raw model preds: {preds}")
                    pred_class_index = int(np.argmax(preds[0]))
                    result = CLASS_NAMES[pred_class_index]
                    confidence = float(np.max(preds[0]) * 100)
                    
                    result_text = f"Hasil Prediksi: {result} ({confidence:.2f}%)"
                    print(f"[DEBUG] pred_class_index={pred_class_index}, result_text={result_text}")

                    # Dapatkan penjelasan dari LLM
                    try:
                        explanation = get_explanation(result, confidence)
                    except Exception as e:
                        explanation = "Maaf, tidak dapat menghasilkan penjelasan saat ini."
                        print(f"[DEBUG] get_explanation failed: {e}")
                except Exception as e:
                    print(f"[ERROR] Kesalahan saat memproses prediksi: {e}")
                    return render_template('predict.html', prediction_text="Error saat memproses prediksi.", image_path=None)
                
                # --- PERBAIKAN DI SINI ---
                # Gunakan forward slash (/) untuk URL, bukan os.path.join
                image_url = f'uploads/{filename}'
                # -------------------------
                
                return render_template('predict.html', 
                                     prediction_text=result_text, 
                                     explanation_text=explanation,
                                     image_path=image_url)
            else:
                return render_template('predict.html', prediction_text="Error saat melakukan prediksi.", image_path=None)

    # Jika method GET
    return render_template('predict.html', prediction_text=None, image_path=None)

# ----------------------------------------------------------------------------
# 8. Jalankan Aplikasi
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)