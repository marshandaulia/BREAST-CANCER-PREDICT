// Perbaikan untuk 'back-forward cache' di browser
// Memastikan loader tersembunyi saat menekan tombol "Back"
window.addEventListener('pageshow', function() {
    const loader = document.getElementById('loader');
    if (loader) {
        loader.classList.add('hidden');
        try { loader.style.display = 'none'; } catch (e) {}
        loader.setAttribute('aria-hidden', 'true');
    }
});

// ==========================================================
// HANYA SATU EVENT LISTENER DOMContentLoaded
// ==========================================================
document.addEventListener("DOMContentLoaded", function() {
    
    // --- 1. Logika Upload Gambar ---
    
    // Menggunakan const untuk elemen agar lebih aman
    const fileInput = document.getElementById('file');
    const imagePreview = document.getElementById('image-preview');
    const dropZone = document.getElementById('drop-zone');
    const previewContainer = document.getElementById('preview-container');
    const fileName = document.querySelector('.file-name');
    const removeButton = document.getElementById('remove-file');
    const predictButton = document.getElementById('predict-button');
    const predictForm = document.getElementById('predict-form'); // Diganti dari 'form' menjadi 'predictForm' agar konsisten

    // Inisialisasi keadaan awal
    if (predictButton) {
        predictButton.disabled = true;
    }
    if (previewContainer) {
        previewContainer.classList.add('hidden');
    }

    // Fungsi untuk menangani file
    function handleFile(file) {
        if (!file) return;
        
        if (['image/jpeg', 'image/png'].includes(file.type)) {
            const reader = new FileReader();
            
            reader.onload = function(e) {
                // Periksa apakah elemen ada sebelum menggunakannya
                if (imagePreview) {
                    imagePreview.src = e.target.result;
                    imagePreview.style.display = 'block';
                }
                if (previewContainer) previewContainer.classList.remove('hidden');
                if (fileName) fileName.textContent = file.name;
                if (predictButton) predictButton.disabled = false;
                if (dropZone) dropZone.style.display = 'none';
            };
            
            reader.onerror = function() {
                alert('Error membaca file. Silakan coba lagi.');
            };
            
            reader.readAsDataURL(file);
        } else {
            alert('Mohon unggah file gambar dengan format PNG atau JPG.');
            resetUpload();
        }
    }

    // Fungsi untuk mereset upload
    function resetUpload() {
        if (fileInput) fileInput.value = '';
        if (previewContainer) previewContainer.classList.add('hidden');
        if (dropZone) dropZone.style.display = 'block';
        if (predictButton) predictButton.disabled = true;
        if (fileName) fileName.textContent = 'Tidak ada file dipilih';
    }

    // Event Listeners untuk Upload
    if (dropZone && fileInput) {
        // File Input Change
        fileInput.addEventListener('change', function(e) {
            handleFile(e.target.files[0]);
        });

        // Drag & Drop Events
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, function(e) {
                e.preventDefault();
                e.stopPropagation();
            });
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, function() {
                dropZone.classList.add('drag-over');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, function() {
                dropZone.classList.remove('drag-over');
            });
        });

        dropZone.addEventListener('drop', function(e) {
            handleFile(e.dataTransfer.files[0]);
        });

        // Remove Button
        if (removeButton) {
            removeButton.addEventListener('click', resetUpload);
        }
    }

    // --- 2. Logika untuk Menampilkan Loading Spinner Halaman Penuh ---
    const loader = document.getElementById('loader');

    // Cek apakah kita di halaman yang benar
    if (predictForm && loader) {
        predictForm.addEventListener('submit', function() {
            // Tampilkan loader saat form disubmit
            loader.classList.remove('hidden');
            try { loader.style.display = 'flex'; } catch (e) {}
            loader.setAttribute('aria-hidden', 'false');

            // Logika loading untuk tombol "Image Prediction" (jika ada)
            if (predictButton) {
                predictButton.classList.add('loading');
                const buttonText = predictButton.querySelector('.button-text');
                if (buttonText) {
                    buttonText.style.opacity = '0';
                }
                predictButton.disabled = true;
            }
        });
    }

    // --- 3. Logika Chat Follow-up ---
    const chatForm = document.getElementById('chat-form');
    const chatContainer = document.getElementById('chat-container');
    const predictionData = document.getElementById('prediction-data');

    // Chat helpers:
    function createChatItem(html, who='bot') {
        const wrapper = document.createElement('div');
        wrapper.className = 'chat-item ' + (who === 'user' ? 'chat-user' : 'chat-bot');
        const inner = document.createElement('div');
        inner.className = 'chat-text';
        inner.innerHTML = html;
        wrapper.appendChild(inner);
        return wrapper;
    }

    // ==========================================================
    // PERBAIKAN: Mengganti total logika submit chat
    // ==========================================================
    if (chatForm && chatContainer) {
        chatForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            const textarea = chatForm.querySelector('textarea[name="message"]');
            const msg = textarea.value.trim();
            if (!msg) return;
    
            // Ambil tombol kirim
            const chatButton = chatForm.querySelector('button[type="submit"]');
    
            // Buat dan tampilkan pesan pengguna
            const userItem = createChatItem(msg, 'user');
            chatContainer.appendChild(userItem);
            textarea.value = '';
            
            // Nonaktifkan tombol dan tambahkan animasi "Memproses..."
            chatButton.disabled = true;
            chatButton.innerHTML = 'Memproses'; // HANYA TEKS, TIDAK ADA SPINNER
            chatButton.classList.add('loading-dots'); // TAMBAHKAN KELAS ANIMASI DOT
    
            // Buat placeholder 'Mengirim...' di dalam kotak chat
            const loading = createChatItem('<em>Mengirim...</em>', 'bot');
            loading.classList.add('loading');
            userItem.insertAdjacentElement('afterend', loading);
            chatContainer.scrollTop = chatContainer.scrollHeight;
    
            try {
                const prediction = predictionData ? predictionData.getAttribute('data-pred') || '' : '';
                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: msg, prediction: prediction })
                });
    
                const data = await res.json();
                // Ganti placeholder 'Mengirim...' dengan jawaban bot
                const botItem = data && data.reply ? createChatItem(data.reply, 'bot') : createChatItem('<div class="text-muted">Tidak ada respons.</div>', 'bot');
                loading.replaceWith(botItem);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            } catch (err) {
                // Ganti placeholder 'Mengirim...' dengan pesan error
                const errItem = createChatItem('<div class="text-danger">Terjadi kesalahan saat mengirim pesan.</div>', 'bot');
                loading.replaceWith(errItem);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            } finally {
                // Apapun hasilnya, kembalikan tombol ke normal
                chatButton.disabled = false;
                chatButton.innerHTML = 'Kirim';
                chatButton.classList.remove('loading-dots'); // HAPUS KELAS ANIMASI DOT
            }
        });
    }
    // ==========================================================
    // AKHIR PERBAIKAN CHAT
    // ==========================================================

});