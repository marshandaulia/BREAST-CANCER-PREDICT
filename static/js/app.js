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

// Menjalankan script setelah halaman (DOM) selesai dimuat
document.addEventListener("DOMContentLoaded", function() {
    const elements = {
        fileInput: document.getElementById('file'),
        imagePreview: document.getElementById('image-preview'),
        dropZone: document.getElementById('drop-zone'),
        previewContainer: document.getElementById('preview-container'),
        fileName: document.querySelector('.file-name'),
        removeButton: document.getElementById('remove-file'),
        predictButton: document.getElementById('predict-button'),
        form: document.getElementById('predict-form')
    };

    // Inisialisasi keadaan awal
    if (elements.predictButton) {
        elements.predictButton.disabled = true;
    }
    if (elements.previewContainer) {
        elements.previewContainer.classList.add('hidden');
    }

    function handleFile(file) {
        if (!file) return;
        
        if (['image/jpeg', 'image/png'].includes(file.type)) {
            const reader = new FileReader();
            
            reader.onload = function(e) {
                // Update preview image
                elements.imagePreview.src = e.target.result;
                elements.imagePreview.style.display = 'block';
                
                // Show preview container
                elements.previewContainer.classList.remove('hidden');
                
                // Update filename
                elements.fileName.textContent = file.name;
                
                // Enable predict button
                elements.predictButton.disabled = false;
                
                // Hide drop zone
                elements.dropZone.style.display = 'none';
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

    function resetUpload() {
        elements.fileInput.value = '';
        elements.previewContainer.classList.add('hidden');
        elements.dropZone.style.display = 'block';
        elements.predictButton.disabled = true;
        elements.fileName.textContent = 'Tidak ada file dipilih';
    }

    if (elements.dropZone && elements.fileInput) {
        // File Input Change
        elements.fileInput.addEventListener('change', function(e) {
            handleFile(e.target.files[0]);
        });

        // Drag & Drop Events
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            elements.dropZone.addEventListener(eventName, function(e) {
                e.preventDefault();
                e.stopPropagation();
            });
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            elements.dropZone.addEventListener(eventName, function() {
                elements.dropZone.classList.add('drag-over');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            elements.dropZone.addEventListener(eventName, function() {
                elements.dropZone.classList.remove('drag-over');
            });
        });

        elements.dropZone.addEventListener('drop', function(e) {
            handleFile(e.dataTransfer.files[0]);
        });

        // Remove Button
        if (elements.removeButton) {
            elements.removeButton.addEventListener('click', resetUpload);
        }

        // Form Submit
        if (elements.form) {
            elements.form.addEventListener('submit', function() {
                elements.predictButton.classList.add('loading');
                elements.predictButton.querySelector('.button-text').style.opacity = '0';
                elements.predictButton.disabled = true;
            });
        }
    }

    function handleFile(file) {
        if (file && ['image/jpeg', 'image/png'].includes(file.type)) {
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                previewContainer.classList.remove('hidden');
                fileName.textContent = file.name;
                predictButton.disabled = false;
                dropZone.style.display = 'none';
            }
            reader.readAsDataURL(file);
        } else {
            alert('Mohon unggah file gambar dengan format PNG atau JPG.');
        }
    }

    function resetUpload() {
        fileInput.value = '';
        previewContainer.classList.add('hidden');
        dropZone.style.display = 'block';
        predictButton.disabled = true;
        fileName.textContent = 'Tidak ada file dipilih';
    }

    // Event Listeners
    if (dropZone && fileInput) {
        // File Input Change
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            handleFile(file);
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
            const file = e.dataTransfer.files[0];
            handleFile(file);
        });

        // Remove Button
        if (removeButton) {
            removeButton.addEventListener('click', resetUpload);
        }
    }
    }

    // --- 2. Logika untuk Menampilkan Loading Spinner ---
    const predictForm = document.getElementById('predict-form');
    const loader = document.getElementById('loader');

    // Cek apakah kita di halaman yang benar
    if (predictForm && loader) {
        predictForm.addEventListener('submit', function() {
            // Tampilkan loader saat form disubmit
            loader.classList.remove('hidden');
            try { loader.style.display = 'flex'; } catch (e) {}
            loader.setAttribute('aria-hidden', 'false');
        });
    }
});