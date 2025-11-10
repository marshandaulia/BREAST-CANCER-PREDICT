// File Upload Handler
class FileUploadHandler {
    constructor() {
        this.elements = {
            fileInput: document.getElementById('file'),
            imagePreview: document.getElementById('image-preview'),
            dropZone: document.getElementById('drop-zone'),
            previewContainer: document.getElementById('preview-container'),
            fileName: document.querySelector('.file-name'),
            removeButton: document.getElementById('remove-file'),
            predictButton: document.getElementById('predict-button'),
            form: document.getElementById('predict-form')
        };
        
        this.initialize();
        this.setupEventListeners();
    }

    initialize() {
        if (this.elements.predictButton) {
            this.elements.predictButton.disabled = true;
        }
        if (this.elements.previewContainer) {
            this.elements.previewContainer.classList.add('hidden');
        }
    }

    handleFile(file) {
        if (!file) return;
        
        if (['image/jpeg', 'image/png'].includes(file.type)) {
            const reader = new FileReader();
            
            reader.onload = (e) => {
                this.updatePreview(e.target.result, file.name);
            };
            
            reader.onerror = () => {
                alert('Error membaca file. Silakan coba lagi.');
                this.resetUpload();
            };
            
            reader.readAsDataURL(file);
        } else {
            alert('Mohon unggah file gambar dengan format PNG atau JPG.');
            this.resetUpload();
        }
    }

    updatePreview(imageUrl, fileName) {
        const { imagePreview, previewContainer, fileName: fileNameElement, 
                predictButton, dropZone } = this.elements;

        if (imagePreview) {
            imagePreview.src = imageUrl;
            imagePreview.style.display = 'block';
        }
        
        if (previewContainer) {
            previewContainer.classList.remove('hidden');
        }
        
        if (fileNameElement) {
            fileNameElement.textContent = fileName;
        }
        
        if (predictButton) {
            predictButton.disabled = false;
        }
        
        if (dropZone) {
            dropZone.style.display = 'none';
        }
    }

    resetUpload() {
        const { fileInput, previewContainer, dropZone, predictButton, 
                fileName, imagePreview } = this.elements;

        if (fileInput) fileInput.value = '';
        if (previewContainer) previewContainer.classList.add('hidden');
        if (dropZone) dropZone.style.display = 'block';
        if (predictButton) predictButton.disabled = true;
        if (fileName) fileName.textContent = 'Tidak ada file dipilih';
        if (imagePreview) {
            imagePreview.src = '#';
            imagePreview.style.display = 'none';
        }
    }

    setupEventListeners() {
        const { fileInput, dropZone, removeButton, form } = this.elements;

        if (fileInput) {
            fileInput.addEventListener('change', (e) => {
                this.handleFile(e.target.files[0]);
            });
        }

        if (dropZone) {
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropZone.addEventListener(eventName, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                });
            });

            ['dragenter', 'dragover'].forEach(eventName => {
                dropZone.addEventListener(eventName, () => {
                    dropZone.classList.add('drag-over');
                });
            });

            ['dragleave', 'drop'].forEach(eventName => {
                dropZone.addEventListener(eventName, () => {
                    dropZone.classList.remove('drag-over');
                });
            });

            dropZone.addEventListener('drop', (e) => {
                this.handleFile(e.dataTransfer.files[0]);
            });
        }

        if (removeButton) {
            removeButton.addEventListener('click', () => this.resetUpload());
        }

        if (form) {
            form.addEventListener('submit', () => {
                const button = this.elements.predictButton;
                if (button) {
                    button.classList.add('loading');
                    const buttonText = button.querySelector('.button-text');
                    if (buttonText) {
                        buttonText.style.opacity = '0';
                    }
                    button.disabled = true;
                }
            });
        }
    }
}

// Initialize the handler when the DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new FileUploadHandler();
});