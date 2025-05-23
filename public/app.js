class DogCatClassifier {
    constructor() {
        this.model = null;
        this.classLabels = ['cats', 'dogs'];
        this.isModelLoaded = false;
        this.initializeElements();
        this.setupEventListeners();
        this.loadModel();
    }

    initializeElements() {
        this.uploadArea = document.getElementById('uploadArea');
        this.imageInput = document.getElementById('imageInput');
        this.selectBtn = document.getElementById('selectBtn');
        this.previewSection = document.getElementById('previewSection');
        this.previewImage = document.getElementById('previewImage');
        this.predictBtn = document.getElementById('predictBtn');
        this.resultSection = document.getElementById('resultSection');
        this.animalIcon = document.getElementById('animalIcon');
        this.predictionText = document.getElementById('predictionText');
        this.confidenceBar = document.getElementById('confidenceBar');
        this.confidenceText = document.getElementById('confidenceText');
        this.loading = document.getElementById('loading');
    }

    setupEventListeners() {
        // Click để chọn file
        this.selectBtn.addEventListener('click', () => {
            this.imageInput.click();
        });

        this.uploadArea.addEventListener('click', () => {
            this.imageInput.click();
        });

        // Xử lý file được chọn
        this.imageInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleImageSelect(e.target.files[0]);
            }
        });

        // Drag and drop
        this.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadArea.classList.add('dragover');
        });

        this.uploadArea.addEventListener('dragleave', () => {
            this.uploadArea.classList.remove('dragover');
        });

        this.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadArea.classList.remove('dragover');

            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type.startsWith('image/')) {
                this.handleImageSelect(files[0]);
            }
        });

        // Nút predict
        this.predictBtn.addEventListener('click', () => {
            this.predictImage();
        });
    }

    async loadModel() {
        try {
            this.showLoading(true);
            console.log('Đang tải mô hình...');

            // Tải model từ thư mục models

            this.model = await tf.loadLayersModel('./models/model.json');
            console.log('Mô hình đã được tải thành công!');

            this.isModelLoaded = true;
            this.showLoading(false);

        } catch (error) {
            console.error('Lỗi khi tải mô hình:', error);
            alert('Không thể tải mô hình. Vui lòng kiểm tra lại file mô hình.');
            this.showLoading(false);
        }
    }

    handleImageSelect(file) {
        if (!file.type.startsWith('image/')) {
            alert('Vui lòng chọn file hình ảnh!');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            this.previewImage.src = e.target.result;
            this.previewSection.style.display = 'block';
            this.resultSection.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    async predictImage() {
        if (!this.isModelLoaded) {
            alert('Mô hình chưa được tải. Vui lòng đợi...');
            return;
        }

        try {
            this.showLoading(true);

            // Tiền xử lý ảnh
            const tensor = await this.preprocessImage(this.previewImage);

            // Dự đoán
            const predictions = await this.model.predict(tensor).data();

            // Xử lý kết quả
            const result = this.processResults(predictions);

            // Hiển thị kết quả
            this.displayResult(result);

            // Cleanup
            tensor.dispose();

        } catch (error) {
            console.error('Lỗi khi dự đoán:', error);
            alert('Có lỗi xảy ra khi phân loại ảnh. Vui lòng thử lại.');
        } finally {
            this.showLoading(false);
        }
    }

    async preprocessImage(imageElement) {
        return tf.tidy(() => {
            // Convert image to tensor
            const tensor = tf.browser.fromPixels(imageElement);

            // Resize to model input size (128x128)
            const resized = tf.image.resizeBilinear(tensor, [128, 128]);

            // Normalize pixel values to [0, 1]
            const normalized = resized.div(255.0);

            // Add batch dimension
            const batched = normalized.expandDims(0);

            return batched;
        });
    }

    processResults(predictions) {
        const catConfidence = predictions[0];
        const dogConfidence = predictions[1];

        const maxConfidence = Math.max(catConfidence, dogConfidence);
        const predictedClass = catConfidence > dogConfidence ? 'mèo' : 'chó';
        const icon = catConfidence > dogConfidence ? '🐱' : '🐶';

        return {
            class: predictedClass,
            confidence: maxConfidence,
            icon: icon
        };
    }

    displayResult(result) {
        this.animalIcon.textContent = result.icon;
        this.predictionText.textContent = `Đây là ${result.class}!`;

        const confidencePercent = Math.round(result.confidence * 100);
        this.confidenceBar.style.width = `${confidencePercent}%`;
        this.confidenceText.textContent = `${confidencePercent}%`;

        this.resultSection.style.display = 'block';

        // Scroll to result
        this.resultSection.scrollIntoView({ behavior: 'smooth' });
    }

    showLoading(show) {
        this.loading.style.display = show ? 'block' : 'none';
    }
}

// Khởi tạo ứng dụng khi DOM đã sẵn sàng
document.addEventListener('DOMContentLoaded', () => {
    new DogCatClassifier();
});