document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const previewImage = document.getElementById('preview-image');
    const removeBtn = document.getElementById('remove-btn');
    const detectBtn = document.getElementById('detect-btn');
    const loader = document.getElementById('loader');
    const resultsContent = document.getElementById('results-content');
    const emptyState = document.getElementById('empty-state');
    const resultImage = document.getElementById('result-image');
    const statsList = document.getElementById('stats-list');

    // New Elements
    const falconInfoBtn = document.getElementById('falcon-info-btn');
    const falconModal = document.getElementById('falcon-modal');
    const reportBtn = document.getElementById('report-btn');
    const feedbackModal = document.getElementById('feedback-modal');
    const feedbackForm = document.getElementById('feedback-form');
    const closeModals = document.querySelectorAll('.close-modal');

    let currentFile = null;
    let currentImageId = null;

    // Modal Handlers
    falconInfoBtn.addEventListener('click', () => falconModal.style.display = 'block');
    reportBtn.addEventListener('click', () => feedbackModal.style.display = 'block');

    closeModals.forEach(btn => {
        btn.addEventListener('click', () => {
            falconModal.style.display = 'none';
            feedbackModal.style.display = 'none';
        });
    });

    window.addEventListener('click', (e) => {
        if (e.target == falconModal) falconModal.style.display = 'none';
        if (e.target == feedbackModal) feedbackModal.style.display = 'none';
    });

    // Drag & Drop Handlers
    dropZone.addEventListener('click', (e) => {
        if (e.target !== removeBtn && e.target.parentElement !== removeBtn) {
            fileInput.click();
        }
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');

        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file');
            return;
        }

        currentFile = file;
        const reader = new FileReader();

        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewContainer.style.display = 'block';
            detectBtn.disabled = false;

            // Reset results
            resultsContent.style.display = 'none';
            emptyState.style.display = 'block';
        };

        reader.readAsDataURL(file);
    }

    function resetUpload() {
        currentFile = null;
        currentImageId = null;
        fileInput.value = '';
        previewContainer.style.display = 'none';
        detectBtn.disabled = true;
        resultsContent.style.display = 'none';
        emptyState.style.display = 'block';
    }

    // Detection Handler
    detectBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // Show loader
        loader.style.display = 'flex';
        detectBtn.disabled = true;

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/detect', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            // Store image ID for feedback
            currentImageId = data.image_id;

            // Update Results
            resultImage.src = `data:image/jpeg;base64,${data.image}`;

            // Update Stats
            statsList.innerHTML = '';
            if (data.detections.length === 0) {
                statsList.innerHTML = '<div class="stat-item" style="border-color: #ef4444;">No objects detected</div>';
            } else {
                data.detections.forEach(det => {
                    const div = document.createElement('div');
                    div.className = 'stat-item';
                    div.innerHTML = `
                        <span class="stat-name">${det.class}</span>
                        <span class="stat-conf">Confidence: ${det.confidence}</span>
                    `;
                    statsList.appendChild(div);
                });
            }

            // Show Results
            emptyState.style.display = 'none';
            resultsContent.style.display = 'flex';

        } catch (error) {
            console.error('Error:', error);
            alert('Detection failed: ' + error.message);
        } finally {
            loader.style.display = 'none';
            detectBtn.disabled = false;
        }
    });

    // Feedback Form Handler
    feedbackForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const correctClass = document.getElementById('feedback-class').value;
        const notes = document.getElementById('feedback-notes').value;

        try {
            const response = await fetch('/feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image_id: currentImageId,
                    correct_class: correctClass,
                    notes: notes
                })
            });

            const data = await response.json();

            if (data.success) {
                alert('Thank you! Your feedback has been sent to Falcon for model improvement.');
                feedbackModal.style.display = 'none';
                feedbackForm.reset();
            } else {
                throw new Error(data.error);
            }
        } catch (error) {
            alert('Error sending feedback: ' + error.message);
        }
    });
});
