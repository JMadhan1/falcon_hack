document.addEventListener('DOMContentLoaded', () => {
    // Elements
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

    // Tabs
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    // Camera Elements
    const videoFeed = document.getElementById('video-feed');
    const cameraCanvas = document.getElementById('camera-canvas');
    const startCameraBtn = document.getElementById('start-camera-btn');
    const captureBtn = document.getElementById('capture-btn');
    const capturedPreview = document.getElementById('captured-preview');
    const capturedImage = document.getElementById('captured-image');
    const retakeBtn = document.getElementById('retake-btn');

    // Modals
    const falconInfoBtn = document.getElementById('falcon-info-btn');
    const falconModal = document.getElementById('falcon-modal');
    const reportBtn = document.getElementById('report-btn');
    const feedbackModal = document.getElementById('feedback-modal');
    const feedbackForm = document.getElementById('feedback-form');
    const closeModals = document.querySelectorAll('.close-modal');

    let currentFile = null;
    let currentImageId = null;
    let stream = null;

    // Tab Switching
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Deactivate all
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            // Activate clicked
            btn.classList.add('active');
            document.getElementById(`${btn.dataset.tab}-tab`).classList.add('active');

            // Stop camera if switching away
            if (btn.dataset.tab !== 'camera' && stream) {
                stopCamera();
            }
        });
    });

    // --- Upload Logic ---
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
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) handleFile(e.target.files[0]);
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
            resetResults();
        };
        reader.readAsDataURL(file);
    }

    function resetUpload() {
        currentFile = null;
        fileInput.value = '';
        previewContainer.style.display = 'none';
        detectBtn.disabled = true;
        resetResults();
    }

    function resetResults() {
        resultsContent.style.display = 'none';
        emptyState.style.display = 'block';
        currentImageId = null;
    }

    // --- Camera Logic ---
    startCameraBtn.addEventListener('click', async () => {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            videoFeed.srcObject = stream;
            startCameraBtn.style.display = 'none';
            captureBtn.disabled = false;
        } catch (err) {
            console.error("Error accessing camera:", err);
            alert("Could not access camera. Please ensure you have granted permission.");
        }
    });

    captureBtn.addEventListener('click', () => {
        if (!stream) return;

        // Set canvas dimensions to match video
        cameraCanvas.width = videoFeed.videoWidth;
        cameraCanvas.height = videoFeed.videoHeight;

        // Draw frame
        const ctx = cameraCanvas.getContext('2d');
        ctx.drawImage(videoFeed, 0, 0, cameraCanvas.width, cameraCanvas.height);

        // Convert to blob/file
        cameraCanvas.toBlob((blob) => {
            const file = new File([blob], "webcam_capture.jpg", { type: "image/jpeg" });
            currentFile = file;

            // Show preview
            capturedImage.src = URL.createObjectURL(blob);
            capturedPreview.style.display = 'block';

            // Enable detection
            detectBtn.disabled = false;
            resetResults();
        }, 'image/jpeg');
    });

    retakeBtn.addEventListener('click', () => {
        capturedPreview.style.display = 'none';
        currentFile = null;
        detectBtn.disabled = true;
        resetResults();
    });

    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
            videoFeed.srcObject = null;
            startCameraBtn.style.display = 'flex';
            captureBtn.disabled = true;
            capturedPreview.style.display = 'none';
        }
    }

    // --- Detection Logic ---
    detectBtn.addEventListener('click', async () => {
        if (!currentFile) return;

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

            if (data.error) throw new Error(data.error);

            currentImageId = data.image_id;
            resultImage.src = `data:image/jpeg;base64,${data.image}`;

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

            emptyState.style.display = 'none';
            resultsContent.style.display = 'flex';

            // Scroll to results
            document.getElementById('results-section').scrollIntoView({ behavior: 'smooth' });

        } catch (error) {
            console.error('Error:', error);
            alert('Detection failed: ' + error.message);
        } finally {
            loader.style.display = 'none';
            detectBtn.disabled = false;
        }
    });

    // --- Modal Logic ---
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

    // Feedback Form
    feedbackForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const correctClass = document.getElementById('feedback-class').value;
        const notes = document.getElementById('feedback-notes').value;

        try {
            const response = await fetch('/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
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
