// Frontend JavaScript for DyLuMo

let selectedFile = null;

// DOM elements
const imageInput = document.getElementById('imageInput');
const uploadSection = document.getElementById('uploadSection');
const previewSection = document.getElementById('previewSection');
const imagePreview = document.getElementById('imagePreview');
const recommendBtn = document.getElementById('recommendBtn');
const loading = document.getElementById('loading');
const error = document.getElementById('error');
const resultsSection = document.getElementById('resultsSection');
const emotionResult = document.getElementById('emotionResult');
const songsGrid = document.getElementById('songsGrid');
const inferenceTime = document.getElementById('inferenceTime');

// File input change event
imageInput.addEventListener('change', handleFileSelect);

// Drag and drop
uploadSection.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadSection.classList.add('drag-over');
});

uploadSection.addEventListener('dragleave', () => {
    uploadSection.classList.remove('drag-over');
});

uploadSection.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadSection.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

// Recommend button click
recommendBtn.addEventListener('click', getRecommendations);

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please select an image file');
        return;
    }
    
    // Validate file size (10 MB)
    if (file.size > 10 * 1024 * 1024) {
        showError('File too large. Maximum size is 10 MB');
        return;
    }
    
    selectedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        previewSection.style.display = 'block';
        resultsSection.style.display = 'none';
        hideError();
    };
    reader.readAsDataURL(file);
}

async function getRecommendations() {
    if (!selectedFile) {
        showError('Please select an image first');
        return;
    }
    
    // Prepare form data
    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('top_k', '10');
    
    // Show loading
    loading.style.display = 'block';
    recommendBtn.disabled = true;
    resultsSection.style.display = 'none';
    hideError();
    
    try {
        // Call API
        const response = await fetch('/recommend', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            displayResults(data);
        } else {
            showError(data.message || 'Failed to get recommendations');
        }
        
    } catch (err) {
        console.error('Error:', err);
        showError('Failed to connect to server. Please try again.');
    } finally {
        loading.style.display = 'none';
        recommendBtn.disabled = false;
    }
}

function displayResults(data) {
    // Display emotion
    const emotion = data.emotion;
    document.getElementById('emotionValue').textContent = emotion.predicted_emotion;
    document.getElementById('emotionConfidence').textContent = 
        `${(emotion.confidence * 100).toFixed(1)}% confidence`;
    
    // Display songs
    songsGrid.innerHTML = '';
    data.recommendations.forEach((song) => {
        const songCard = document.createElement('div');
        songCard.className = 'song-card';
        songCard.innerHTML = `
            <div class="song-rank">${song.rank}</div>
            <div class="song-info">
                <div class="song-title">${song.track_name}</div>
                <div class="song-artist">${song.artist_name}</div>
                <div class="song-meta">
                    <div class="meta-item">
                        <span>Emotion:</span>
                        <span class="meta-value">${song.emotion}</span>
                    </div>
                    <div class="meta-item">
                        <span>Match:</span>
                        <span class="meta-value">${(song.similarity_score * 100).toFixed(1)}%</span>
                    </div>
                </div>
            </div>
        `;
        songsGrid.appendChild(songCard);
    });
    
    // Display inference time
    inferenceTime.textContent = `Processed in ${data.inference_time}`;
    
    // Show results
    resultsSection.style.display = 'block';
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function showError(message) {
    error.textContent = 'Error: ' + message;
    error.style.display = 'block';
}

function hideError() {
    error.style.display = 'none';
}

// Check server health on load
async function checkHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        console.log('Server health:', data);
        
        if (!data.model_loaded) {
            showError('Server is starting up. Please wait a moment and refresh...');
        }
    } catch (err) {
        console.error('Server not responding:', err);
        showError('Cannot connect to server. Please make sure the backend is running.');
    }
}

// Check health on page load
checkHealth();

