const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const uploadBtn = document.getElementById('uploadBtn');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const errorMessage = document.getElementById('errorMessage');

let selectedFile = null;
let currentJobId = null;

// Click to upload
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragging');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragging');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragging');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

// File input change
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

function handleFileSelect(file) {
    const validTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska'];
    
    if (!validTypes.includes(file.type)) {
        showError('Please select a valid video file (MP4, AVI, MOV, MKV)');
        return;
    }

    if (file.size > 100 * 1024 * 1024) {
        showError('File size exceeds 100MB limit');
        return;
    }

    selectedFile = file;
    fileName.textContent = file.name;
    fileInfo.classList.add('active');
    uploadBtn.disabled = false;
    errorMessage.classList.remove('active');
}

// Upload and analyze
uploadBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('video', selectedFile);

    loading.classList.add('active');
    uploadBtn.disabled = true;
    results.classList.remove('active');

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Upload failed');
        }

        currentJobId = data.job_id;
        displayResults(data);
        
    } catch (error) {
        showError(error.message);
        uploadBtn.disabled = false;
    } finally {
        loading.classList.remove('active');
    }
});

function displayResults(data) {
    const summary = data.summary;
    const trackingData = data.tracking_data;

    // Update stats
    document.getElementById('totalUniqueObjects').textContent = summary.total_unique_objects;
    document.getElementById('uniqueClasses').textContent = Object.keys(summary.object_counts).length;

    // Display objects by class
    const objectsByClass = document.getElementById('objectsByClass');
    objectsByClass.innerHTML = '';
    
    for (const [className, objects] of Object.entries(summary.objects_by_class)) {
        const classSection = document.createElement('div');
        classSection.className = 'class-section';
        
        const classHeader = document.createElement('div');
        classHeader.className = 'class-header';
        classHeader.innerHTML = `
            <h4>${className}</h4>
            <span class="count-badge">${objects.length} tracked</span>
        `;
        classSection.appendChild(classHeader);
        objectsByClass.appendChild(classSection);
    }

    // Fill tracking table
    const trackingTable = document.getElementById('trackingTable');
    trackingTable.innerHTML = '';
    
    // Sort by track ID
    const sortedEntries = Object.entries(trackingData).sort((a, b) => {
        return parseInt(a[0]) - parseInt(b[0]);
    });
    
    for (const [trackId, info] of sortedEntries) {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td><strong>#${trackId}</strong></td>
            <td><span class="class-badge">${info.class}</span></td>
            <td>${info.first_seen}</td>
        `;
        trackingTable.appendChild(row);
    }

    // Display JSON preview
    const jsonPreview = document.getElementById('jsonPreview');
    jsonPreview.textContent = JSON.stringify(trackingData, null, 2);

    results.classList.add('active');
}

// Download buttons
document.getElementById('downloadJson').addEventListener('click', () => {
    if (currentJobId) {
        window.location.href = `/api/download/json/${currentJobId}`;
    }
});

document.getElementById('analyzeAnother').addEventListener('click', () => {
    selectedFile = null;
    currentJobId = null;
    fileInfo.classList.remove('active');
    fileInput.value = '';
    results.classList.remove('active');
    uploadBtn.disabled = true;
});

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.add('active');
}