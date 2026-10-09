const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const dropContent = document.getElementById('dropContent');
const preview = document.getElementById('preview');
const submitBtn = document.getElementById('submitBtn');
const statusMsg = document.getElementById('statusMsg');
const form = document.getElementById('uploadForm');

let selectedFile = null;

// Click to browse
dropZone.addEventListener('click', () => fileInput.click());

// File selected via dialog
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// Drag and drop
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
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

function handleFile(file) {
    const validTypes = ['image/jpeg', 'image/png'];
    if (!validTypes.includes(file.type)) {
        statusMsg.textContent = '❌ Please select a JPG or PNG image';
        statusMsg.style.color = '#ef4444';
        return;
    }
    if (file.size > 10 * 1024 * 1024) {
        statusMsg.textContent = '❌ File too large (max 10 MB)';
        statusMsg.style.color = '#ef4444';
        return;
    }

    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        preview.src = e.target.result;
        preview.style.display = 'block';
        dropContent.style.display = 'none';
    };
    reader.readAsDataURL(file);

    submitBtn.disabled = false;
    statusMsg.textContent = '✅ Ready to analyze';
    statusMsg.style.color = '#22c55e';
}

// Submit with loading state
form.addEventListener('submit', (e) => {
    if (!selectedFile) {
        e.preventDefault();
        return;
    }

    submitBtn.disabled = true;
    submitBtn.textContent = 'Analyzing...';

    const overlay = document.createElement('div');
    overlay.className = 'loading-overlay';
    overlay.innerHTML = `
        <div class="spinner"></div>
        <p style="margin-top:16px; color:#94a3b8;">Running YOLOv8 detection...</p>
    `;
    document.body.appendChild(overlay);
});