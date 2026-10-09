const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const fileList = document.getElementById('fileList');
const submitBtn = document.getElementById('submitBtn');
const statusMsg = document.getElementById('statusMsg');
const form = document.getElementById('batchForm');

let selectedFiles = [];

// Click to browse
dropZone.addEventListener('click', () => fileInput.click());

// Drag & drop
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
        handleFiles(e.dataTransfer.files);
    }
});

// File input
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFiles(e.target.files);
    }
});

function handleFiles(files) {
    const validTypes = ['image/jpeg', 'image/png'];
    const valid = [];
    let skipped = 0;

    for (const f of files) {
        if (!validTypes.includes(f.type)) {
            skipped++;
            continue;
        }
        if (f.size > 10 * 1024 * 1024) {
            skipped++;
            continue;
        }
        valid.push(f);
    }

    // Merge with existing, cap at 20
    const combined = [...selectedFiles, ...valid].slice(0, 20);
    selectedFiles = combined;

    if (skipped > 0) {
        statusMsg.textContent = `${skipped} file(s) skipped (wrong type or too large)`;
        statusMsg.style.color = '#ef4444';
    }

    renderFileList();
    updateSubmitState();
}

function renderFileList() {
    fileList.innerHTML = '';
    selectedFiles.forEach((f, idx) => {
        const div = document.createElement('div');
        div.className = 'file-item';
        div.innerHTML = `
            <span class="file-name">${f.name}</span>
            <span class="file-size">${(f.size / 1024).toFixed(0)} KB</span>
            <button type="button" class="file-remove" data-idx="${idx}">×</button>
        `;
        fileList.appendChild(div);
    });

    fileList.querySelectorAll('.file-remove').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const idx = parseInt(e.target.getAttribute('data-idx'));
            selectedFiles.splice(idx, 1);
            renderFileList();
            updateSubmitState();
        });
    });

    // Sync file input with selectedFiles
    const dt = new DataTransfer();
    selectedFiles.forEach(f => dt.items.add(f));
    fileInput.files = dt.files;
}

function updateSubmitState() {
    if (selectedFiles.length >= 2) {
        submitBtn.disabled = false;
        statusMsg.textContent = `${selectedFiles.length} file(s) ready — click Analyze All`;
        statusMsg.style.color = '#22c55e';
    } else if (selectedFiles.length === 1) {
        submitBtn.disabled = true;
        statusMsg.textContent = 'Add at least 2 images for batch mode (or use single upload)';
        statusMsg.style.color = '#f59e0b';
    } else {
        submitBtn.disabled = true;
        statusMsg.textContent = 'Select at least 2 images to begin';
        statusMsg.style.color = '#94a3b8';
    }
}

// Loading overlay
form.addEventListener('submit', (e) => {
    if (selectedFiles.length < 2) {
        e.preventDefault();
        return;
    }
    submitBtn.disabled = true;
    submitBtn.textContent = `Analyzing ${selectedFiles.length} images...`;

    const overlay = document.createElement('div');
    overlay.className = 'loading-overlay';
    overlay.innerHTML = `
        <div class="spinner"></div>
        <p style="margin-top:16px; color:#94a3b8;">
            Running YOLOv8 on ${selectedFiles.length} images...<br>
            <span style="font-size:0.85rem;">This may take a minute</span>
        </p>
    `;
    document.body.appendChild(overlay);
});