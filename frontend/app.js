const API = 'http://localhost:5000';

// ── Elements ──────────────────────────────────────────
const uploadArea   = document.getElementById('uploadArea');
const fileInput    = document.getElementById('fileInput');
const filePreview  = document.getElementById('filePreview');
const fileNameEl   = document.getElementById('fileName');
const fileSizeEl   = document.getElementById('fileSize');
const removeFileBtn= document.getElementById('removeFile');
const processBtn   = document.getElementById('processBtn');

const uploadCard   = document.getElementById('uploadCard');
const progressCard = document.getElementById('progressCard');
const doneCard     = document.getElementById('doneCard');
const errorCard    = document.getElementById('errorCard');

const progressBar  = document.getElementById('progressBar');
const progressText = document.getElementById('progressText');
const downloadBtn  = document.getElementById('downloadBtn');
const anotherBtn   = document.getElementById('anotherBtn');
const tryAgainBtn  = document.getElementById('tryAgainBtn');
const errorText    = document.getElementById('errorText');

// ── State ─────────────────────────────────────────────
let currentFile = null;
let currentJobId = null;
let pollInterval = null;

// ── Upload interactions ───────────────────────────────
uploadArea.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', e => e.target.files[0] && setFile(e.target.files[0]));

uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('dragover'); });
uploadArea.addEventListener('dragleave', ()  => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', e => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
});

removeFileBtn.addEventListener('click', clearFile);
processBtn.addEventListener('click', startProcessing);
downloadBtn.addEventListener('click', downloadResult);
anotherBtn.addEventListener('click', reset);
tryAgainBtn.addEventListener('click', reset);

// ── File handling ─────────────────────────────────────
function setFile(file) {
    const allowed = ['mp3','wav','flac','ogg','m4a','aac'];
    const ext = file.name.split('.').pop().toLowerCase();

    if (!allowed.includes(ext)) {
        showError('File type not supported. Please upload an audio file.');
        return;
    }
    if (file.size > 100 * 1024 * 1024) {
        showError('File is too large. Maximum size is 100 MB.');
        return;
    }

    currentFile = file;
    fileNameEl.textContent = file.name;
    fileSizeEl.textContent = fmtSize(file.size);
    uploadArea.style.display = 'none';
    filePreview.style.display = 'flex';
    processBtn.style.display = 'block';
}

function clearFile() {
    currentFile = null;
    fileInput.value = '';
    uploadArea.style.display = '';
    filePreview.style.display = 'none';
    processBtn.style.display = 'none';
}

function fmtSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(1) + ' MB';
}

// ── Processing ────────────────────────────────────────
async function startProcessing() {
    if (!currentFile) return;

    showCard('progress');
    setProgress(0);

    const form = new FormData();
    form.append('file', currentFile);

    try {
        const res = await fetch(`${API}/upload`, { method: 'POST', body: form });
        const data = await res.json();

        if (!res.ok) throw new Error(data.detail || data.error || 'Upload failed');

        currentJobId = data.job_id;
        pollStatus();

    } catch (err) {
        showError(err.message);
    }
}

function pollStatus() {
    if (pollInterval) clearInterval(pollInterval);

    pollInterval = setInterval(async () => {
        try {
            const res  = await fetch(`${API}/status/${currentJobId}`);
            const data = await res.json();

            if (!res.ok) throw new Error(data.detail || 'Status check failed');

            setProgress(data.progress || 0);

            if (data.status === 'completed') {
                clearInterval(pollInterval);
                setProgress(100);
                setTimeout(() => showCard('done'), 400);
            } else if (data.status === 'error') {
                clearInterval(pollInterval);
                showError(data.error_message || 'Processing failed');
            }
        } catch (err) {
            clearInterval(pollInterval);
            showError(err.message);
        }
    }, 1000);
}

function setProgress(pct) {
    progressBar.style.width = `${pct}%`;
    progressText.textContent = `${Math.round(pct)}%`;
}

// ── Download ──────────────────────────────────────────
async function downloadResult() {
    try {
        const res = await fetch(`${API}/download/${currentJobId}`);
        if (!res.ok) {
            const d = await res.json().catch(() => ({}));
            throw new Error(d.detail || 'Download failed');
        }
        const blob = await res.blob();
        const url  = URL.createObjectURL(blob);
        const a    = document.createElement('a');
        a.href     = url;
        a.download = `vocals_${currentFile.name.replace(/\.[^.]+$/, '')}.wav`;
        document.body.appendChild(a);
        a.click();
        URL.revokeObjectURL(url);
        document.body.removeChild(a);
    } catch (err) {
        showError(err.message);
    }
}

// ── UI helpers ────────────────────────────────────────
function showCard(name) {
    uploadCard.style.display   = name === 'upload'   ? '' : 'none';
    progressCard.style.display = name === 'progress' ? '' : 'none';
    doneCard.style.display     = name === 'done'     ? '' : 'none';
    errorCard.style.display    = name === 'error'    ? '' : 'none';
}

function showError(msg) {
    errorText.textContent = msg;
    showCard('error');
}

function reset() {
    currentFile  = null;
    currentJobId = null;
    if (pollInterval) clearInterval(pollInterval);
    fileInput.value = '';
    clearFile();
    showCard('upload');
}
