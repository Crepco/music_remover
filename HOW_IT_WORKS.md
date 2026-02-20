# 🔬 How It Works - Technical Deep Dive

This document explains the technical architecture, algorithms, and implementation details of the Music Remover application.

## 📐 Architecture Overview

The application follows a client-server architecture with asynchronous processing:

```
┌─────────────┐         HTTP/REST          ┌──────────────┐
│   Browser   │◄───────────────────────────►│  FastAPI     │
│  (Frontend) │                             │  (Backend)   │
└─────────────┘                             └──────┬───────┘
                                                    │
                                                    ▼
                                            ┌──────────────┐
                                            │   Demucs     │
                                            │  AI Model    │
                                            └──────────────┘
```

### Components

1. **Frontend**: Static HTML/CSS/JavaScript - handles UI and user interactions
2. **Backend API**: FastAPI server - manages file uploads, job queuing, and status tracking
3. **Audio Processor**: Demucs integration - performs the actual source separation
4. **Job System**: In-memory job tracking with background processing

## 🎯 Request Flow

### 1. File Upload (`POST /upload`)

```python
# User uploads file via multipart/form-data
file → FastAPI receives UploadFile
     → Validates file type and size
     → Saves to uploads/ directory
     → Creates ProcessingJob object
     → Starts background thread
     → Returns job_id to client
```

**Key Code** (`backend/app.py`):
```python
@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile):
    # Validate file
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400)
    
    # Save file
    contents = await file.read()
    with open(filepath, "wb") as f:
        f.write(contents)
    
    # Create job
    job = ProcessingJob(job_id, filename)
    processing_jobs[job_id] = job
    
    # Start background processing
    background_tasks.add_task(process_audio_background, job_id, filepath)
    
    return {"job_id": job_id, "status": "pending"}
```

### 2. Background Processing

The actual audio processing happens in a background thread to avoid blocking the API:

```python
def process_audio_background(job_id: str, input_file: str):
    job = processing_jobs[job_id]
    job.status = 'processing'
    
    separator = AudioSeparator()
    output_file = separator.process(input_file, progress_callback)
    
    job.status = 'completed'
    job.output_file = output_file
```

### 3. Status Polling (`GET /status/{job_id}`)

The frontend polls the status endpoint every second:

```javascript
setInterval(async () => {
    const response = await fetch(`/status/${job_id}`);
    const data = await response.json();
    
    if (data.status === 'completed') {
        // Show download button
    }
}, 1000);
```

### 4. File Download (`GET /download/{job_id}`)

Once processing completes, the client downloads the result:

```python
@app.get("/download/{job_id}")
async def download_file(job_id: str):
    job = processing_jobs[job_id]
    return FileResponse(job.output_file, media_type='audio/wav')
```

## 🧠 Demucs: The AI Model

### What is Demucs?

**Demucs** (Deep Music Source Separation) is a state-of-the-art deep learning model developed by Meta Research. Unlike traditional signal processing methods, Demucs uses neural networks trained on thousands of songs to understand what vocals sound like vs instruments.

### Model Architecture

The app uses **HTDemucs** (Hybrid Transformer Demucs), which combines:
- **Convolutional layers**: Extract local audio features
- **Transformer layers**: Capture long-range dependencies
- **Hybrid approach**: Best of both architectures

### How Demucs Works

1. **Input Processing**
   ```
   Audio File → librosa.load() → NumPy array → PyTorch tensor
   ```

2. **Spectrogram Analysis**
   - Audio is converted to frequency domain (STFT)
   - Model analyzes spectrogram patterns
   - Identifies which frequencies belong to vocals vs instruments

3. **Source Separation**
   - Model outputs 4 sources: `[drums, bass, other, vocals]`
   - Each source is a separate audio track
   - We extract only the `vocals` source (index 3)

4. **Output Reconstruction**
   - Vocals tensor → NumPy array → WAV file
   - Saved with original sample rate

### Code Flow (`backend/audio_processor/separator.py`)

```python
class AudioSeparator:
    def process(self, input_path, progress_callback=None):
        # 1. Load audio with librosa (handles all formats)
        y, sr = librosa.load(input_path, sr=None, mono=False)
        
        # 2. Ensure stereo format
        if y.ndim == 1:
            y = np.vstack([y, y])  # Mono → Stereo
        
        # 3. Convert to PyTorch tensor
        wav = torch.from_numpy(y).float()  # (channels, samples)
        wav = wav.unsqueeze(0)  # Add batch dim: (1, channels, samples)
        
        # 4. Load Demucs model (lazy loading)
        if self.model is None:
            self.model = get_model('htdemucs')
            self.model.to(self.device)
            self.model.eval()
        
        # 5. Apply model
        with torch.no_grad():
            sources = apply_model(
                self.model, 
                wav, 
                shifts=1,      # Test-time augmentation
                split=True,    # Split long files
                overlap=0.25   # Overlap for seamless joins
            )
        
        # 6. Extract vocals (last source)
        # sources shape: (batch=1, sources=4, channels=2, samples)
        vocals = sources[0, 3].cpu().numpy()  # (channels, samples)
        
        # 7. Save as WAV
        sf.write(output_path, vocals.T, sr, format='WAV')
```

## 🔄 Audio Processing Pipeline

### Step-by-Step Breakdown

#### 1. Audio Loading (`librosa.load()`)

```python
y, sr = librosa.load(input_path, sr=None, mono=False)
```

- **`sr=None`**: Preserves original sample rate (typically 44.1kHz)
- **`mono=False`**: Keeps stereo channels
- **Handles**: MP3, WAV, FLAC, OGG, M4A, AAC automatically
- **Output**: NumPy array shape `(channels, samples)` or `(samples,)` for mono

**Why librosa?**
- Avoids torchcodec dependency issues
- Handles all audio formats reliably
- Well-tested and stable

#### 2. Format Normalization

```python
# Ensure stereo
if y.ndim == 1:
    y = np.vstack([y, y])  # Mono → Stereo
elif y.shape[0] > 2:
    y = y[:2]  # Multi-channel → Stereo
```

Demucs expects stereo input, so we normalize:
- Mono files → duplicate channel to create stereo
- Multi-channel → take first 2 channels
- Already stereo → use as-is

#### 3. Tensor Conversion

```python
wav = torch.from_numpy(y).float()
wav = wav.unsqueeze(0)  # Add batch dimension
```

PyTorch models expect:
- **Batch dimension**: `(batch, channels, samples)`
- **Float32**: Standard precision for neural networks
- **Device placement**: Moved to GPU if available

#### 4. Model Application

```python
sources = apply_model(
    self.model,
    wav,
    shifts=1,      # Test-time augmentation (improves quality)
    split=True,    # Split long files into chunks
    overlap=0.25   # Overlap chunks for seamless reconstruction
)
```

**Parameters Explained:**

- **`shifts=1`**: Applies test-time augmentation
  - Processes audio multiple times with slight shifts
  - Averages results for better quality
  - Increases processing time but improves separation

- **`split=True`**: Handles long files
  - Splits files > 10 seconds into chunks
  - Processes each chunk separately
  - Reconstructs seamlessly with overlap

- **`overlap=0.25`**: Overlap between chunks
  - 25% overlap prevents artifacts at boundaries
  - Uses windowing function for smooth transitions

**Output Shape:**
```
sources: (batch=1, sources=4, channels=2, samples)
         └─┬──┘  └───┬────┘  └───┬────┘  └───┬────┘
           1      drums/bass/    stereo    audio
                  other/vocals              samples
```

#### 5. Vocals Extraction

```python
vocals = sources[0, 3].cpu().numpy()
```

- **`[0]`**: First (and only) batch item
- **`[3]`**: Fourth source = vocals (index 3)
- **`.cpu()`**: Move from GPU to CPU
- **`.numpy()`**: Convert PyTorch tensor to NumPy

**Source Indices:**
- `0`: Drums
- `1`: Bass
- `2`: Other instruments
- `3`: Vocals ← We want this one

#### 6. Output Formatting

```python
if vocals.ndim == 2 and vocals.shape[0] == 2:
    vocals_out = vocals.T  # (2, samples) → (samples, 2)
```

Soundfile expects `(samples, channels)` format:
- Our tensor: `(channels, samples)`
- Need to transpose: `(samples, channels)`

#### 7. File Saving

```python
sf.write(output_path, vocals_out, sr, format='WAV')
```

- **WAV format**: Uncompressed, highest quality
- **Original sample rate**: Preserves audio quality
- **Stereo preserved**: Maintains spatial information

## 🎛️ Model Loading & Caching

### Lazy Loading

The model is loaded only when needed:

```python
def _load_model(self):
    if self.model is None:
        self.model = get_model('htdemucs')
        self.model.to(self.device)
        self.model.eval()
```

**Benefits:**
- Faster startup (no model load on import)
- Memory efficient (only loads when processing)
- First request downloads model (~83 MB)

### Model Download

On first use, Demucs downloads the model:
- **Location**: `~/.cache/torch/hub/` (Linux/Mac) or `%USERPROFILE%\.cache\torch\hub\` (Windows)
- **Size**: ~83 MB
- **One-time**: Cached for all future uses
- **Automatic**: No manual download needed

### Device Selection

```python
self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

- **GPU (CUDA)**: 5-10x faster processing
- **CPU**: Works but slower (still functional)
- **Automatic**: Detects available hardware

## 📊 Progress Tracking

### Callback System

```python
def progress_callback(progress):
    job.progress = min(90, 10 + int(progress * 0.8))
```

**Progress Mapping:**
- `0-10%`: File upload and validation
- `10-90%`: Audio processing (mapped from Demucs internal progress)
- `90-95%`: File saving
- `95-100%`: Finalization

**Why 10-90%?**
- Reserves 10% for setup/teardown
- Maps Demucs progress (0-1.0) to 10-90%
- Provides smooth progress updates

### Frontend Polling

```javascript
setInterval(async () => {
    const response = await fetch(`/status/${job_id}`);
    const data = await response.json();
    updateProgressBar(data.progress);
}, 1000);  // Poll every second
```

**Polling Strategy:**
- **Frequency**: 1 second intervals
- **Efficiency**: Lightweight status checks
- **Stops**: When status is 'completed' or 'error'

## 🗄️ Job Management

### In-Memory Storage

```python
processing_jobs = {}  # {job_id: ProcessingJob}
```

**Job Object:**
```python
class ProcessingJob:
    job_id: str
    filename: str
    status: str  # 'pending', 'processing', 'completed', 'error'
    progress: int  # 0-100
    error_message: Optional[str]
    output_file: Optional[str]
    created_at: float  # timestamp
```

### Cleanup System

Old jobs are automatically cleaned up:

```python
def cleanup_old_jobs():
    # Runs every 5 minutes
    # Removes jobs older than 1 hour
    # Deletes associated output files
```

**Why cleanup?**
- Prevents memory leaks
- Frees disk space
- Removes old processed files

## 🔒 Error Handling

### File Validation

```python
# Type check
if not allowed_file(file.filename):
    raise HTTPException(status_code=400)

# Size check
if len(contents) > MAX_UPLOAD_SIZE:
    raise HTTPException(status_code=400)

# Format validation
if not validate_audio_file(filepath):
    raise HTTPException(status_code=400)
```

### Processing Errors

```python
try:
    output_file = separator.process(...)
except Exception as e:
    job.status = 'error'
    job.error_message = str(e)
    # Clean up input file
```

### Frontend Error Display

```javascript
if (data.status === 'error') {
    showError(data.error_message);
}
```

## 🚀 Performance Considerations

### Processing Time

**Factors:**
- **File length**: Longer songs = more processing time
- **Hardware**: GPU vs CPU makes huge difference
- **Model complexity**: HTDemucs is high-quality but slower

**Typical Times:**
- **3-minute song, CPU**: ~2-3 minutes
- **3-minute song, GPU**: ~20-30 seconds
- **First run**: +30 seconds (model download)

### Memory Usage

- **Model**: ~500 MB RAM
- **Audio buffer**: Depends on file length
- **Peak usage**: ~1-2 GB for typical files

### Optimization Strategies

1. **Lazy model loading**: Only load when needed
2. **Chunk processing**: Split long files automatically
3. **GPU acceleration**: Automatic when available
4. **File cleanup**: Automatic removal of old files

## 🔧 Technical Stack Details

### Backend Stack

**FastAPI:**
- Async/await support
- Automatic API documentation
- Type validation
- CORS middleware

**Demucs:**
- PyTorch-based neural network
- Pre-trained on MusDB dataset
- State-of-the-art separation quality

**librosa:**
- Audio loading (all formats)
- Format conversion
- Sample rate handling

**soundfile:**
- WAV file writing
- High-quality output
- Cross-platform support

### Frontend Stack

**Vanilla JavaScript:**
- No framework overhead
- Direct DOM manipulation
- Fetch API for HTTP requests

**Modern CSS:**
- Flexbox/Grid layouts
- CSS animations
- Responsive design

## 🎨 UI/UX Flow

### User Journey

1. **Landing**: Clean upload interface
2. **Upload**: Drag & drop or file picker
3. **Processing**: Real-time progress bar
4. **Complete**: Download button appears
5. **Download**: File downloads automatically

### State Management

```javascript
// States
- 'upload': File selection
- 'processing': Progress bar visible
- 'done': Download button shown
- 'error': Error message displayed
```

## 🔐 Security Considerations

### File Upload Security

- **Type validation**: Only allowed extensions
- **Size limits**: 100 MB maximum
- **Filename sanitization**: Prevents path traversal
- **Temporary storage**: Files cleaned up after processing

### API Security

- **CORS**: Configured for development (should restrict in production)
- **Input validation**: FastAPI automatic validation
- **Error messages**: Don't expose internal details

## 📈 Future Improvements

### Potential Enhancements

1. **Database**: Replace in-memory jobs with persistent storage
2. **Queue System**: Use Celery/RQ for better job management
3. **WebSockets**: Real-time progress updates (no polling)
4. **Multiple Models**: Let users choose model quality/speed
5. **Batch Processing**: Process multiple files at once
6. **Cloud Storage**: Store files in S3/cloud storage
7. **Authentication**: User accounts and file history

## 📚 References

- [Demucs Paper](https://arxiv.org/abs/2111.03600)
- [Demucs GitHub](https://github.com/facebookresearch/demucs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [librosa Documentation](https://librosa.org/doc/latest/index.html)

---

This document provides a comprehensive technical overview. For user-facing documentation, see [README.md](README.md).

