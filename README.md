# 🎤 Music Remover

A simple web application that removes background music from audio files, leaving only the vocals. Built with FastAPI and Demucs (Meta's state-of-the-art source separation model).

## ✨ Features

- **AI-Powered Music Removal**: Uses Demucs deep learning model for professional-quality vocal isolation
- **Simple Interface**: Upload → Process → Download - no complicated settings
- **Real-time Progress**: Watch your file being processed in real-time
- **Multiple Formats**: Supports MP3, WAV, FLAC, OGG, M4A, AAC
- **Fast Processing**: GPU acceleration when available, CPU fallback otherwise

## 🎯 What It Does

Upload any song, and the app will:
1. Analyze the audio using AI
2. Separate vocals from background music
3. Return a clean vocals-only WAV file

Perfect for:
- Creating karaoke tracks
- Extracting vocal samples
- Isolating voice recordings
- Music production workflows

## 📋 Prerequisites

- **Python 3.8+** (Python 3.13 recommended)
- **Node.js 14+** and npm
- **FFmpeg** (for audio processing)

### Installing FFmpeg

**Windows:**
```bash
# Using Chocolatey
choco install ffmpeg

# Or download from https://ffmpeg.org/download.html
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd music_remover
```

### 2. Install Dependencies

**Python dependencies:**
```bash
npm run install-deps
```

Or manually:
```bash
cd backend
pip install -r requirements.txt
```

**npm dependencies:**
```bash
npm install
```

### 3. Start the Application

```bash
npm start
```

This starts:
- **Backend API**: `http://localhost:5000`
- **Frontend**: `http://localhost:8000`

Open `http://localhost:8000` in your browser.

## 📖 How to Use

### Step-by-Step Guide

1. **Start the Application**
   ```bash
   npm start
   ```
   Wait for both servers to start (you'll see confirmation messages).

2. **Open the Web Interface**
   - Navigate to `http://localhost:8000` in your browser
   - You'll see a clean upload interface

3. **Upload Your Audio File**
   - **Option 1**: Drag and drop an audio file onto the upload area
   - **Option 2**: Click the upload area to browse and select a file
   - Supported formats: MP3, WAV, FLAC, OGG, M4A, AAC
   - Maximum file size: 100 MB

4. **Process the File**
   - Once uploaded, click the **"Remove Music"** button
   - Processing starts automatically
   - Watch the progress bar as your file is processed
   - Processing time varies (typically 1-5 minutes depending on file length)

5. **Download Your Vocals**
   - When processing completes, click **"Download Vocals"**
   - Your vocals-only WAV file will download automatically
   - The file will be named `vocals_[original_filename].wav`

### First Time Using?

- On your first upload, the Demucs AI model (~83 MB) will download automatically
- This only happens once and is cached for future use
- Subsequent uploads will be faster

### Tips

- **Best results**: Use high-quality source files (WAV or FLAC recommended)
- **Processing time**: Longer songs take more time to process
- **File size**: Keep files under 100 MB for best performance
- **Multiple files**: Process one file at a time for optimal results

## 🏗️ Project Structure

```
music_remover/
├── backend/
│   ├── app.py                    # FastAPI server
│   ├── requirements.txt          # Python dependencies
│   └── audio_processor/
│       ├── separator.py          # Demucs integration
│       ├── utils.py              # Audio utilities
│       └── filters.py            # (Legacy - not used)
├── frontend/
│   ├── index.html               # Main UI
│   ├── styles.css               # Styling
│   └── app.js                   # Frontend logic
├── package.json                 # npm scripts
└── README.md                    # This file
```

## 🔧 Development

### Running Services Separately

**Backend only:**
```bash
npm run backend
```

**Frontend only:**
```bash
npm run frontend
```

### API Documentation

FastAPI provides interactive API docs:
- **Swagger UI**: http://localhost:5000/docs
- **ReDoc**: http://localhost:5000/redoc

## 🔌 API Endpoints

### `POST /upload`
Upload an audio file for processing.

**Request:**
- `file`: Audio file (multipart/form-data)

**Response:**
```json
{
  "job_id": "uuid",
  "filename": "song.mp3",
  "file_info": {
    "sample_rate": 44100,
    "duration": 180.5,
    "channels": "stereo"
  },
  "status": "pending"
}
```

### `GET /status/{job_id}`
Check processing status.

**Response:**
```json
{
  "job_id": "uuid",
  "status": "processing",
  "progress": 45,
  "error_message": null
}
```

### `GET /download/{job_id}`
Download the processed vocals file.

## 🧠 How It Works

The app uses **Demucs** (by Meta Research), a state-of-the-art deep learning model trained on thousands of songs. Unlike traditional DSP methods, Demucs actually understands what vocals sound like vs instruments.

**Process:**
1. Audio is loaded and converted to the model's expected format
2. Demucs neural network analyzes the audio spectrogram
3. The model separates sources: drums, bass, other instruments, and vocals
4. Only the vocals track is extracted and returned

**Model:** `htdemucs` (Hybrid Transformer Demucs) - the highest quality model available.

**Want more technical details?** See [HOW_IT_WORKS.md](HOW_IT_WORKS.md) for a comprehensive deep dive into the architecture, algorithms, and implementation.

## 🛠️ Technologies

**Backend:**
- FastAPI - Modern Python web framework
- Demucs - AI-powered source separation
- librosa - Audio loading and processing
- Uvicorn - ASGI server

**Frontend:**
- Vanilla JavaScript - No frameworks needed
- Modern CSS - Clean, responsive design

**Tools:**
- npm - Package management
- concurrently - Run multiple processes

## ⚠️ Troubleshooting

### "FFmpeg not found"
- Install FFmpeg and ensure it's in your PATH
- Verify: `ffmpeg -version`

### "Module not found" errors
- Run `npm run install-deps` to install Python packages
- Ensure you're using the correct Python version (3.8+)

### Port already in use
- Backend: port 5000
- Frontend: port 8000
- Change ports in `package.json` if needed

### Processing is slow
- First run downloads the model (~83 MB) - this is normal
- GPU acceleration requires CUDA-compatible GPU
- CPU processing works but is slower

### "torchcodec" errors
- Already fixed! The app uses librosa for audio loading
- No torchcodec dependency needed

## 📝 Notes

- **First run**: The Demucs model (~83 MB) downloads automatically
- **File size limit**: 100 MB per file
- **Output format**: Always WAV (highest quality)
- **Processing time**: Depends on file length and hardware (typically 1-5 minutes)

## 📄 License

MIT License - feel free to use for personal or commercial projects.

## 🙏 Acknowledgments

- [Demucs](https://github.com/facebookresearch/demucs) by Meta Research
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
- [librosa](https://librosa.org/) for audio processing

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.
