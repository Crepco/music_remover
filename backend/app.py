from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import threading
import time
from audio_processor.separator import AudioSeparator
from audio_processor.utils import validate_audio_file, get_file_info

app = FastAPI(title="Music Remover API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'flac', 'ogg', 'm4a', 'aac'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

processing_jobs = {}


class ProcessingJob:
    def __init__(self, job_id: str, filename: str):
        self.job_id = job_id
        self.filename = filename
        self.status = 'pending'
        self.progress = 0
        self.error_message = None
        self.output_file = None
        self.created_at = time.time()


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def safe_filename(job_id: str, original: str) -> str:
    ext = original.rsplit('.', 1)[-1].lower() if '.' in original else 'wav'
    return f"{job_id}.{ext}"


@app.get("/")
async def root():
    return {"message": "Music Remover API is running."}


@app.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    if not allowed_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"File type not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS).upper()}"
        )

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds 100 MB limit")

    job_id = str(uuid.uuid4())
    filename = safe_filename(job_id, file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    try:
        with open(filepath, "wb") as f:
            f.write(contents)

        if not validate_audio_file(filepath):
            os.remove(filepath)
            raise HTTPException(status_code=400, detail="Invalid or corrupted audio file")

        file_info = get_file_info(filepath)

        job = ProcessingJob(job_id, filename)
        processing_jobs[job_id] = job

        background_tasks.add_task(process_audio_background, job_id, filepath)

        return {
            "job_id": job_id,
            "filename": file.filename,
            "file_info": file_info,
            "status": "pending"
        }

    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = processing_jobs[job_id]
    return {
        "job_id": job_id,
        "status": job.status,
        "progress": job.progress,
        "error_message": job.error_message
    }


@app.get("/download/{job_id}")
async def download_file(job_id: str):
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = processing_jobs[job_id]
    if job.status != 'completed':
        raise HTTPException(status_code=400, detail="Processing not completed yet")

    if not job.output_file or not os.path.exists(job.output_file):
        raise HTTPException(status_code=404, detail="Output file not found")

    return FileResponse(
        job.output_file,
        media_type='audio/wav',
        filename=f"vocals_{os.path.basename(job.output_file)}"
    )


def process_audio_background(job_id: str, input_file: str):
    job = processing_jobs[job_id]
    try:
        job.status = 'processing'
        job.progress = 10

        separator = AudioSeparator()

        def progress_callback(progress):
            job.progress = min(90, 10 + int(progress * 0.8))

        output_file = separator.process(input_file, progress_callback=progress_callback)

        job.progress = 95

        output_filename = f"vocals_{job.filename.rsplit('.', 1)[0]}.wav"
        final_output = os.path.join(OUTPUT_FOLDER, output_filename)
        os.rename(output_file, final_output)

        job.output_file = final_output
        job.progress = 100
        job.status = 'completed'

        if os.path.exists(input_file):
            os.remove(input_file)

    except Exception as e:
        job.status = 'error'
        job.error_message = str(e)
        if os.path.exists(input_file):
            os.remove(input_file)


def cleanup_old_jobs():
    def _cleanup():
        while True:
            current_time = time.time()
            to_remove = []
            for job_id, job in processing_jobs.items():
                if current_time - job.created_at > 3600:
                    if job.output_file and os.path.exists(job.output_file):
                        os.remove(job.output_file)
                    to_remove.append(job_id)
            for job_id in to_remove:
                del processing_jobs[job_id]
            time.sleep(300)

    t = threading.Thread(target=_cleanup, daemon=True)
    t.start()


cleanup_old_jobs()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
