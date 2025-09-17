from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid
from werkzeug.utils import secure_filename
from audio_processor.separator import AudioSeparator
from audio_processor.utils import validate_audio_file, get_file_info
import threading
import time



app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
Max_Upload_Size = 100 * 1024 * 1024  # 100 MB
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'flac', 'ogg', 'm4a', 'aac'}


# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)       

# In-memory storage for job statuses
processing_jobs = {}


class ProcessingJob:
    def __init__(self, job_id, filename):
        self.job_id = job_id 
        self.filename = filename
        self.status = 'pending'  # pending, processing, completed, error
        self.progress = 0
        self.error_message = None
        self.output_file = None
        self.created_at = time.time()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return "music seperator API is running."
