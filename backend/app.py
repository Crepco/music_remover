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

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not supported'}), 400
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    filename = secure_filename(f"{job_id}_{file.filename}")
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    try:
        file.save(filepath)
        
        # Validate file
        if not validate_audio_file(filepath):
            os.remove(filepath)
            return jsonify({'error': 'Invalid audio file'}), 400
        
        # Get file info
        file_info = get_file_info(filepath)
        
        # Create processing job
        job = ProcessingJob(job_id, filename)
        processing_jobs[job_id] = job
        
        # Start processing in background
        separation_method = request.form.get('method', 'vocal_removal')
        quality = request.form.get('quality', 'medium')
        
        thread = threading.Thread(
            target=process_audio_background,
            args=(job_id, filepath, separation_method, quality)
        )
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'filename': file.filename,
            'file_info': file_info,
            'status': 'pending'
        })
        
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

@app.route('/status/<job_id>')
def get_status(job_id):
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = processing_jobs[job_id]
    return jsonify({
        'job_id': job_id,
        'status': job.status,
        'progress': job.progress,
        'error_message': job.error_message
    })

@app.route('/download/<job_id>')
def download_file(job_id):
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = processing_jobs[job_id]
    if job.status != 'completed':
        return jsonify({'error': 'Processing not completed'}), 400
    
    if not job.output_file or not os.path.exists(job.output_file):
        return jsonify({'error': 'Output file not found'}), 404
    
    return send_file(job.output_file, as_attachment=True)

@app.route('/methods')
def get_methods():
    return jsonify({
        'methods': [
            {
                'id': 'vocal_removal',
                'name': 'Vocal Removal',
                'description': 'Remove vocals using center channel extraction'
            },
            {
                'id': 'instrumental_isolation',
                'name': 'Instrumental Isolation', 
                'description': 'Isolate instrumental parts'
            },
            {
                'id': 'harmonic_percussive',
                'name': 'Harmonic-Percussive Separation',
                'description': 'Separate harmonic and percussive elements'
            }
        ],
        'quality_options': ['low', 'medium', 'high']
    })

def process_audio_background(job_id, input_file, method, quality):
    job = processing_jobs[job_id]
    try:
        job.status = 'processing'
        job.progress = 10
        
        separator = AudioSeparator()
        
        # Set progress callback
        def progress_callback(progress):
            job.progress = min(90, 10 + int(progress * 0.8))
        
        # Process audio
        output_file = separator.process(
            input_file, 
            method=method, 
            quality=quality,
            progress_callback=progress_callback
        )
        
        job.progress = 95
        
        # Move output to outputs folder
        output_filename = f"processed_{job.filename}"
        final_output = os.path.join(OUTPUT_FOLDER, output_filename)
        os.rename(output_file, final_output)
        
        job.output_file = final_output
        job.progress = 100
        job.status = 'completed'
        
        # Clean up input file
        if os.path.exists(input_file):
            os.remove(input_file)
            
    except Exception as e:
        job.status = 'error'
        job.error_message = str(e)
        
        # Clean up files
        if os.path.exists(input_file):
            os.remove(input_file)

# Clean up old jobs periodically
@app.before_first_request
def cleanup_old_jobs():
    def cleanup():
        while True:
            current_time = time.time()
            jobs_to_remove = []
            
            for job_id, job in processing_jobs.items():
                # Remove jobs older than 1 hour
                if current_time - job.created_at > 3600:
                    if job.output_file and os.path.exists(job.output_file):
                        os.remove(job.output_file)
                    jobs_to_remove.append(job_id)
            
            for job_id in jobs_to_remove:
                del processing_jobs[job_id]
            
            time.sleep(300)  # Check every 5 minutes
    
    cleanup_thread = threading.Thread(target=cleanup)
    cleanup_thread.daemon = True
    cleanup_thread.start()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)




