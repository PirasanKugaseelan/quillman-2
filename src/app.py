from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import requests
import json
from dotenv import load_dotenv
import os
import openai

app = Flask(__name__)
CORS(app)

load_dotenv()  # take environment variables from .env.
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
ELEVEN_LABS_VOICE_ID = os.getenv("ELEVEN_LABS_VOICE_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure allowed file types and maximum size (in bytes)
app.config['UPLOAD_EXTENSIONS'] = ['.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm']
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024  # 25MB

openai.api_key = OPENAI_API_KEY

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part in the request.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file.'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join('/tmp', filename)
        file.save(filepath)

        with open(filepath, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
        
        return jsonify({'status': 'success', 'transcript': transcript.text}), 200

    return jsonify({'status': 'error', 'message': 'Invalid file.'}), 400

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    
    # GPT-based LLM text generation
    gpt_headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {OPENAI_API_KEY}',
    }
    gpt_data = {
        "prompt": data.get('input', ''),
        "max_tokens": 60
    }
    gpt_response = requests.post('https://api.openai.com/v1/engines/davinci-codex/completions', headers=gpt_headers, data=json.dumps(gpt_data))
    gpt_response_data = gpt_response.json()
    
    generated_text = gpt_response_data.get('choices')[0].get('text')

    # Eleven Labs text-to-speech
    eleven_labs_headers = {
        'Content-Type': 'application/json',
        'xi-api-key': ELEVEN_LABS_API_KEY,
    }
    model_id = data.get('model_id', 'eleven_monolingual_v1')
    voice_settings = {
        "stability": 0.5,
        "similarity_boost": 0.5
    }
    eleven_labs_data = {
        "text": generated_text,
        "model_id": model_id,
        "voice_settings": voice_settings
    }
    eleven_labs_response = requests.post(f'https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_LABS_VOICE_ID}', headers=eleven_labs_headers, data=json.dumps(eleven_labs_data))
    eleven_labs_response_data = eleven_labs_response.json()

    return jsonify(eleven_labs_response_data), eleven_labs_response.status_code

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['UPLOAD_EXTENSIONS']

if __name__ == '__main__':
    app.run(debug=True)