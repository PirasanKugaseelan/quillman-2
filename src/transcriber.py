"""
Speech-to-text transcription service based on OpenAI Whisper.
"""

import tempfile
import time
import os
import openai  # New import

from modal import Image, method

from .common import stub
from .gpt_3_5 import GPT35Turbo  # Import GPT-3.5 Turbo class

MODEL_NAME = "base.en"

openai.api_key = os.getenv('OPENAI_API_KEY')  # Set OpenAI API key


def download_model():
    import whisper

    whisper.load_model(MODEL_NAME)


transcriber_image = (
    Image.debian_slim(python_version="3.10.8")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "https://github.com/openai/whisper/archive/v20230314.tar.gz",
        "ffmpeg-python",
    )
    .run_function(download_model)
)


def load_audio(data: bytes, sr: int = 16000):
    import ffmpeg
    import numpy as np

    try:
        fp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        fp.write(data)
        fp.close()
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(
                fp.name,
                threads=0,
                format="f32le",
                acodec="pcm_f32le",
                ac=1,
                ar="48k",
            )
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(
                cmd=["ffmpeg", "-nostdin"],
                capture_stdout=True,
                capture_stderr=True,
            )
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.float32).flatten()


@stub.cls(
    gpu="A10G",
    container_idle_timeout=180,
    image=transcriber_image,
)
class Whisper:
    def __enter__(self):
        import torch
        import whisper

        self.use_gpu = torch.cuda.is_available()
        device = "cuda" if self.use_gpu else "cpu"
        self.model = whisper.load_model(MODEL_NAME, device=device)
        self.gpt35turbo = GPT35Turbo()  # Initialize GPT-3.5 Turbo

    @method()
    def transcribe_and_respond(
        self,
        audio_data: bytes,
    ):
        t0 = time.time()
        np_array = load_audio(audio_data)
        transcript = self.model.transcribe(np_array, language="en", fp16=self.use_gpu)  # type: ignore
        print(f"Transcribed in {time.time() - t0:.2f}s")

        # Use the transcription in the chat model
        t1 = time.time()
        response = self.gpt35turbo.send_message(transcript)  # Call GPT-3.5 Turbo
        print(f"Response generated in {time.time() - t1:.2f}s")

        # Return both the transcription and the response
        return transcript, response
