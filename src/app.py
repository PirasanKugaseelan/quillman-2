import json
import httpx
import openai
from pathlib import Path
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from modal import Stub

stub = Stub(name="quillman")

openai.api_key = 'your-openai-api-key'

static_path = Path(__file__).with_name("frontend").resolve()

PUNCTUATION = [".", "?", "!", ":", ";", "*"]

web_app = FastAPI()

@web_app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    audio_file = await file.read()
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript["text"]

@web_app.post("/generate")
async def generate(request: Request):
    body = await request.json()
    tts_enabled = body["tts"]

    if "noop" in body:
        openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=[
              {
                  "role": "system",
                  "content": "You are a helpful assistant."
              },
              {
                  "role": "user",
                  "content": ""
              }
          ]
        )
        return

    async def text_to_speech(sentence):
        url = f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}'

        headers = {
            'accept': 'audio/mpeg',
            'xi-api-key': 'your-xi-api-key',
            'Content-Type': 'application/json',
        }

        data = {
            'text': sentence,
            'model_id': 'eleven_monolingual_v1',
            'voice_settings': {
                'stability': 0.5,
                'similarity_boost': 0.5,
            },
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, data=json.dumps(data))

        # TODO: Handle the response appropriately. The response will be an audio file.
        return response

    async def speak(sentence):
        if tts_enabled:
            try:
                audio_file = await text_to_speech(sentence)
                # Here, you need to handle the response, which is an audio file.
                # You might need to convert it to a format that your frontend can handle.
                return {
                    'type': 'audio',
                    'value': audio_file,
                }
            except Exception as e:
                return {
                    'type': 'error',
                    'value': str(e),
                }
        else:
            return {
                'type': 'sentence',
                'value': sentence,
            }

    def gen():
        sentence = ""

        response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=[
              {
                  "role": "system",
                  "content": "You are a helpful assistant."
              },
              {
                  "role": "user",
                  "content": body["input"]
              }
          ]
        )
        generated_text = response['choices'][0]['message']['content']

        yield {"type": "text", "value": generated_text}

        for p in PUNCTUATION:
            if p in sentence:
                prev_sentence, new_sentence = sentence.rsplit(p, 1)
                yield speak(prev_sentence)
                sentence = new_sentence

        if sentence:
            yield speak(sentence)

    def gen_serialized():
        for i in gen():
            yield json.dumps(i) + "\x1e"
            
    return StreamingResponse(
        gen_serialized(),
        media_type="text/event-stream",
    )

    web_app.mount("/", StaticFiles(directory="/assets", html=True))
