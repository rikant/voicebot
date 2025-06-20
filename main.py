import os
import asyncio
import json
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from gtts import gTTS
import aiofiles
from deepgram import Deepgram
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

if not OPENAI_API_KEY:
    raise Exception("OPENAI_API_KEY not found in environment.")

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
deepgram = Deepgram(DEEPGRAM_API_KEY)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.websocket("/voice")
async def voicebot(websocket: WebSocket):
    await websocket.accept()
    print("[WS] WebSocket connection accepted")

    dg_connection = await deepgram.transcription.live(
        {'punctuate': True, 'interim_results': False},
        socket=websocket
    )

    @dg_connection.on('transcript')
    async def on_transcript(data):
        transcript = data.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "")
        if transcript:
            print(f"[USER]: {transcript}")
            reply = await ask_chatgpt(transcript)
            print(f"[BOT]: {reply}")
            audio_data = await text_to_speech(reply)
            await websocket.send_bytes(audio_data)

    while True:
        try:
            frame = await websocket.receive_bytes()
            await dg_connection.send(frame)
        except Exception as e:
            print(f"[ERROR] WebSocket error: {e}")
            break

async def ask_chatgpt(user_input: str) -> str:
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful and polite voice assistant. Keep responses under 50 words."},
                {"role": "user", "content": user_input}
            ],
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] ChatGPT call failed: {e}")
        return "Sorry, I couldn't process that."

async def text_to_speech(text: str) -> bytes:
    try:
        if not text.strip():
            return b''
        tts = gTTS(text=text)
        output_path = "response.mp3"
        tts.save(output_path)
        async with aiofiles.open(output_path, mode='rb') as f:
            return await f.read()
    except Exception as e:
        print(f"[ERROR] TTS failed: {e}")
        return b''
