import os
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from deepgram import Deepgram
from gtts import gTTS
import aiofiles
from dotenv import load_dotenv

load_dotenv()

# Load credentials
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Verify required keys
if not OPENAI_API_KEY:
    raise Exception("❌ Missing OPENAI_API_KEY in environment")
if not DEEPGRAM_API_KEY:
    raise Exception("❌ Missing DEEPGRAM_API_KEY in environment")

# Initialize clients
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
deepgram = Deepgram(DEEPGRAM_API_KEY)

# Setup FastAPI app
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
    print("[WS] ✅ WebSocket connection accepted from Twilio")

    # Start Deepgram live transcription
    try:
        dg_connection = await deepgram.transcription.live({
            "punctuate": True,
            "interim_results": False
        })

        @dg_connection.on("transcript")
        async def on_transcript(data):
            transcript = data.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "")
            if transcript:
                print(f"[USER]: {transcript}")
                reply = await ask_chatgpt(transcript)
                print(f"[BOT]: {reply}")
                audio_data = await text_to_speech(reply)
                print("[TTS] 🔊 Sending voice response to Twilio...")
                await websocket.send_bytes(audio_data)

        print("[DG] 🎙️ Deepgram connection ready, waiting for audio...")
        while True:
            audio_chunk = await websocket.receive_bytes()
            await dg_connection.send(audio_chunk)

    except Exception as e:
        print(f"[ERROR] WebSocket error: {e}")
        await websocket.close()

# GPT-3.5 response
async def ask_chatgpt(user_input: str) -> str:
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a polite voice assistant. Reply in under 50 words."},
                {"role": "user", "content": user_input}
            ],
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] GPT call failed: {e}")
        return "Sorry, I couldn't understand you. Please repeat."

# Google Text-to-Speech
async def text_to_speech(text: str) -> bytes:
    try:
        tts = gTTS(text=text)
        file_path = "response.mp3"
        tts.save(file_path)
        async with aiofiles.open(file_path, mode='rb') as f:
            return await f.read()
    except Exception as e:
        print(f"[ERROR] TTS failed: {e}")
        return b''
