# Real-time voice agent (Pipecat + Ollama)

Prototype of a real-time voice agent with interruption handling, conversation
context and tool calling, running on a 6 GB GPU (RTX 4050).

## Configurations
- `cloud-hybrid.py`: Deepgram STT + local Ollama LLM + Cartesia TTS (needs API keys)
- `local.py`: faster-whisper STT + Ollama LLM + Kokoro TTS (no API keys)
- `local-tools.py`: local stack plus tools (weather, time, Wikipedia)

## Setup
1. Clone Pipecat: `git clone https://github.com/pipecat-ai/pipecat.git`
2. Python 3.12 venv, then:
   `pip install -e ".[webrtc,silero,runner,daily,livekit,websocket,ollama,whisper,kokoro,deepgram,cartesia]"`
3. Install Ollama and run `ollama pull qwen2.5:3b`
4. Copy the files into `pipecat/examples/getting-started/` and run with Python
5. For cloud-hybrid, put DEEPGRAM_API_KEY and CARTESIA_API_KEY in a `.env` file (never commit it)

## GPU note
faster-whisper on the GPU needs the pip packages `nvidia-cublas-cu12` and
`nvidia-cudnn-cu12`, with their folders on LD_LIBRARY_PATH (see `run_local.sh`).

## Status
Working prototype. Latency and multilingual (English/Hindi/Punjabi)
evaluation are in progress.
