<img width="1916" height="906" alt="Screenshot 2026-11-20 171033" src="https://github.com/user-attachments/assets/1931557b-a535-4ec8-b1a4-4a63a03509c8" />
<img width="1870" height="875" alt="Screenshot 2026-11-20 171226" src="https://github.com/user-attachments/assets/c70c5f2a-9eaa-4c14-bfae-9fc42619b540" />
# Low-VRAM Real-Time Voice Agent

A real-time, interruptible voice assistant built on the open-source
[Pipecat](https://github.com/pipecat-ai/pipecat) framework, designed to run on a
**6 GB laptop GPU (RTX 4050)**. It compares a fully local pipeline against a
hybrid (local LLM + cloud speech) pipeline and adds live tool calling.

> **Status:** working prototype. Latency benchmarking and multilingual
> (English / Hindi / Punjabi) evaluation are in progress. Items marked `TBD`
> have not been measured yet.

---

## 1. What it does

- **Real-time speech interaction** in the browser (speech in, spoken reply out)
- **Interruption / turn-taking:** the bot stops speaking when the user talks over it
- **Context-aware conversation:** dialogue history is kept within a session
- **Tool / API integration:** live weather, current time and Wikipedia summaries
- **Logging:** every tool call is recorded with duration and success flag
- **Two deployable configurations** for comparison (local vs hybrid)

## 2. Architecture

```
 Browser (mic + speaker)
        |  WebRTC (SmallWebRTC)
        v
 +---------------- Pipecat pipeline ----------------+
 |  Silero VAD -> STT -> LLM (+ tools) -> TTS         |
 +----------------------------------------------------+
                         |
                         +--> get_weather        (Open-Meteo API)
                         +--> get_current_time   (local, offline)
                         +--> wikipedia_summary  (Wikipedia REST API)
```

### Configurations

| Name | STT | LLM | TTS | API keys needed |
|---|---|---|---|---|
| `cloud-hybrid.py` | Deepgram (cloud) | Ollama qwen2.5:3b (local) | Cartesia (cloud) | Deepgram, Cartesia |
| `local.py` | faster-whisper (local) | Ollama qwen2.5:3b (local) | Kokoro (local) | None |
| `local-tools.py` | faster-whisper (local) | Ollama qwen2.5:3b (local) | Kokoro (local) | None |

`local-tools.py` is `local.py` plus the three tools above.

## 3. Results so far

### 3.1 GPU memory (full local stack running, Whisper `small`)

| Measurement | Value |
|---|---|
| Total VRAM in use | 2,917 MiB of 6,141 MiB |
| LLM (qwen2.5:3b, 4-bit) | 2.2 GB, 100% on GPU, 4096-token context |
| Whisper + Kokoro + overhead | about 0.7 GB combined |

*Note: `nvidia-smi` under WSL reports the whole GPU, including Windows
applications, so the bot-only figure may be slightly lower.*

### 3.2 Tool calls (informal session, 5 calls)

| Tool | Duration | Outcome |
|---|---|---|
| `get_current_time` | 6-8 ms | success (runs locally, no network) |
| `wikipedia_summary` | 432 ms | success |
| `wikipedia_summary` | 1,484 ms | failed (see observation 2) |
| `get_weather` | 1,847 ms | success (two web requests: geocoding, then forecast) |

### 3.3 Observations

1. **A 3B local model can issue multiple tool calls in one turn.** In one run it
   called three tools in parallel and combined the answers into one reply.
   This is a single observed run, not a measured success rate.
2. **Speech-recognition errors propagate to tools.** One failed Wikipedia lookup
   searched for a mis-heard term, so the tool failed even though it worked
   correctly on the next, correctly heard request.
3. **The bot did not invent weather data when it had no tool.** The version
   without tools said it could not check the weather (one observed example).
4. **Tool latency is audible.** The weather tool adds roughly 1.8 s before the
   LLM can start answering. A spoken filler while a tool runs would help.
5. **GPU speech recognition is much faster than CPU recognition.** Moving
   faster-whisper from CPU to GPU removed the main delay in the local pipeline.

### 3.4 Latency comparison (TBD)

| Stage | Local | Hybrid (cloud STT/TTS) |
|---|---|---|
| STT | TBD | TBD |
| LLM time to first token | TBD | TBD |
| TTS time to first audio | TBD | TBD |
| End of speech to first audio | TBD | TBD |

### 3.5 Screenshots

Add images to a `docs/` folder and reference them here, for example:

```
![Tools bot](docs/tools-bot.png)
![Cloud-hybrid bot](docs/cloud-no-tools.png)
```

## 4. Setup

Tested on Windows 11 with WSL2 (Ubuntu), an RTX 4050 (6 GB) and Python 3.12.

1. **Install WSL2 + Ubuntu** and confirm the GPU is visible with `nvidia-smi`.
2. **Install Ollama** inside Ubuntu and pull the model:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull qwen2.5:3b
   ```
3. **Clone Pipecat** and create a Python 3.12 environment (using `uv`):
   ```bash
   git clone https://github.com/pipecat-ai/pipecat.git
   cd pipecat
   uv venv --python 3.12 venv312
   source venv312/bin/activate
   uv pip install -e ".[webrtc,silero,runner,daily,livekit,websocket,ollama,whisper,kokoro,deepgram,cartesia]"
   ```
4. **Install the GPU libraries** that faster-whisper needs:
   ```bash
   uv pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12>=9,<10"
   ```
5. **Copy the bot files** into `pipecat/examples/getting-started/` with these names:
   - `cloud-hybrid.py` as `06-ollama-agent.py`
   - `local.py` as `07-local-agent.py`
   - `local-tools.py` as `08-local-tools-agent.py`
6. **Copy `run_local.sh`** to your home folder and make it executable
   (`chmod +x ~/run_local.sh`). It sets the library path and starts a bot.
7. **API keys (cloud-hybrid only):** create `pipecat/.env` with
   `DEEPGRAM_API_KEY=...` and `CARTESIA_API_KEY=...`. Never commit this file.

## 5. Running

```bash
# Fully local, with tools
AGENT=08-local-tools-agent.py WHISPER_MODEL=small ~/run_local.sh

# Fully local, no tools
WHISPER_MODEL=small ~/run_local.sh

# CPU fallback if GPU libraries cause problems
WHISPER_DEVICE=cpu WHISPER_COMPUTE=int8 WHISPER_MODEL=base ~/run_local.sh
```

Then open `http://localhost:7860/client/` in Chrome, click **Connect**, allow
the microphone and talk. **Use headphones:** speaker echo makes the bot
interrupt itself.

### Configuration variables

| Variable | Default | Meaning |
|---|---|---|
| `AGENT` | `07-local-agent.py` | Which bot file `run_local.sh` starts |
| `WHISPER_MODEL` | `small` | faster-whisper model size |
| `WHISPER_DEVICE` | `cuda` | `cuda` or `cpu` |
| `WHISPER_COMPUTE` | `float16` | `float16` (GPU) or `int8` (CPU) |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Local LLM served by Ollama |
| `KOKORO_VOICE` | `af_heart` | Kokoro TTS voice |

## 6. Project structure

```
voice-agent-project/
  cloud-hybrid.py    # Deepgram STT + Ollama LLM + Cartesia TTS
  local.py           # faster-whisper + Ollama + Kokoro
  local-tools.py     # local stack + weather, time, Wikipedia tools
  run_local.sh       # start script (sets GPU library path)
  tool_calls.csv     # sample tool-call log
  docs/              # screenshots
```

## 7. Problems encountered (and fixes)

| Problem | Cause | Fix |
|---|---|---|
| `pipecat.exe` blocked on Windows | Windows application-control policy blocks unsigned executables | Moved the project to WSL2 (Ubuntu) |
| `ImportError: DLL load failed` on Windows | Same policy blocked a compiled dependency | Same: WSL2 |
| `libcublas.so.12 is not found` | CUDA libraries missing from the Python environment | `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` and set `LD_LIBRARY_PATH` |
| Slow transcription | Whisper was running on the CPU | Use the GPU; smaller model if needed |
| Bot interrupts itself | Speaker echo reaches the microphone | Use headphones |
| Python 3.14 package problems | Some AI packages lack support for the newest Python | Use Python 3.12 |

## 8. Limitations

- A 3B model is unreliable at choosing and parameterising tools. Tool-call
  accuracy has only been checked informally.
- Speech-recognition mistakes can break tool calls.
- Only English has been evaluated so far.
- Tool calls add network latency to the response.
- Measurements come from one machine and short sessions.

## 9. Roadmap

1. Per-stage latency benchmark: local vs hybrid, fixed question set (`TBD`)
2. Tool-call accuracy test with known expected tools
3. Spoken filler while a tool runs
4. **Multilingual speech (English, Hindi, Punjabi):** compare speech-recognition
   models on word error rate, latency and VRAM within the 6 GB limit
5. Extensible tool registry with caching, timeouts and fallbacks
6. Source links and dates in tool answers

## 10. Acknowledgements

- [Pipecat](https://github.com/pipecat-ai/pipecat) (voice agent framework)
- [Ollama](https://ollama.com) and the Qwen2.5 model
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [Kokoro](https://github.com/hexgrad/kokoro) text-to-speech
- [Open-Meteo](https://open-meteo.com) and the Wikipedia REST API

## 11. Author

Rajat, [your degree / institution] | [email] | [LinkedIn or contact]

*License: TBD*
