import csv
import os
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.livekit.transport import LiveKitParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "float16")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_heart")

TOOL_LOG = Path.home() / "tool_calls.csv"
HTTP_TIMEOUT = aiohttp.ClientTimeout(total=5)
HEADERS = {"User-Agent": "voice-agent-student-project/0.1"}

transport_params = {
    "eval": lambda: EvalTransportParams(audio_in_enabled=True, audio_out_enabled=True),
    "daily": lambda: DailyParams(audio_in_enabled=True, audio_out_enabled=True),
    "livekit": lambda: LiveKitParams(audio_in_enabled=True, audio_out_enabled=True),
    "twilio": lambda: FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True),
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


def log_tool_call(name, args, started, result):
    """Append one row per tool call, for the evaluation tables."""
    ms = round((time.perf_counter() - started) * 1000)
    ok = "error" not in result
    logger.info(f"TOOL {name} args={args} took={ms}ms ok={ok}")
    new_file = not TOOL_LOG.exists()
    with open(TOOL_LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["time", "tool", "args", "ms", "ok"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), name, str(args), ms, ok])


async def get_weather(params: FunctionCallParams):
    started = time.perf_counter()
    location = str(params.arguments.get("location", "")).strip()
    try:
        async with aiohttp.ClientSession(timeout=HTTP_TIMEOUT, headers=HEADERS) as s:
            async with s.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": location, "count": 1},
            ) as r:
                geo = await r.json()
            places = geo.get("results") or []
            if not places:
                result = {"error": f"Could not find a place called {location}"}
            else:
                p = places[0]
                async with s.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": p["latitude"],
                        "longitude": p["longitude"],
                        "current": "temperature_2m,wind_speed_10m",
                    },
                ) as r:
                    w = await r.json()
                cur = w.get("current", {})
                result = {
                    "place": p.get("name"),
                    "country": p.get("country"),
                    "temperature_celsius": cur.get("temperature_2m"),
                    "wind_speed_kmh": cur.get("wind_speed_10m"),
                    "observed_at": cur.get("time"),
                    "source": "open-meteo.com",
                }
    except Exception as e:
        result = {"error": f"Weather service unavailable ({type(e).__name__})"}
    log_tool_call("get_weather", {"location": location}, started, result)
    await params.result_callback(result)


async def get_current_time(params: FunctionCallParams):
    started = time.perf_counter()
    tz = str(params.arguments.get("timezone", "Asia/Kolkata")).strip() or "Asia/Kolkata"
    try:
        now = datetime.now(ZoneInfo(tz))
        result = {"timezone": tz, "time": now.strftime("%I:%M %p"), "date": now.strftime("%A, %d %B %Y")}
    except Exception:
        result = {"error": f"Unknown timezone {tz}. Use a name like Asia/Kolkata or Europe/London."}
    log_tool_call("get_current_time", {"timezone": tz}, started, result)
    await params.result_callback(result)


async def wikipedia_summary(params: FunctionCallParams):
    started = time.perf_counter()
    topic = str(params.arguments.get("topic", "")).strip()
    try:
        title = quote(topic.replace(" ", "_"))
        async with aiohttp.ClientSession(timeout=HTTP_TIMEOUT, headers=HEADERS) as s:
            async with s.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}") as r:
                if r.status != 200:
                    result = {"error": f"No Wikipedia page found for {topic}"}
                else:
                    d = await r.json()
                    result = {
                        "title": d.get("title"),
                        "summary": (d.get("extract") or "")[:500],
                        "source_url": d.get("content_urls", {}).get("desktop", {}).get("page"),
                    }
    except Exception as e:
        result = {"error": f"Wikipedia unavailable ({type(e).__name__})"}
    log_tool_call("wikipedia_summary", {"topic": topic}, started, result)
    await params.result_callback(result)


weather_schema = FunctionSchema(
    name="get_weather",
    description="Get the current weather for a city.",
    properties={"location": {"type": "string", "description": "City name, e.g. Chandigarh"}},
    required=["location"],
)
time_schema = FunctionSchema(
    name="get_current_time",
    description="Get the current time and date in a timezone.",
    properties={
        "timezone": {
            "type": "string",
            "description": "IANA timezone name, e.g. Asia/Kolkata, Europe/London, Asia/Tokyo",
        }
    },
    required=["timezone"],
)
wiki_schema = FunctionSchema(
    name="wikipedia_summary",
    description="Look up a short factual summary of a person, place or topic on Wikipedia.",
    properties={"topic": {"type": "string", "description": "The topic title, e.g. Eiffel Tower"}},
    required=["topic"],
)
tools = ToolsSchema(standard_tools=[weather_schema, time_schema, wiki_schema])


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(
        f"Starting local tools bot: whisper={WHISPER_MODEL}/{WHISPER_DEVICE}/{WHISPER_COMPUTE}, "
        f"ollama={OLLAMA_MODEL}, kokoro={KOKORO_VOICE}"
    )

    stt = WhisperSTTService(
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE,
        settings=WhisperSTTService.Settings(model=WHISPER_MODEL),
    )
    tts = KokoroTTSService(settings=KokoroTTSService.Settings(voice=KOKORO_VOICE))

    llm = OLLamaLLMService(
        base_url="http://localhost:11434/v1",
        settings=OLLamaLLMService.Settings(
            model=OLLAMA_MODEL,
            system_instruction=(
                "You are a helpful assistant in a voice conversation. Your replies are spoken aloud, "
                "so use short plain sentences with no lists, emojis or formatting. "
                "Use the tools for current weather, the time, or facts from Wikipedia. "
                "Only state what a tool returned. If a tool returns an error, say you could not get "
                "that information. Never invent weather, times or sources."
            ),
        ),
    )
    llm.register_function("get_weather", get_weather)
    llm.register_function("get_current_time", get_current_time)
    llm.register_function("wikipedia_summary", wikipedia_summary)

    context = LLMContext(tools=tools)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        context.add_message(
            {"role": "developer", "content": "Please introduce yourself to the user in one sentence."}
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
