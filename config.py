import os
import shutil

# Ollama
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Whisper
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

# Video pipeline
MAX_HIGHLIGHT_DURATION = int(os.getenv("MAX_HIGHLIGHT_DURATION", "60"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")

# Orchestrator
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# ── FFmpeg Discovery ──────────────────────────────────────────────────
_FFMPEG_FALLBACKS = [
    "/opt/homebrew/bin/ffmpeg",   # Apple Silicon
    "/usr/local/bin/ffmpeg",      # Intel Mac
    "/usr/bin/ffmpeg",            # Linux/System
]

def _discover_ffmpeg():
    # Try system path first
    cmd = shutil.which("ffmpeg")
    if cmd: return cmd
    
    # Try common fallback locations
    for p in _FFMPEG_FALLBACKS:
        if os.path.isfile(p):
            # Mutate PATH so child processes (like Whisper's ffmpeg calls) can find it
            bin_dir = os.path.dirname(p)
            if bin_dir not in os.environ["PATH"]:
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]
            return p
            
    return "ffmpeg" # fallback to bare command

FFMPEG_PATH = _discover_ffmpeg()
FFPROBE_PATH = FFMPEG_PATH.replace("ffmpeg", "ffprobe")
# Ensure ffprobe's dir is also in PATH (usually same as ffmpeg)
_ffp_dir = os.path.dirname(FFPROBE_PATH)
if _ffp_dir and _ffp_dir != "." and _ffp_dir not in os.environ["PATH"]:
    os.environ["PATH"] = _ffp_dir + os.pathsep + os.environ["PATH"]
