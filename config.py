import os
import shutil

# ── Ollama ────────────────────────────────────────────────────────────
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Seconds to wait for a single LLM response. Large prompts on CPU take 3-5 min.
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))

# ── Whisper ───────────────────────────────────────────────────────────
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

# ── Video pipeline ────────────────────────────────────────────────────
MAX_HIGHLIGHT_DURATION = int(os.getenv("MAX_HIGHLIGHT_DURATION", "60"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")

# ── Orchestrator ──────────────────────────────────────────────────────
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# ── Embeddings ───────────────────────────────────────────────────────
# Used to match scene descriptions against the user's style query.
# nomic-embed-text is small (274 MB) and purpose-built for semantic search.
# Pull once: ollama pull nomic-embed-text
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
# Weight of embedding similarity vs LLM importance score (must sum to 1.0)
EMBED_WEIGHT = float(os.getenv("EMBED_WEIGHT", "0.4"))

# ── Vision (LLaVA frame description) ─────────────────────────────────
# Requires a GPU (NVIDIA CUDA or Apple Silicon MPS) for practical speed.
# Set ENABLE_VISION=false only if running on CPU-only hardware temporarily.
ENABLE_VISION = os.getenv("ENABLE_VISION", "true").lower() in ("1", "true", "yes")

VISION_MODEL = os.getenv("VISION_MODEL", "llava")
MAX_VISUAL_DESCRIPTIONS = int(os.getenv("MAX_VISUAL_DESCRIPTIONS", "20"))
VISION_TIMEOUT = int(os.getenv("VISION_TIMEOUT", "180"))
VISION_MULTI_FRAME_MIN = float(os.getenv("VISION_MULTI_FRAME_MIN", "15.0"))
VISION_MAX_CONSECUTIVE_FAILURES = int(os.getenv("VISION_MAX_CONSECUTIVE_FAILURES", "2"))

# ── Zoom / Focus ──────────────────────────────────────────────────────
# Shared 3x3 grid vocabulary used by:
#   - TranscriberAgent  : LLaVA reports where the subject/action is in frame
#   - HighlighterAgent  : picks a zoom target + level per highlight clip
#   - EditorAgent       : maps the grid cell to a crop anchor for ffmpeg
FOCUS_POSITIONS = [
    "center",
    "top-left", "top-center", "top-right",
    "middle-left", "middle-right",
    "bottom-left", "bottom-center", "bottom-right",
]
# Maximum "punch-in" zoom level (1.0 = no zoom, 2.0 = crop to half width/height)
ZOOM_MAX_LEVEL = float(os.getenv("ZOOM_MAX_LEVEL", "1.6"))

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
