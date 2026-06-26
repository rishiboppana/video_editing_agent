# Video Editing Agent

A multi-agent pipeline that turns any video into a highlight reel using local AI — Whisper for speech, LLaVA for vision, Ollama for reasoning. No cloud APIs, no per-call cost, everything runs on your machine or a free Colab GPU.

```bash
python main.py videos/Concert.mp4 --duration 30 \
  --style "fast-paced hype reel with crowd reactions" \
  --video-type "live concert"
```

---

## How it works

```
video.mp4
    │
    ▼
┌─────────────────────┐   Whisper (speech) + ffmpeg (scene cuts) + LLaVA (vision)
│  TranscriberAgent    │   → speech segments, visual scenes, focus position per scene
└──────────┬───────────┘
           ▼
┌─────────────────────┐   Ollama LLM + nomic-embed-text embeddings
│  ExplainerAgent      │   → summary, tone, per-segment importance score
└──────────┬───────────┘
           ▼
┌─────────────────────┐   Ollama LLM
│  HighlighterAgent    │   → selected clips + zoom/focus decision per clip
└──────────┬───────────┘
           ▼
┌─────────────────────┐   ffmpeg
│  EditorAgent         │   → cuts, zooms, joins clips into the final video
└──────────┬───────────┘
           ▼
    output/<name>_highlight.mp4
```

An **Orchestrator** sits above all four agents. After every agent runs it validates the output structurally, and for the two LLM-based agents (Explainer, Highlighter) it also asks the LLM to review its own output and retries with targeted feedback if not satisfactory — up to 3 attempts per step.

---

## Agents

| Agent | Uses LLM? | What it does |
|---|---|---|
| `TranscriberAgent` | No (Whisper + ffmpeg + LLaVA) | Transcribes speech, detects scene cuts, describes each scene in rich detail (people, objects, setting, actions, emotions, energy, crowd, focus position) |
| `ExplainerAgent` | Yes (Ollama + embeddings) | Builds a synchronized speech+visual timeline, scores every segment's importance, blends the LLM score with embedding similarity to the user's style |
| `HighlighterAgent` | Yes (Ollama) | Selects which segments make the final cut based on content match (not just duration filling), merges adjacent clips, decides zoom/focus per clip |
| `EditorAgent` | No (ffmpeg) | Cuts each clip, applies the zoom/crop filter, joins everything into the final video |

---

## Features

### Speech + Vision understanding
- Whisper transcribes speech with per-segment confidence (`no_speech_prob`, `avg_logprob`)
- ffmpeg scene-cut detection at multiple sensitivity thresholds (0.15 → 0.08 → adaptive chunks fallback)
- LLaVA describes every scene in structured detail: people, objects, setting, actions, emotions, energy level, crowd reaction
- Hallucination filtering — Whisper segments the model itself isn't confident about are dropped automatically

### Visual-dominant video handling
Videos with little or no speech (weddings, music, b-roll) are auto-detected. When vision is unavailable too, the pipeline falls back to time-proportional scene scoring instead of analyzing garbage transcript, so it never silently produces nonsense.

### Style-aware highlight selection
```bash
--style "excited crowd reactions and peak energy moments"
--video-type "live concert"
```
Both are free text — no fixed categories. The Explainer embeds the style query and every scene description with `nomic-embed-text`, then blends cosine similarity with the LLM's importance score:
```
importance = 0.6 × LLM_score + 0.4 × embedding_similarity
```
If no style is given, the agent defaults to finding the best overall moments from the video's own summary.

### Auto-zoom / focus
LLaVA tags each scene with a focus position on a 3×3 grid (`top-left` … `center` … `bottom-right`). The Highlighter decides a zoom level (1.0–1.6x) per selected clip, and the Editor applies it as an ffmpeg crop+scale filter — correctly handling rotated/portrait video dimensions.

### Robust JSON handling
LLM responses are repaired through several stages before giving up:
- Strip markdown fences
- Strip trailing commas, numeric unit suffixes (`4.5s` → `4.5`)
- Bracket-stack-based truncation recovery (handles responses cut off mid-generation by token limits)
- Normalize importance scores that come back as text (`"high"` → `0.8`) instead of failing

### Hardware-aware
- Whisper auto-selects CUDA → MPS → CPU
- Automatic fallback from MPS to CPU if a `SparseMPS` backend error occurs mid-run (no wasted retries)
- Vision calls back off after repeated timeouts so a slow CPU doesn't stall the whole pipeline

---

## Setup

```bash
pip install -r requirements.txt
brew install ffmpeg        # or: apt install ffmpeg

ollama pull llama3.2           # reasoning (explainer + highlighter)
ollama pull llava              # vision (scene descriptions)
ollama pull nomic-embed-text   # embeddings (style matching)

ollama serve                   # leave running in a separate terminal
```

---

## Usage

```bash
# Basic
python main.py videos/Concert.mp4 --duration 30

# With style + video type (both free text)
python main.py videos/Wedding.mp4 --duration 20 \
  --video-type "wedding ceremony" \
  --style "best moments of the couple"

# Custom output path
python main.py videos/Concert.mp4 -o exports/cut.mp4

# JSON result instead of logs
python main.py videos/Concert.mp4 --json
```

### CLI options

| Flag | Default | Description |
|---|---|---|
| `video` | — | Path to the input video (required) |
| `-o, --output` | `output/<name>_highlight.mp4` | Output path |
| `-d, --duration` | `60` | Max highlight duration (seconds) |
| `--style` | none | Free-text preference for which moments to pick |
| `--video-type` | none | Free-text description of the video content |
| `--json` | off | Print structured JSON result, suppress progress logs |

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.2` | LLM for explanation + highlighting |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_TIMEOUT` | `300` | Seconds to wait for one LLM response |
| `WHISPER_MODEL` | `base` | Whisper model size (`tiny`/`base`/`small`/`medium`) |
| `MAX_HIGHLIGHT_DURATION` | `60` | Default highlight length if `-d` not passed |
| `OUTPUT_DIR` | `output` | Default output directory |
| `MAX_RETRIES` | `3` | Per-agent retry limit |
| `ENABLE_VISION` | `true` | Toggle LLaVA scene descriptions |
| `VISION_MODEL` | `llava` | Vision model name |
| `VISION_TIMEOUT` | `180` | Seconds to wait for one LLaVA call |
| `VISION_MULTI_FRAME_MIN` | `15.0` | Min scene length (s) to sample 3 frames instead of 1 |
| `VISION_MAX_CONSECUTIVE_FAILURES` | `2` | Stop trying vision after this many timeouts in a row |
| `MAX_VISUAL_DESCRIPTIONS` | `20` | Max scenes described per video (caps long videos) |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model for style matching |
| `EMBED_WEIGHT` | `0.4` | Weight of embedding similarity vs LLM score |
| `SCENE_THRESHOLD` | `0.15` | ffmpeg scene-cut sensitivity (lower = more cuts) |
| `ZOOM_MAX_LEVEL` | `1.6` | Maximum zoom magnification for auto-crop |

---

## Running on Google Colab (recommended for vision)

LLaVA needs a GPU to run in practical time — on CPU it can take 3–5 minutes per frame. Colab's free T4 GPU runs it in ~3 seconds.

```
https://colab.research.google.com/github/rishiboppana/video_editing_agent/blob/main/colab_setup.ipynb
```

1. `Runtime → Change runtime type → T4 GPU`
2. Run all cells top to bottom (installs Ollama, pulls models, clones repo, uploads your video)
3. Download the finished highlight from the last cell

If running locally without a GPU, disable vision to avoid timeouts:
```bash
ENABLE_VISION=false python main.py videos/Concert.mp4 --duration 30
```
The pipeline still works using speech + scene-position fallback scoring, just with less visual context.

---

## Project structure

```
video_editing_agent/
├── main.py                  # CLI entry point
├── orchestrator.py          # Runs all 4 agents, validates + retries
├── config.py                # All settings, env-var driven
├── agents/
│   ├── base_agent.py        # Shared LLM call + JSON repair logic
│   ├── transcriber_agent.py # Whisper + ffmpeg scenes + LLaVA vision
│   ├── explainer_agent.py   # Importance scoring + style embeddings
│   ├── highlighter_agent.py # Clip selection + zoom decisions
│   └── editor_agent.py      # ffmpeg cut/zoom/join
├── colab_setup.ipynb        # One-click GPU pipeline on Colab
├── requirements.txt
├── videos/                  # Input videos (not tracked)
└── output/                  # Generated highlight reels (not tracked)
```
