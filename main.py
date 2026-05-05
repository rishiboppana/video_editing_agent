#!/usr/bin/env python3
"""
AI Video Highlight Generator
Usage:
    python main.py <video_path> [options]

Environment variables (all optional — see config.py for defaults):
    OLLAMA_MODEL            Ollama model name        (default: llama3.2)
    OLLAMA_BASE_URL         Ollama server URL         (default: http://localhost:11434)
    WHISPER_MODEL           Whisper model size        (default: base)
    MAX_HIGHLIGHT_DURATION  Max highlight length (s)  (default: 60)
    OUTPUT_DIR              Directory for outputs     (default: output/)
    MAX_RETRIES             Per-agent retry limit     (default: 3)
"""
import os
import sys

# ── Fix: PyTorch's bundled libiomp5 conflicts with macOS system OpenMP on CPU.
# Setting OMP_NUM_THREADS=1 BEFORE torch/whisper are imported prevents the
# segmentation fault that otherwise occurs during Whisper transcription.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json

from config import MAX_HIGHLIGHT_DURATION, OUTPUT_DIR
from orchestrator import OrchestratorAgent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a highlight reel from any video using local AI (Ollama + Whisper).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument(
        "-o", "--output",
        help="Output path for the highlight video (default: output/<name>_highlight.mp4)",
    )
    parser.add_argument(
        "-d", "--duration",
        type=int,
        default=MAX_HIGHLIGHT_DURATION,
        help=f"Max highlight duration in seconds (default: {MAX_HIGHLIGHT_DURATION})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full result as JSON (suppresses progress logs)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.video):
        print(f"Error: video file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not args.output:
        base = os.path.splitext(os.path.basename(args.video))[0]
        args.output = os.path.join(OUTPUT_DIR, f"{base}_highlight.mp4")

    orchestrator = OrchestratorAgent()

    try:
        result = orchestrator.run(
            video_path=args.video,
            output_path=args.output,
            max_duration=args.duration,
        )

        if args.json:
            # Strip the large nested pipeline blob for clean output
            clean = {k: v for k, v in result.items() if k != "pipeline"}
            print(json.dumps(clean, indent=2))
        else:
            print(f"\nHighlight video : {result['output_path']}")
            print(f"Duration        : {result['duration']}s")
            print(f"Clips           : {result['clips_count']}")
            print(f"Summary         : {result['summary']}")
            print(f"Topics          : {', '.join(result['topics'])}")
            print(f"Narrative       : {result['narrative']}")

    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\nPipeline error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
