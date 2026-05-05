"""
Legacy tool-calling agent — kept for reference.

The new multi-agent pipeline is in main.py:
    python main.py <video_path> [--duration 60] [--output output/highlight.mp4]

This file now proxies to the new system for backwards compatibility.
"""
from main import main

if __name__ == "__main__":
    print(
        "Note: agent.py is deprecated. "
        "Use `python main.py <video>` for the full multi-agent pipeline.\n"
    )
    main()
