#!/usr/bin/env python3
"""
Video Semantic Search
=====================
Index videos once, then search them by natural language.

Commands:

  python search_videos.py index <folder>
      Walk folder recursively, detect scenes, compute pHash, describe with
      LLaVA, embed descriptions. Skips already-indexed videos.

  python search_videos.py index <folder> --force
      Re-index all videos, even already-indexed ones.

  python search_videos.py index <folder> --no-vision
      Skip LLaVA descriptions. Embeds scene position text instead.
      Faster but less accurate.

  python search_videos.py search "<query>"
      Search all indexed scenes and print the top-5 results.

  python search_videos.py search "<query>" --top 10 --min-score 0.4
      Return up to 10 results with similarity >= 0.4.

  python search_videos.py stats
      Print how many videos and scenes are indexed.

Examples:

  python search_videos.py index videos/
  python search_videos.py search "couple walking down the aisle"
  python search_videos.py search "excited crowd reactions" --top 3
"""
import argparse
import json
import logging
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)

from search.indexer import index_folder
from search.searcher import format_results, search
from search.store import DEFAULT_INDEX_PATH, index_stats, load_index


def cmd_index(args):
    if not os.path.exists(args.folder):
        print(f"Error: folder not found: {args.folder}", file=sys.stderr)
        sys.exit(1)

    use_vision = not args.no_vision
    if not use_vision:
        print("Vision disabled — using scene-position embeddings as fallback.")

    print(f"Indexing '{args.folder}' ...")
    index = index_folder(
        folder=args.folder,
        index_path=args.index,
        force=args.force,
        vision=use_vision,
    )
    stats = index_stats(index)
    print(
        f"\nIndex updated:"
        f"\n  Videos  : {stats['videos']}"
        f"\n  Scenes  : {stats['scenes']}"
        f"\n  Described: {stats['described']}"
        f"\n  Embedded : {stats['embedded']}"
        f"\n  Saved to : {args.index}"
    )


def cmd_search(args):
    if not os.path.exists(args.index):
        print(
            f"No index found at '{args.index}'. "
            "Run 'python search_videos.py index <folder>' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    results = search(
        query=args.query,
        index_path=args.index,
        top_k=args.top,
        min_score=args.min_score,
    )

    if args.json:
        clean = [{k: v for k, v in r.items() if k != "phash"} for r in results]
        print(json.dumps(clean, indent=2))
    else:
        print(format_results(results, args.query))


def cmd_stats(args):
    if not os.path.exists(args.index):
        print("No index found. Run indexing first.")
        return
    index = load_index(args.index)
    stats = index_stats(index)
    print(
        f"Index: {args.index}\n"
        f"  Videos indexed : {stats['videos']}\n"
        f"  Total scenes   : {stats['scenes']}\n"
        f"  Described (LLaVA): {stats['described']}\n"
        f"  Embedded         : {stats['embedded']}\n"
    )
    if index.get("indexed_videos"):
        print("Indexed videos:")
        for path, ts in index["indexed_videos"].items():
            n = sum(1 for s in index["scenes"] if s["video"] == path)
            print(f"  [{n} scenes]  {path}  (indexed {ts[:10]})")


def main():
    parser = argparse.ArgumentParser(
        description="Video semantic search — index videos and search by natural language.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--index",
        default=DEFAULT_INDEX_PATH,
        help=f"Path to the index file (default: {DEFAULT_INDEX_PATH})",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # index command
    p_index = sub.add_parser("index", help="Index videos in a folder")
    p_index.add_argument("folder", help="Folder to walk recursively for videos")
    p_index.add_argument("--force", action="store_true", help="Re-index already-indexed videos")
    p_index.add_argument("--no-vision", action="store_true", help="Skip LLaVA frame description")

    # search command
    p_search = sub.add_parser("search", help="Search indexed scenes")
    p_search.add_argument("query", help="Natural language search query")
    p_search.add_argument("--top", type=int, default=5, help="Number of results to return")
    p_search.add_argument("--min-score", type=float, default=0.0, help="Minimum similarity score")
    p_search.add_argument("--json", action="store_true", help="Output results as JSON")

    # stats command
    sub.add_parser("stats", help="Show index statistics")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "stats":
        cmd_stats(args)


if __name__ == "__main__":
    main()
