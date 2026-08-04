"""
End-to-End Orchestration CLI for Causal-Transient Video RAG System (main.py)

Usage:
  1. Full Pipeline Run:
     python main.py --video_path sample.mp4 --query "Why was the apple missing at 02:40?"

  2. Query-only Run (if graph already built):
     python main.py --skip_ingest --query "What happened after the man entered the room?"
"""

import os
import argparse
try:
    # pyrefly: ignore [missing-import]
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from ingestion import run_ingestion
from graph_builder import run_graph_builder
from retrieval_app import CausalVideoRetriever


def main():
    parser = argparse.ArgumentParser(description="Causal-Transient Video RAG Pipeline")
    parser.add_argument("--video_path", type=str, help="Path to input .mp4 video file")
    parser.add_argument("--query", type=str, help="User natural language question about the video")
    parser.add_argument("--timeline_json", type=str, default="timeline.json", help="Path for timeline JSON file")
    
    # Ingestion args
    parser.add_argument("--fps", type=float, default=4.0, help="YOLO frame sampling FPS rate")
    parser.add_argument("--yolo_model", type=str, default="yolo11n.pt", help="YOLO model (yolo11n.pt, yolov8s-world.pt)")
    parser.add_argument("--whisper_model", type=str, default="base", help="Whisper model (tiny, base, small)")
    
    # Graph Builder args
    parser.add_argument("--neo4j_uri", type=str, default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"), help="Neo4j Bolt URI")
    parser.add_argument("--neo4j_user", type=str, default=os.environ.get("NEO4J_USER", os.environ.get("NEO4J_USERNAME", "neo4j")), help="Neo4j username")
    parser.add_argument("--neo4j_password", type=str, default=os.environ.get("NEO4J_PASSWORD", "password"), help="Neo4j password")
    
    # Pipeline execution controls
    parser.add_argument("--skip_ingest", action="store_true", help="Skip Stage 1 ingestion if timeline.json exists")
    parser.add_argument("--skip_graph", action="store_true", help="Skip Stage 2 graph building if Neo4j is already populated")

    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[WARNING] GEMINI_API_KEY environment variable is not set. Please set it before running graph building or retrieval.")

    # Stage 1: Ingestion
    if not args.skip_ingest:
        if not args.video_path:
            raise ValueError("--video_path is required unless --skip_ingest is specified.")
        run_ingestion(
            video_path=args.video_path,
            output_json=args.timeline_json,
            target_fps=args.fps,
            yolo_model=args.yolo_model,
            whisper_model=args.whisper_model
        )

    # Stage 2: Knowledge Graph Building
    if not args.skip_graph:
        run_graph_builder(
            timeline_json_path=args.timeline_json,
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password
        )

    # Stage 3 & 4: Retrieval & QA
    if args.query:
        print("\n" + "=" * 60)
        print("STAGE 3 & 4: RETRIEVAL & ANSWER SYNTHESIS")
        print("=" * 60)
        retriever = CausalVideoRetriever(
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password
        )
        try:
            res = retriever.query_video_rag(args.query)
            print("\n--- RETRIEVED GRAPH CONTEXT ---")
            print(res["graph_context"])
            print("\n--- GEMINI GROUNDED ANSWER ---")
            print(res["answer"])
        finally:
            retriever.close()


if __name__ == "__main__":
    main()
