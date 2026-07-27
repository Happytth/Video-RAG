"""
Unit & Integration Verification Test for Causal-Transient Video RAG Pipeline
"""

import json
from graph_builder import chunk_timeline_sliding_window, Node, Edge, SESGraph
from retrieval_app import CausalVideoRetriever

def test_sliding_window_chunker():
    # Mock timeline with 5s granularity across 30s total duration
    mock_timeline = [
        {"window_start": 0.0, "window_end": 5.0, "visual_objects": ["apple", "table"], "transcript_text": "Look at the apple."},
        {"window_start": 5.0, "window_end": 10.0, "visual_objects": ["apple", "hand"], "transcript_text": "A hand approaches."},
        {"window_start": 10.0, "window_end": 15.0, "visual_objects": ["hand", "apple"], "transcript_text": "Picking up the apple."},
        {"window_start": 15.0, "window_end": 20.0, "visual_objects": ["hand", "apple"], "transcript_text": "Moving apple to mouth."},
        {"window_start": 20.0, "window_end": 25.0, "visual_objects": ["apple"], "transcript_text": "Eating the apple."},
        {"window_start": 25.0, "window_end": 30.0, "visual_objects": [], "transcript_text": "Apple is gone."}
    ]

    chunks = chunk_timeline_sliding_window(mock_timeline, window_size_sec=15.0, overlap_sec=5.0)

    # Validate chunk bounds
    assert len(chunks) > 1, "Should create multiple overlapping chunks"
    assert chunks[0]["chunk_start"] == 0.0 and chunks[0]["chunk_end"] == 15.0
    assert chunks[1]["chunk_start"] == 10.0 and chunks[1]["chunk_end"] == 25.0
    print(f"[PASS] Sliding window chunker created {len(chunks)} overlapping windows as expected.")

def test_pydantic_schemas():
    node1 = Node(
        id="state_apple_table_0s",
        type="State",
        description="Apple is on table",
        timestamp="00:00",
        image_path="./saved_frames/frame_0.jpg"
    )
    node2 = Node(
        id="event_grab_apple_10s",
        type="Event",
        description="Hand grabs apple",
        timestamp="00:10",
        image_path=None
    )
    edge = Edge(
        source_id="state_apple_table_0s",
        target_id="event_grab_apple_10s",
        relationship="PRECEDES"
    )
    res = SESGraph(nodes=[node1, node2], edges=[edge])
    
    dumped = json.loads(res.model_dump_json())
    assert len(dumped["nodes"]) == 2
    assert len(dumped["edges"]) == 1
    assert dumped["edges"][0]["relationship"] == "PRECEDES"
    print("[PASS] Pydantic structured output schemas validated successfully.")

if __name__ == "__main__":
    print("Running pipeline verification tests...")
    test_sliding_window_chunker()
    test_pydantic_schemas()
    print("All unit tests passed cleanly!")
