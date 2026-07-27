"""
Stage 2: Building the State-Event-State (SES) Knowledge Graph (graph_builder.py)

This module reads the merged timeline from Stage 1 and constructs a State-Event-State (SES)
Knowledge Graph in Neo4j using Google Gemini Pro with Pydantic Structured Outputs and text-embedding-004.

Key Architectural Features:
  1. Sliding Window Chunking: Uses 15-second windows with 5-second overlap to prevent splitting
     transient events across chunk boundaries.
  2. SES Extraction Logic (Inline Explanation):
     - State Nodes: Represent static situations/conditions holding true at a given timestamp 
       (e.g., "Apple is resting on the table at 01:40").
     - Event Nodes: Represent active physical actions or state-changes occurring at a timestamp
       (e.g., "Man reaches out and grabs apple at 01:45").
     - Relationships: Connect States and Events via directed edges:
         (State A) -[:PRECEDES]-> (Event) -[:CAUSES]-> (State B)
         (State) -[:DURING]-> (Event)
  3. Embedding: Vectorizes node descriptions using Google's text-embedding-004 (768 dimensions).
  4. Neo4j Vector Index & Graph Storage: Sets up uniqueness constraints and native Neo4j vector index
     for hybrid graph retrieval.
"""

from __future__ import annotations

import os
import json
import argparse
import time
from typing import List, Dict, Any, Literal, Optional
from pydantic import BaseModel, Field

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import PydanticOutputParser
except ImportError:
    ChatGroq = None
    PromptTemplate = None
    PydanticOutputParser = None

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ============================================================================
# 1. Pydantic Schemas for Groq Structured Output
# ============================================================================

class Node(BaseModel):
    id: str = Field(description="Unique string identifier")
    type: str = Field(description="'State' or 'Event'")
    description: str = Field(description="Detailed entity description")
    timestamp: str = Field(description="Format MM:SS")
    image_path: Optional[str] = Field(default=None, description="Path to representative video frame image")
    clip_embedding: Optional[list[float]] = Field(default=None, description="512-dimensional CLIP embedding vector")


class Edge(BaseModel):
    source_id: str
    target_id: str
    relationship: str = Field(description="'PRECEDES', 'CAUSES', or 'DURING'")


class SESGraph(BaseModel):
    nodes: list[Node]
    edges: list[Edge]


# ============================================================================
# 2. Timeline Sliding Window Chunker
# ============================================================================

def chunk_timeline_sliding_window(
    timeline: List[Dict[str, Any]],
    window_size_sec: float = 15.0,
    overlap_sec: float = 5.0
) -> List[Dict[str, Any]]:
    """
    Chunks the chronological timeline into overlapping windows (e.g. 15s window, 5s overlap).
    Guarantees that events spanning 5s-granularity boundaries are not cut in half.
    """
    if not timeline:
        return []

    max_time = timeline[-1]["window_end"]
    step_sec = max(1.0, window_size_sec - overlap_sec)

    chunks = []
    curr_start = 0.0

    while curr_start < max_time:
        curr_end = curr_start + window_size_sec
        
        # Gather all timeline blocks overlapping with [curr_start, curr_end]
        matching_blocks = [
            b for b in timeline 
            if not (b["window_end"] <= curr_start or b["window_start"] >= curr_end)
        ]

        if matching_blocks:
            combined_objects = set()
            combined_transcripts = []
            for b in matching_blocks:
                combined_objects.update(b.get("visual_objects", []))
                if b.get("transcript_text"):
                    combined_transcripts.append(b["transcript_text"])

            chunk_image_path = None
            chunk_clip_embedding = None
            best_block = None
            for b in matching_blocks:
                if b.get("image_path"):
                    if best_block is None or len(b.get("visual_objects", [])) > len(best_block.get("visual_objects", [])):
                        best_block = b
            if best_block:
                chunk_image_path = best_block["image_path"]
                chunk_clip_embedding = best_block.get("clip_embedding")

            chunks.append({
                "chunk_start": curr_start,
                "chunk_end": curr_end,
                "timestamp_range": f"{format_sec(curr_start)} - {format_sec(curr_end)}",
                "visual_objects": sorted(list(combined_objects)),
                "transcript_text": " ".join(combined_transcripts).strip(),
                "image_path": chunk_image_path,
                "clip_embedding": chunk_clip_embedding
            })

        curr_start += step_sec

    print(f"[Chunker] Created {len(chunks)} overlapping windows ({window_size_sec}s window, {overlap_sec}s overlap).")
    return chunks


def format_sec(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def parse_timestamp_to_seconds(ts_str: str) -> float:
    """
    Parses a timestamp string of format MM:SS or HH:MM:SS or ranges like 'MM:SS - MM:SS' to total seconds.
    """
    if not ts_str:
        return 0.0
    if "-" in ts_str:
        # Take the start of the range
        ts_str = ts_str.split("-")[0].strip()
    
    parts = ts_str.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        elif len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        elif len(parts) >= 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    except Exception:
        pass
    return 0.0


# ============================================================================
# 3. Groq SES Graph Extractor
# ============================================================================

EXTRACT_SES_PROMPT_TEMPLATE = """
You are an expert Causal Knowledge Graph Extractor analyzing a video timeline segment.

CONTEXT TIMELINE SEGMENT:
Timestamp Range: {timestamp_range}
Detected Visual Objects: {visual_objects}
Spoken Audio Transcript: "{transcript_text}"

INSTRUCTIONS FOR STATE-EVENT-STATE (SES) EXTRACTION:
1. Extract 'State' nodes representing static physical situations or conditions (e.g. "Apple is on the table", "Hand is empty").
2. Extract 'Event' nodes representing actions, movements, or state changes (e.g. "Person grabs the apple", "Apple is removed from table").
3. Connect State nodes to Event nodes using directed relationships:
   - (State A) -[PRECEDES]-> (Event): State A exists immediately before Event happens.
   - (Event) -[CAUSES]-> (State B): Event directly produces or results in State B.
   - (State) -[DURING]-> (Event): State co-occurs alongside the Event.
4. Ensure every node has a clear, factual description, timestamp, approximate time in seconds, and list of involved objects.
5. Create unique, descriptive string IDs for each node (e.g. "state_apple_on_table_00m10s", "event_pick_apple_00m14s").

Extract the SES graph strictly conforming to the requested JSON schema.
"""


def extract_ses_graph_from_chunk(
    groq_api_key: str,
    chunk: Dict[str, Any],
    model_name: str = "llama-3.3-70b-versatile"
) -> Optional[SESGraph]:
    """
    Calls Groq API using LangChain with_structured_output to extract SES graph nodes and edges cleanly.
    """
    if ChatGroq is None:
        raise ImportError(
            "langchain-groq is not installed or failed to import in the active Python environment. "
            "Please ensure you run Streamlit using the virtual environment."
        )

    prompt = EXTRACT_SES_PROMPT_TEMPLATE.format(
        timestamp_range=chunk["timestamp_range"],
        visual_objects=", ".join(chunk["visual_objects"]) if chunk["visual_objects"] else "None detected",
        transcript_text=chunk["transcript_text"] if chunk["transcript_text"] else "No speech"
    )

    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Enforce strict 0.0 temperature for JSON parsing stability
            llm = ChatGroq(
                model=model_name,
                temperature=0.0,
                api_key=groq_api_key
            )
            
            structured_llm = llm.with_structured_output(SESGraph)
            res = structured_llm.invoke(prompt)
            
            # Artificial sleep to avoid Groq Free Tier API rate limits (30 RPM)
            time.sleep(2.0)
            
            return res
        except Exception as e:
            if ("429" in str(e) or "rate limit" in str(e).lower()) and attempt < max_retries - 1:
                sleep_time = (attempt + 1) * 30
                print(f"[Groq 429 Rate Limit] Retrying in {sleep_time}s... (Error: {e})")
                time.sleep(sleep_time)
            else:
                print(f"[Groq Extraction Error] Failed chunk {chunk['timestamp_range']}: {e}")
                return None


# ============================================================================
# 4. Google Text Embedding Vectorizer
# ============================================================================

def generate_node_embeddings(
    client: genai.Client,
    nodes: List[Dict[str, Any]],
    embedding_model: str = "text-embedding-004"
) -> List[Dict[str, Any]]:
    """
    Generates 768-dimensional vector embeddings for each node description.
    """
    resolved_model = embedding_model
    # Dynamic fallback to developer API equivalent if text-embedding-004 is missing
    if "text-embedding-004" in embedding_model:
        try:
            available = [m.name for m in client.models.list()]
            if "models/gemini-embedding-2" in available:
                resolved_model = "models/gemini-embedding-2"
            elif "models/gemini-embedding-001" in available:
                resolved_model = "models/gemini-embedding-001"
        except Exception:
            resolved_model = "models/gemini-embedding-2"

    print(f"[Embedder] Generating embeddings using '{resolved_model}' for {len(nodes)} unique nodes...")
    for node in nodes:
        text_to_embed = f"{node['node_type']}: {node['description']} (Timestamp: {node['timestamp']})"
        embedded_successfully = False
        max_retries = 3
        for attempt in range(max_retries):
            try:
                res = client.models.embed_content(
                    model=resolved_model,
                    contents=text_to_embed
                )
                node["embedding"] = res.embeddings[0].values
                embedded_successfully = True
                break
            except Exception as e:
                if ("503" in str(e) or "429" in str(e) or "unavailable" in str(e).lower()) and attempt < max_retries - 1:
                    sleep_time = (attempt + 1) * 3
                    print(f"[Embedder Warning] Gemini API transient error ({e}). Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    print(f"[Embedding Error] Failed node '{node['id']}': {e}")
                    break

        if not embedded_successfully:
            # Fallback zero vector matching current resolved model dim
            fallback_dim = 3072 if "embedding-2" in resolved_model else 768
            node["embedding"] = [0.0] * fallback_dim

    return nodes


# ============================================================================
# 5. Neo4j Batch Writer & Vector Index Config
# ============================================================================

class Neo4jSESWriter:
    def __init__(self, uri: str, auth: tuple):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.database = os.environ.get("NEO4J_DATABASE")

    def close(self):
        self.driver.close()

    def setup_schema_and_vector_index(self, vector_dim: int = 768):
        """Creates uniqueness constraints and vector index on Neo4j database."""
        with self.driver.session(database=self.database) as session:
            print("[Neo4j Schema] Setting up uniqueness constraints...")
            session.run("CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;").consume()
            
            # Check existing index configuration
            res = session.run("SHOW INDEXES YIELD name, type, options")
            existing_dim = None
            for r in res:
                if r["name"] == "ses_node_vector_index" and r["type"] == "VECTOR":
                    existing_dim = r.get("options", {}).get("indexConfig", {}).get("vector.dimensions")
                    break
            
            if existing_dim is not None and existing_dim != vector_dim:
                print(f"[Neo4j Vector Index] Found existing vector index with dimension {existing_dim}. Dropping to recreate with {vector_dim}...")
                try:
                    session.run("DROP INDEX ses_node_vector_index IF EXISTS;").consume()
                    print("[Neo4j Vector Index] Waiting 8 seconds for index deletion to propagate...")
                    time.sleep(8)
                except Exception as e:
                    print(f"[Neo4j Index Warning] Could not drop index: {e}")
            elif existing_dim is not None:
                print(f"[Neo4j Vector Index] Vector index 'ses_node_vector_index' already exists with correct dimensions ({vector_dim}). Skipping creation.")
                return
            
            print(f"[Neo4j Vector Index] Setting up vector index 'ses_node_vector_index' (dims: {vector_dim})...")
            create_index_cypher = f"""
            CREATE VECTOR INDEX ses_node_vector_index IF NOT EXISTS
            FOR (n:Entity) ON (n.embedding)
            OPTIONS {{indexConfig: {{
              `vector.similarity_function`: 'cosine',
              `vector.dimensions`: {vector_dim}
            }}}}
            """
            session.run(create_index_cypher).consume()
            print("[Neo4j Schema] Schema setup completed.")

    def insert_graph_data(
        self,
        nodes: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ):
        """Batch inserts nodes and edges into Neo4j."""
        print(f"[Neo4j Ingest] Writing {len(nodes)} nodes and {len(relationships)} relationships...")
        
        # Avoid APOC by using conditional FOREACH label setting in pure Cypher
        insert_nodes_cypher = """
        UNWIND $nodes AS n
        MERGE (e:Entity {id: n.id})
        SET e.node_type = n.node_type,
            e.description = n.description,
            e.timestamp = n.timestamp,
            e.time_seconds = n.time_seconds,
            e.objects = n.objects,
            e.embedding = n.embedding,
            e.image_path = n.image_path
        FOREACH (ignore IN CASE WHEN n.node_type = 'State' THEN [1] ELSE [] END |
            SET e:State)
        FOREACH (ignore IN CASE WHEN n.node_type = 'Event' THEN [1] ELSE [] END |
            SET e:Event)
        RETURN count(e)
        """

        insert_rels_cypher = """
        UNWIND $rels AS r
        MATCH (src:Entity {id: r.source_id})
        MATCH (tgt:Entity {id: r.target_id})
        CALL apoc.create.relationship(src, r.relation_type, {explanation: r.explanation}, tgt) YIELD rel
        RETURN count(rel)
        """

        # Fallback relationship creation without APOC if APOC is not present (standard for Aura)
        insert_rels_fallback_cypher = """
        UNWIND $rels AS r
        MATCH (src:Entity {id: r.source_id})
        MATCH (tgt:Entity {id: r.target_id})
        FOREACH (ignore IN CASE WHEN r.relation_type = 'PRECEDES' THEN [1] ELSE [] END |
            MERGE (src)-[:PRECEDES {explanation: r.explanation}]->(tgt))
        FOREACH (ignore IN CASE WHEN r.relation_type = 'CAUSES' THEN [1] ELSE [] END |
            MERGE (src)-[:CAUSES {explanation: r.explanation}]->(tgt))
        FOREACH (ignore IN CASE WHEN r.relation_type = 'DURING' THEN [1] ELSE [] END |
            MERGE (src)-[:DURING {explanation: r.explanation}]->(tgt))
        """

        with self.driver.session(database=self.database) as session:
            # Batch write nodes
            res_nodes = session.run(insert_nodes_cypher, nodes=nodes)
            summary_nodes = res_nodes.consume()
            print(f"[Neo4j Ingest Debug] Nodes query consumed. Counters: {summary_nodes.counters}")
            
            # Batch write relationships with APOC check
            try:
                res_rels = session.run(insert_rels_cypher, rels=relationships)
                summary_rels = res_rels.consume()
                print(f"[Neo4j Ingest Debug] Rels query consumed (APOC). Counters: {summary_rels.counters}")
            except Exception as e:
                print(f"[Neo4j Ingest Debug] APOC rels failed: {e}. Trying fallback...")
                # Fallback to standard Cypher if APOC plugin is absent
                res_fallback = session.run(insert_rels_fallback_cypher, rels=relationships)
                summary_fallback = res_fallback.consume()
                print(f"[Neo4j Ingest Debug] Fallback rels consumed. Counters: {summary_fallback.counters}")

        print("[Neo4j Ingest] Graph batch insertion finished successfully.")


def run_graph_builder(
    timeline_json_path: str = "timeline.json",
    gemini_api_key: Optional[str] = None,
    neo4j_uri: str = os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
    neo4j_user: str = os.environ.get("NEO4J_USER", os.environ.get("NEO4J_USERNAME", "neo4j")),
    neo4j_password: str = os.environ.get("NEO4J_PASSWORD", "password"),
    window_size_sec: float = 15.0,
    overlap_sec: float = 5.0,
    gemini_model: str = "gemini-2.0-flash",
    embedding_model: str = "text-embedding-004"
):
    print("=" * 60)
    print("STAGE 2: SES KNOWLEDGE GRAPH CONSTRUCTION (GROQ)")
    print("=" * 60)

    # 1. Initialize API Keys
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY must be provided as an environment variable in your .env file.")

    g_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
    if not g_api_key:
        raise ValueError("GEMINI_API_KEY must be provided or set as environment variable to compute embeddings.")

    client = genai.Client(api_key=g_api_key)

    with open(timeline_json_path, "r", encoding="utf-8") as f:
        timeline = json.load(f)

    # Calculate duration
    total_duration_sec = timeline[-1]["window_end"] if timeline else 0.0
    total_duration_min = total_duration_sec / 60.0

    # Step 1: Chunk timeline into overlapping windows
    chunks = chunk_timeline_sliding_window(
        timeline,
        window_size_sec=window_size_sec,
        overlap_sec=overlap_sec
    )
    total_chunks = len(chunks)

    # Dynamic Model Selection
    if total_duration_min <= 20.0 or total_chunks <= 80:
        groq_model = "llama-3.3-70b-versatile"
    else:
        groq_model = "llama-3.1-8b-instant"

    print(f"[Model Routing] Video Duration: {total_duration_min:.2f} mins, Chunks: {total_chunks}.")
    print(f"[Model Routing] Selected Groq model: '{groq_model}'")

    # Step 2: Extract SES Graph Triples per chunk via Groq Structured Output
    all_nodes_dict: Dict[str, Dict[str, Any]] = {}
    all_relationships: List[Dict[str, Any]] = []
    seen_rels = set()

    print(f"[Groq SES] Extracting graph triples from {len(chunks)} chunks...")
    for idx, chunk in enumerate(chunks, 1):
        print(f" -> Processing Chunk {idx}/{len(chunks)} ({chunk['timestamp_range']})...")
        extracted = extract_ses_graph_from_chunk(groq_api_key, chunk, model_name=groq_model)
        
        if extracted:
            # Map Pydantic Node schema to Neo4j ingestion format
            for n in extracted.nodes:
                if n.id not in all_nodes_dict:
                    # Resolve entities list from description and objects
                    objects_list = []
                    # Simple heuristic: find objects in the chunk's visual_objects present in the node description
                    for obj in chunk.get("visual_objects", []):
                        if obj.lower() in n.description.lower() and obj not in objects_list:
                            objects_list.append(obj)

                    all_nodes_dict[n.id] = {
                        "id": n.id,
                        "node_type": n.type,  # Map type -> node_type
                        "description": n.description,
                        "timestamp": n.timestamp,
                        "time_seconds": parse_timestamp_to_seconds(n.timestamp),
                        "objects": objects_list,
                        "image_path": chunk.get("image_path"),
                        "embedding": chunk.get("clip_embedding") or ([0.0] * 512)
                    }
            
            # Map Pydantic Edge schema to Neo4j relationship format
            for r in extracted.edges:
                rel_key = (r.source_id, r.target_id, r.relationship)
                if rel_key not in seen_rels:
                    seen_rels.add(rel_key)
                    all_relationships.append({
                        "source_id": r.source_id,
                        "target_id": r.target_id,
                        "relation_type": r.relationship,  # Map relationship -> relation_type
                        "explanation": f"Extracted chronological/causal relation: {r.relationship}"
                    })

    nodes_list = list(all_nodes_dict.values())
    print(f"[Deduplication] Extracted {len(nodes_list)} unique nodes and {len(all_relationships)} unique relationships.")

    if not nodes_list:
        print("[Warning] No nodes were extracted. Skipping database write.")
        return

    # Step 3: Skip Gemini description embedding, use pre-generated visual CLIP embeddings
    nodes_with_embeddings = nodes_list

    # Step 4: Write to Neo4j
    vector_dim = len(nodes_with_embeddings[0]["embedding"]) if nodes_with_embeddings else 768
    neo4j_writer = Neo4jSESWriter(uri=neo4j_uri, auth=(neo4j_user, neo4j_password))
    try:
        neo4j_writer.setup_schema_and_vector_index(vector_dim=vector_dim)
        neo4j_writer.insert_graph_data(nodes_with_embeddings, all_relationships)
        # Verify inserted nodes
        with neo4j_writer.driver.session(database=neo4j_writer.database) as test_sess:
            cnt = test_sess.run("MATCH (n) RETURN count(n)").single()[0]
            print(f"[Neo4j Verification] Verified inserted nodes count in database '{neo4j_writer.database}': {cnt}")
            labels_cnt = test_sess.run("MATCH (n) RETURN labels(n), count(n)").values()
            print(f"[Neo4j Verification] Label counts: {labels_cnt}")
    finally:
        neo4j_writer.close()

    print(f"[Stage 2 Complete] State-Event-State graph successfully populated in Neo4j.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 SES Knowledge Graph Builder")
    parser.add_argument("--timeline_json", type=str, default="timeline.json", help="Path to input timeline JSON")
    parser.add_argument("--neo4j_uri", type=str, default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"), help="Neo4j Bolt URI")
    parser.add_argument("--neo4j_user", type=str, default=os.environ.get("NEO4J_USER", os.environ.get("NEO4J_USERNAME", "neo4j")), help="Neo4j username")
    parser.add_argument("--neo4j_password", type=str, default=os.environ.get("NEO4J_PASSWORD", "password"), help="Neo4j password")
    parser.add_argument("--window_size", type=float, default=15.0, help="Sliding window size in seconds")
    parser.add_argument("--overlap", type=float, default=5.0, help="Sliding window overlap in seconds")
    parser.add_argument("--gemini_model", type=str, default="gemini-2.0-flash", help="Gemini model for SES extraction")

    args = parser.parse_args()

    run_graph_builder(
        timeline_json_path=args.timeline_json,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        window_size_sec=args.window_size,
        overlap_sec=args.overlap,
        gemini_model=args.gemini_model
    )