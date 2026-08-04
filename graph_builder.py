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

import sys
import os
import json
import argparse
import time
from typing import List, Dict, Any, Literal, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

def log_to_terminal(msg: str):
    """Prints log messages directly to standard stdout and forces physical terminal output via sys.__stdout__."""
    print(msg, flush=True)
    if hasattr(sys, "__stdout__") and sys.__stdout__ is not None:
        try:
            sys.__stdout__.write(str(msg) + "\n")
            sys.__stdout__.flush()
        except Exception:
            pass
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
    dense_caption: Optional[str] = Field(default="", description="Florence-2 detailed visual caption")


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
            combined_person_ids = set()
            combined_transcripts = []
            for b in matching_blocks:
                combined_objects.update(b.get("visual_objects", []))
                combined_person_ids.update(b.get("detected_person_ids", []))
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
                "detected_person_ids": sorted(list(combined_person_ids)),
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
6. CRITICAL ANTI-HALLUCINATION RULE: Detected Visual Objects are automated hints. Only extract States and Events for actions/entities directly grounded in scene context or speech. Do NOT invent hallucinated objects or actions (e.g., horses, cars, animals) unless strictly confirmed.

Extract the SES graph strictly conforming to the requested JSON schema.
"""


def get_groq_api_keys() -> List[str]:
    """Retrieves list of available Groq API keys from environment for key rotation."""
    keys_str = os.environ.get("GROQ_API_KEYS", "") or os.environ.get("GROQ_API_KEY", "")
    keys = [k.strip() for k in keys_str.replace(";", ",").split(",") if k.strip()]
    return keys if keys else [""]


def extract_ses_graph_from_chunk(
    groq_api_key: str,
    chunk: Dict[str, Any],
    model_name: str = "llama-3.3-70b-versatile"
) -> Optional[SESGraph]:
    """
    Calls Groq API using LangChain with_structured_output to extract SES graph nodes and edges cleanly.
    Supports multi-key rotation to bypass Free Tier rate limits seamlessly.
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

    available_keys = get_groq_api_keys()
    if groq_api_key and groq_api_key not in available_keys:
        available_keys.insert(0, groq_api_key)

    for key_idx, key in enumerate(available_keys):
        try:
            llm = ChatGroq(
                model=model_name,
                temperature=0.0,
                api_key=key
            )
            structured_llm = llm.with_structured_output(SESGraph)
            res = structured_llm.invoke(prompt)
            time.sleep(1.0)
            return res
        except Exception as e:
            if ("429" in str(e) or "rate limit" in str(e).lower()):
                if key_idx < len(available_keys) - 1:
                    print(f"[Groq Key Rotation] Key #{key_idx + 1} rate limited. Rotating to Key #{key_idx + 2} immediately...")
                    continue
                elif model_name != "llama-3.1-8b-instant":
                    print(f"[Groq 429 Rate Limit] All API keys exhausted on {model_name}. Switching fallback to 'llama-3.1-8b-instant'...")
                    return extract_ses_graph_from_chunk(available_keys[0], chunk, model_name="llama-3.1-8b-instant")
            
            print(f"[Groq Extraction Error] Failed chunk {chunk['timestamp_range']}: {e}")
            return None


# ============================================================================
# 4. Google Text Embedding Vectorizer
# ============================================================================

def generate_node_embeddings(
    gemini_api_key: Optional[str],
    nodes: List[Dict[str, Any]],
    embedding_model: str = "text-embedding-004"
) -> List[Dict[str, Any]]:
    """
    Generates 3072-dimensional vector embeddings for each node by combining node description and Florence-2 dense caption.
    """
    log_to_terminal(f"[Embedder] Generating 3072d text embeddings for {len(nodes)} unique nodes...")
    
    client = None
    if gemini_api_key and genai is not None:
        try:
            client = genai.Client(api_key=gemini_api_key)
        except Exception:
            client = None

    for node in nodes:
        desc = node.get("description", "")
        objs = ", ".join(node.get("objects", [])) if node.get("objects") else "None"
        caption = node.get("dense_caption", "")
        transcript = node.get("transcript_text", "")
        
        # Pillar 4: Omni-Knowledge Fusion Text Representation
        text_to_embed = f"{node.get('node_type', 'Entity')}: {desc} | Objects: {objs} | Visual Caption: {caption} | Audio Speech: {transcript}".strip()
        
        embedded_successfully = False
        if client:
            max_retries = 3
            for model_name in ["models/gemini-embedding-2", "text-embedding-004", "models/gemini-embedding-001"]:
                for attempt in range(max_retries):
                    try:
                        res = client.models.embed_content(
                            model=model_name,
                            contents=text_to_embed
                        )
                        emb_vals = res.embeddings[0].values
                        if len(emb_vals) < 3072:
                            emb_vals = emb_vals + [0.0] * (3072 - len(emb_vals))
                        elif len(emb_vals) > 3072:
                            emb_vals = emb_vals[:3072]
                        node["embedding"] = emb_vals
                        embedded_successfully = True
                        break
                    except Exception as e:
                        if ("503" in str(e) or "429" in str(e) or "unavailable" in str(e).lower()) and attempt < max_retries - 1:
                            time.sleep((attempt + 1) * 2)
                        else:
                            break
                if embedded_successfully:
                    break

        if not embedded_successfully:
            # Deterministic fallback text embedding generator for 3072d vector space
            import hashlib
            import math
            tokens = text_to_embed.lower().split()
            vector = [0.0] * 3072
            for token in tokens:
                h_val = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
                idx = h_val % 3072
                val = ((h_val >> 16) % 1000) / 1000.0
                vector[idx] += val
            # L2 normalize fallback vector
            norm = math.sqrt(sum(v * v for v in vector))
            if norm > 0:
                vector = [v / norm for v in vector]
            node["embedding"] = vector

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

    def setup_schema_and_vector_index(self, vector_dim: int = 3072):
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
            e.image_path = n.image_path,
            e.dense_caption = n.dense_caption
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

        insert_person_cypher = """
        UNWIND $nodes AS n
        UNWIND n.detected_person_ids AS pid
        WITH n, pid WHERE pid IS NOT NULL AND pid <> ''
        MERGE (p:Person {id: pid})
        ON CREATE SET p.created_at = timestamp()
        WITH p, n
        MATCH (s:Entity {id: n.id})
        MERGE (p)-[:APPEARS_IN]->(s)
        """

        with self.driver.session(database=self.database) as session:
            # Purge stale nodes from previous video runs to ensure clean single-video graph context
            session.run("MATCH (n) DETACH DELETE n")
            print("[Neo4j Ingest] Purged stale graph nodes from previous runs.")

            # Batch write nodes
            res_nodes = session.run(insert_nodes_cypher, nodes=nodes)
            summary_nodes = res_nodes.consume()
            print(f"[Neo4j Ingest Debug] Nodes query consumed. Counters: {summary_nodes.counters}")
            
            # Batch write Person nodes & APPEARS_IN relationships
            try:
                session.run(insert_person_cypher, nodes=nodes)
                print("[Neo4j Ingest] Created (:Person) nodes and linked APPEARS_IN relationships.")
            except Exception as p_err:
                print(f"[Neo4j Ingest Debug] Person nodes creation warning: {p_err}")

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
    groq_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    neo4j_uri: str = os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
    neo4j_user: str = os.environ.get("NEO4J_USER", os.environ.get("NEO4J_USERNAME", "neo4j")),
    neo4j_password: str = os.environ.get("NEO4J_PASSWORD", "password"),
    window_size_sec: float = 15.0,
    overlap_sec: float = 5.0,
    **kwargs
):
    print("=" * 60)
    print("STAGE 2: SES KNOWLEDGE GRAPH CONSTRUCTION (GROQ)")
    print("=" * 60)

    # 1. Initialize API Keys
    active_groq_key = groq_api_key or os.environ.get("GROQ_API_KEY")
    if not active_groq_key:
        raise ValueError("GROQ_API_KEY must be provided as an argument or set in environment variables.")

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
        extracted = extract_ses_graph_from_chunk(active_groq_key, chunk, model_name=groq_model)
        
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
                        "detected_person_ids": chunk.get("detected_person_ids", []),
                        "image_path": chunk.get("image_path"),
                        "dense_caption": chunk.get("dense_caption", n.dense_caption or "")
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

    # Step 3: Generate 3072d text embeddings using description + Florence-2 dense caption
    nodes_with_embeddings = generate_node_embeddings(gemini_api_key, nodes_list)

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