"""
Stage 3 & 4: Bidirectional Retrieval & Grounded Generation (retrieval_app.py)

This module implements the retrieval and answer synthesis pipeline:
  1. Vector Search: Embeds user queries using Google text-embedding-004 and queries Neo4j's vector index
     to locate the most relevant seed node (State or Event).
  2. Robust Cypher Traversal (Omnidirectional + Causal Directional):
     Traverses 2 hops backward (to uncover root causes/preceding states) and 2 hops forward (to uncover effects),
     plus 1-hop omnidirectional neighbors. This handles cases whether vector search lands on cause or effect.
  3. Grounded Synthesis via Gemini Pro: Synthesizes a precise answer citing exact timestamps and causal steps.
"""

from __future__ import annotations

import sys
import os
import json
import argparse
import base64
from typing import List, Dict, Any, Optional

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

from PIL import Image

try:
    from google import genai
except ImportError:
    genai = None

try:
    # pyrefly: ignore [missing-import]
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from langchain_groq import ChatGroq
    from langchain_core.messages import SystemMessage, HumanMessage
except ImportError:
    ChatGroq = None
    SystemMessage = None
    HumanMessage = None

try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
except ImportError:
    CLIPProcessor = None
    CLIPModel = None
    torch = None

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None


import re

def parse_timestamp_query(query: str) -> Optional[float]:
    """Parses explicit timestamp requests from user query (e.g. '7th second', 'frame at 00:07', '51st sec', '12s')."""
    match_ord = re.search(r'(\d+)(?:st|nd|rd|th)?\s*(?:sec|second)', query, re.IGNORECASE)
    if match_ord:
        return float(match_ord.group(1))

    match_mmss = re.search(r'(\d{1,2}):(\d{2})', query)
    if match_mmss:
        mins = int(match_mmss.group(1))
        secs = int(match_mmss.group(2))
        return float(mins * 60 + secs)

    match_sec = re.search(r'at\s*(\d+)\s*s', query, re.IGNORECASE)
    if match_sec:
        return float(match_sec.group(1))

    return None


def local_image_to_base64(image_path: str) -> str:
    """Encodes a local image file to a base64 string."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


class CausalVideoRetriever:
    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        groq_api_key: Optional[str] = None,
        groq_model: str = "qwen/qwen3.6-27b"
    ):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        self.groq_model = groq_model
        self.database = os.environ.get("NEO4J_DATABASE")

    def close(self):
        self.driver.close()

    def embed_query(self, query: str) -> List[float]:
        """Embeds natural language user query into 3072d text vector space using Google gemini-embedding-2 or fallback."""
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if gemini_key and genai is not None:
            try:
                client = genai.Client(api_key=gemini_key)
                for model_name in ["models/gemini-embedding-2", "text-embedding-004", "models/gemini-embedding-001"]:
                    try:
                        res = client.models.embed_content(
                            model=model_name,
                            contents=query
                        )
                        emb_vals = res.embeddings[0].values
                        if len(emb_vals) < 3072:
                            emb_vals = emb_vals + [0.0] * (3072 - len(emb_vals))
                        elif len(emb_vals) > 3072:
                            emb_vals = emb_vals[:3072]
                        return emb_vals
                    except Exception:
                        continue
            except Exception as e:
                log_to_terminal(f"[Embedder Warning] Gemini API query embedding failed: {e}")

        # Deterministic 3072d text embedding generator matching graph builder vector space
        import hashlib
        import math
        tokens = query.lower().split()
        vector = [0.0] * 3072
        for token in tokens:
            h_val = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h_val % 3072
            val = ((h_val >> 16) % 1000) / 1000.0
            vector[idx] += val
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector

    def vector_search_seed_nodes(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Executes Neo4j vector search using `db.index.vector.queryNodes` on `ses_node_vector_index`.
        Deduplicates results by image_path so returned seed nodes are strictly from 3 distinct timeframes.
        """
        cypher = """
        CALL db.index.vector.queryNodes('ses_node_vector_index', 15, $query_embedding)
        YIELD node, score
        WITH node, score, node.image_path AS img_path
        ORDER BY score DESC
        WITH img_path, head(collect(node)) AS unique_node, max(score) AS max_score
        ORDER BY max_score DESC
        LIMIT $top_k
        RETURN unique_node.id AS id,
               unique_node.node_type AS node_type,
               unique_node.description AS description,
               unique_node.timestamp AS timestamp,
               unique_node.time_seconds AS time_seconds,
               unique_node.objects AS objects,
               unique_node.image_path AS image_path,
               unique_node.dense_caption AS dense_caption,
               max_score AS score
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, top_k=top_k, query_embedding=query_embedding)
            seed_nodes = [record.data() for record in result]

        print(f"[Vector Search] Retrieved {len(seed_nodes)} distinct seed nodes (top score: {seed_nodes[0]['score'] if seed_nodes else 0:.4f}).")
        return seed_nodes

    def traverse_causal_graph(self, seed_node_ids: List[str]) -> Dict[str, Any]:
        """
        Executes robust bidirectional Cypher graph traversal from seed nodes:
          - Backward traversal (1..2 hops): Uncovers preceding states & root causes.
          - Forward traversal (1..2 hops): Uncovers succeeding effects & state changes.
          - Omnidirectional neighborhood (1 hop): Captures co-occurring DURING relations.
        """
        cypher_traversal = """
        MATCH (seed:Entity) WHERE seed.id IN $seed_ids

        // 1. Backward Causes/Preceding States (2 hops)
        OPTIONAL MATCH cause_path = (cause:Entity)-[r1:PRECEDES|CAUSES|DURING*1..2]->(seed)

        // 2. Forward Effects/Succeeding States (2 hops)
        OPTIONAL MATCH effect_path = (seed)-[r2:PRECEDES|CAUSES|DURING*1..2]->(effect:Entity)

        // 3. Direct Omnidirectional Neighbors (1 hop)
        OPTIONAL MATCH (seed)-[r_omni]-(neighbor:Entity)

        WITH seed,
             collect(distinct cause) AS causes,
             collect(distinct effect) AS effects,
             collect(distinct neighbor) AS neighbors,
             collect(distinct cause_path) AS c_paths,
             collect(distinct effect_path) AS e_paths

        RETURN seed, causes, effects, neighbors, c_paths, e_paths
        """

        # Also pull explicit triples for detailed relationship reporting with exact edge directionality
        cypher_triples = """
        MATCH (seed:Entity) WHERE seed.id IN $seed_ids
        MATCH (a:Entity)-[r:PRECEDES|CAUSES|DURING]->(b:Entity)
        WHERE a.id IN $seed_ids OR b.id IN $seed_ids
           OR EXISTS { (seed)<-[:PRECEDES|CAUSES|DURING*1..2]-(a) }
           OR EXISTS { (seed)-[:PRECEDES|CAUSES|DURING*1..2]->(a) }
           OR EXISTS { (seed)<-[:PRECEDES|CAUSES|DURING*1..2]-(b) }
           OR EXISTS { (seed)-[:PRECEDES|CAUSES|DURING*1..2]->(b) }
        RETURN DISTINCT a.id AS source_id,
                        a.node_type AS source_type,
                        a.description AS source_desc,
                        a.timestamp AS source_time,
                        a.time_seconds AS source_time_seconds,
                        a.image_path AS source_image_path,
                        type(r) AS relation_type,
                        coalesce(r.explanation, '') AS explanation,
                        b.id AS target_id,
                        b.node_type AS target_type,
                        b.description AS target_desc,
                        b.timestamp AS target_time,
                        b.image_path AS target_image_path
        ORDER BY source_time_seconds ASC
        """

        with self.driver.session(database=self.database) as session:
            res_traversal = session.run(cypher_traversal, seed_ids=seed_node_ids)
            traversal_records = [rec.data() for rec in res_traversal]

            res_triples = session.run(cypher_triples, seed_ids=seed_node_ids)
            triples = [rec.data() for rec in res_triples]

        return {
            "traversal_records": traversal_records,
            "triples": triples
        }

    def format_graph_context_string(self, seed_nodes: List[Dict[str, Any]], graph_data: Dict[str, Any]) -> str:
        """Formats seed nodes and causal graph triples into clear markdown for LLM prompt context."""
        context_lines = ["### SEED NODES MATCHED FROM VECTOR DB:"]
        for s in seed_nodes:
            context_lines.append(
                f"- [{s['node_type']}] ID: {s['id']} | Timestamp: {s['timestamp']} | "
                f"Description: \"{s['description']}\" (Vector Relevance Score: {s['score']:.4f})"
            )

        context_lines.append("\n### TRAVERSED STATE-EVENT-STATE CAUSAL CHAINS:")
        triples = graph_data.get("triples", [])
        if not triples:
            context_lines.append("No explicit causal edges connected to seed node.")
        else:
            for t in triples:
                context_lines.append(
                    f"- ({t['source_time']}) [{t['source_type']}: \"{t['source_desc']}\"] "
                    f"==[{t['relation_type']}]==> "
                    f"({t['target_time']}) [{t['target_type']}: \"{t['target_desc']}\"]"
                    + (f" (Reason: {t['explanation']})" if t['explanation'] else "")
                )

        return "\n".join(context_lines)

    def generate_answer(self, query: str, seed_nodes: List[Dict[str, Any]], graph_context: str, images: List[Any] = None) -> str:
        """
        Invokes Groq Qwen Vision API to synthesize a grounded answer citing exact timestamps and causal reasons,
        incorporating representative multimodal visual frames.
        """
        system_prompt = """
You are a Multimodal Causal Video RAG Answering Agent. Your role is to synthesize a natural language response to user queries by analyzing BOTH the temporal State-Event-State (SES) graph text context AND the accompanying video frame images.

CRITICAL ANSWERING RULES:
1. VISUAL IDENTIFICATION: You are expected and permitted to use your visual knowledge to identify specific people, famous figures, logos, brands, or objects present in the video frames. If the text context refers to a generic entity like "a person" but you can visually identify them in the corresponding image frame, identify them by name or description.
2. CHRONOLOGICAL GROUNDING: Explicitly link your visual observations to the timestamps of the frames where they appear (e.g. [00:10], [00:15]).
3. TEMPORAL & CAUSAL REASONING: Base the sequence of actions and events on the provided Knowledge Graph context and visual timeline. Do not invent actions or outcomes not shown or documented.
4. If a query asks about a specific person or object, verify if they are visible in any of the passed images or mentioned in the text context.
"""

        user_prompt = f"""
USER QUERY:
"{query}"

RETRIEVED KNOWLEDGE GRAPH CONTEXT:
{graph_context}

Provide a clear, timestamp-grounded causal explanation answering the user query:
"""

        if ChatGroq is None or HumanMessage is None:
            raise ImportError("langchain-groq and langchain-core are required to run the Groq Vision model.")

        keys_str = os.environ.get("GROQ_API_KEYS", "") or os.environ.get("GROQ_API_KEY", "") or (self.groq_api_key or "")
        available_keys = [k.strip() for k in keys_str.replace(";", ",").split(",") if k.strip()]
        if not available_keys:
            return "⚠️ GROQ_API_KEY environment variable is not set. Please set GROQ_API_KEY in .env."

        # Format a HumanMessage content containing text instructions and base64 encoded image URLs
        message_content = []
        groq_prompt = f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\nUSER PROMPT & RETRIEVED CONTEXT:\n{user_prompt}"
        message_content.append({"type": "text", "text": groq_prompt})

        if images:
            # Slice to max 2 images to stay safely within Groq Vision model constraints (max 3 images)
            fallback_images = images[:2]
            for item in fallback_images:
                if isinstance(item, tuple) and len(item) == 3:
                    label, _, img_path = item
                    if os.path.exists(img_path):
                        try:
                            b64_str = local_image_to_base64(img_path)
                            message_content.append({"type": "text", "text": label})
                            message_content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_str}"
                                }
                            })
                        except Exception as img_err:
                            print(f"[Answering Warning] Failed to convert image to base64: {img_err}")

        print(f"[Answering] Generating grounded answer using Groq Vision API ('{self.groq_model}')...")
        for key_idx, key in enumerate(available_keys):
            try:
                llm = ChatGroq(
                    model=self.groq_model,
                    temperature=0.2,
                    api_key=key,
                    timeout=60.0
                )
                res = llm.invoke([HumanMessage(content=message_content)])
                raw_answer = str(res.content) if res.content else ""
                import re
                if "</think>" in raw_answer:
                    after_think = raw_answer.split("</think>")[-1].strip()
                    if after_think:
                        return after_think
                cleaned = re.sub(r'</?think>', '', raw_answer, flags=re.IGNORECASE).strip()
                return cleaned if cleaned else raw_answer.strip()
            except Exception as groq_err:
                if ("429" in str(groq_err) or "rate limit" in str(groq_err).lower()) and key_idx < len(available_keys) - 1:
                    print(f"[Groq Answering Key Rotation] Key #{key_idx + 1} rate limited. Rotating to Key #{key_idx + 2}...")
                    continue
                print(f"[Answering Error] Groq Vision model failed: {groq_err}")
                return f"⚠️ Groq Vision model failed: {groq_err}"

    def search_nodes_by_timestamp(self, target_sec: float, tolerance: float = 5.0) -> List[Dict[str, Any]]:
        """Directly queries Neo4j for nodes matching a target timestamp in seconds."""
        query_cypher = """
        MATCH (n:Entity)
        WHERE n.time_seconds IS NOT NULL AND abs(n.time_seconds - $target_sec) <= $tolerance
        RETURN n.id AS id,
               n.node_type AS node_type,
               n.description AS description,
               n.timestamp AS timestamp,
               n.time_seconds AS time_seconds,
               n.objects AS objects,
               n.image_path AS image_path,
               1.0 AS score
        ORDER BY abs(n.time_seconds - $target_sec) ASC
        LIMIT 3
        """
        with self.driver.session(database=self.database) as session:
            res = session.run(query_cypher, target_sec=target_sec, tolerance=tolerance)
            nodes = [r.data() for r in res]
            return nodes

    def is_person_counting_query(self, query: str) -> bool:
        """Determines whether user query is asking for person counts or unique individual identities."""
        q_lower = query.lower()
        keywords = [
            "how many people", "how many person", "how many persons",
            "how many individual", "how many individuals", "count of people",
            "count of person", "number of people", "number of person",
            "how many different people", "how many different individuals",
            "how many men", "how many women", "how many human", "total people",
            "total person", "total persons", "who is in", "list of people"
        ]
        return any(k in q_lower for k in keywords)

    def execute_person_counting_query(self) -> List[Dict[str, Any]]:
        """Executes exact Cypher query on Neo4j (:Person) nodes to return deterministic count analytics."""
        cypher = """
        MATCH (p:Person)
        OPTIONAL MATCH (p)-[:APPEARS_IN]->(s:Entity)
        RETURN p.id AS person_id,
               collect(DISTINCT s.timestamp) AS timestamps_appeared,
               count(DISTINCT s) AS total_scenes
        ORDER BY p.id ASC
        """
        with self.driver.session(database=self.database) as session:
            res = session.run(cypher)
            results = [record.data() for record in res]
        return results

    def query_video_rag(self, query: str, top_k_seeds: int = 3) -> Dict[str, Any]:
        """End-to-end execution of retrieval & grounded generation pipeline."""
        print(f"\n[RAG Pipeline] Processing Query: \"{query}\"")

        # Intent Router Branch: Person Re-ID Analytics Query
        if self.is_person_counting_query(query):
            print(f"[Person Analytics Router] Detected person-counting query. Querying Neo4j (:Person) graph nodes...")
            person_results = self.execute_person_counting_query()
            total_people = len(person_results)

            summary_lines = [f"### DETERMINISTIC PERSON TRACKING ANALYTICS (Neo4j Graph):"]
            summary_lines.append(f"- **Total Unique Tracked Individuals**: {total_people}")
            for p in person_results:
                ts = ", ".join(p["timestamps_appeared"]) if p["timestamps_appeared"] else "N/A"
                summary_lines.append(f"- **{p['person_id']}**: Appeared in {p['total_scenes']} scene(s) at timestamps `[{ts}]`.")

            graph_context = "\n".join(summary_lines)
            answer = self.generate_answer(query, [], graph_context, images=None)
            return {
                "query": query,
                "answer": answer,
                "seed_nodes": [],
                "graph_context": graph_context,
                "image_paths": [],
                "person_analytics": person_results
            }

        # Check if query contains an explicit timestamp lookup request (e.g. "7th second", "00:07", "51st sec")
        target_sec = parse_timestamp_query(query)
        if target_sec is not None:
            print(f"[Timestamp Lookup] Detected explicit target timestamp: {target_sec} seconds. Querying Neo4j temporal index...")
            seed_nodes = self.search_nodes_by_timestamp(target_sec)
        else:
            # Step 1: Embed query
            query_vector = self.embed_query(query)
            # Step 2: Vector Search seed nodes
            seed_nodes = self.vector_search_seed_nodes(query_vector, top_k=top_k_seeds)

        for idx, node in enumerate(seed_nodes, 1):
            print(f"  [{idx}] Time: {node.get('timestamp')} | Frame: {node.get('image_path')} | Score: {node.get('score'):.4f}")

        if not seed_nodes:
            return {
                "query": query,
                "answer": "No relevant nodes found in video knowledge graph vector index.",
                "seed_nodes": [],
                "graph_context": ""
            }

        seed_ids = [s["id"] for s in seed_nodes]

        # Step 3: Traversal
        graph_data = self.traverse_causal_graph(seed_ids)
        graph_context = self.format_graph_context_string(seed_nodes, graph_data)

        # Step 4: Grounded Answer Synthesis with Multimodal Frame-Passing
        image_paths = set()
        for node in seed_nodes:
            if node.get("image_path"):
                image_paths.add(node["image_path"])

        for t in graph_data.get("triples", []):
            if t.get("source_image_path"):
                image_paths.add(t["source_image_path"])
            if t.get("target_image_path"):
                image_paths.add(t["target_image_path"])

        # Score and rank candidate images using CLIP visual-text similarity against query_vector
        scored_images = []
        for path in image_paths:
            if os.path.exists(path):
                try:
                    # Filter out solid white/overexposed flash frames (e.g. end credit text screens)
                    img_check = Image.open(path).convert("L")
                    import numpy as np
                    arr = np.array(img_check)
                    if arr.std() < 12.0 or arr.mean() > 240:
                        scored_images.append((-1.0, path))
                        continue

                    from ingestion import generate_clip_image_embedding
                    img_emb = generate_clip_image_embedding(path)
                    score = sum(q * m for q, m in zip(query_vector, img_emb))
                    scored_images.append((score, path))
                except Exception as emb_err:
                    scored_images.append((0.0, path))

        # Sort candidate image paths by CLIP similarity score descending
        scored_images.sort(key=lambda x: x[0], reverse=True)
        sorted_paths = [p for _, p in scored_images]

        images = []
        for path in sorted_paths:
            try:
                img = Image.open(path)
                img.load()
                
                # Parse timestamp from filename to pass as context
                basename = os.path.basename(path)
                try:
                    seconds = int(basename.replace("frame_", "").replace(".jpg", ""))
                    mins = seconds // 60
                    secs = seconds % 60
                    label = f"[Video Frame at {mins:02d}:{secs:02d}]"
                except Exception:
                    label = f"[Video Frame: {basename}]"
                
                images.append((label, img, path))
            except Exception as e:
                print(f"[Answering Warning] Failed to load image '{path}': {e}")

        answer = self.generate_answer(query, seed_nodes, graph_context, images=images)

        return {
            "query": query,
            "answer": answer,
            "seed_nodes": seed_nodes,
            "graph_context": graph_context,
            "image_paths": sorted_paths
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3 & 4 Causal-Transient Video RAG Retrieval & Q&A")
    parser.add_argument("--query", type=str, required=True, help="User query string")
    parser.add_argument("--neo4j_uri", type=str, default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"), help="Neo4j Bolt URI")
    parser.add_argument("--neo4j_user", type=str, default=os.environ.get("NEO4J_USER", os.environ.get("NEO4J_USERNAME", "neo4j")), help="Neo4j username")
    parser.add_argument("--neo4j_password", type=str, default=os.environ.get("NEO4J_PASSWORD", "password"), help="Neo4j password")
    parser.add_argument("--groq_model", type=str, default="qwen/qwen3.6-27b", help="Groq model for synthesis")

    args = parser.parse_args()

    retriever = CausalVideoRetriever(
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        groq_model=args.groq_model
    )

    try:
        result = retriever.query_video_rag(args.query)
        print("\n" + "=" * 60)
        print("RETRIEVED GRAPH CONTEXT:")
        print("=" * 60)
        print(result["graph_context"])
        print("\n" + "=" * 60)
        print("GROQ GROUNDED ANSWER:")
        print("=" * 60)
        print(result["answer"])
    finally:
        retriever.close()
