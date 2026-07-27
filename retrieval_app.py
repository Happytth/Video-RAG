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

import os
import json
import argparse
import base64
from typing import List, Dict, Any, Optional
from PIL import Image

try:
    # pyrefly: ignore [missing-import]
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

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
        gemini_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-004",
        gemini_model: str = "gemini-2.0-flash"
    ):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be provided or set as environment variable.")
        
        self.client = genai.Client(api_key=api_key)
        
        # Dynamic fallback to developer API equivalent if text-embedding-004 is missing
        self.embedding_model = embedding_model
        if "text-embedding-004" in embedding_model:
            try:
                available = [m.name for m in self.client.models.list()]
                if "models/gemini-embedding-2" in available:
                    self.embedding_model = "models/gemini-embedding-2"
                elif "models/gemini-embedding-001" in available:
                    self.embedding_model = "models/gemini-embedding-001"
            except Exception:
                self.embedding_model = "models/gemini-embedding-2"
                
        self.gemini_model = gemini_model
        self.database = os.environ.get("NEO4J_DATABASE")
        self.groq_api_key = os.environ.get("GROQ_API_KEY")

    def close(self):
        self.driver.close()

    def embed_query(self, query: str) -> List[float]:
        """Embeds natural language user query using CLIP text model to search in visual vector space."""
        if CLIPModel is None or CLIPProcessor is None or torch is None:
            raise ImportError("transformers and torch packages are required to embed queries via CLIP.")
            
        global _clip_model, _clip_processor, _clip_device
        if "_clip_model" not in globals():
            print("[Embedder] Loading CLIP model for query embedding (openai/clip-vit-base-patch32)...")
            globals()["_clip_processor"] = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            globals()["_clip_model"] = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            globals()["_clip_model"] = globals()["_clip_model"].to(device)
            globals()["_clip_device"] = device

        device = globals()["_clip_device"]
        processor = globals()["_clip_processor"]
        model = globals()["_clip_model"]

        inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
        # Normalize features
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features[0].tolist()

    def vector_search_seed_nodes(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Executes Neo4j vector search using `db.index.vector.queryNodes` on `ses_node_vector_index`.
        Returns top-k matching seed nodes.
        """
        print(f"[Vector Search Debug] Connecting to database: '{self.database}'")
        with self.driver.session(database=self.database) as session:
            try:
                available_indexes = [r.data() for r in session.run("SHOW INDEXES")]
                print(f"[Vector Search Debug] Available indexes: {[idx['name'] for idx in available_indexes]}")
            except Exception as e:
                print(f"[Vector Search Debug] Could not fetch indexes: {e}")
                
        cypher = """
        CALL db.index.vector.queryNodes('ses_node_vector_index', $top_k, $query_embedding)
        YIELD node, score
        RETURN node.id AS id,
               node.node_type AS node_type,
               node.description AS description,
               node.timestamp AS timestamp,
               node.time_seconds AS time_seconds,
               node.objects AS objects,
               node.image_path AS image_path,
               score
        ORDER BY score DESC
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, top_k=top_k, query_embedding=query_embedding)
            seed_nodes = [record.data() for record in result]

        print(f"[Vector Search] Retrieved {len(seed_nodes)} seed nodes (top score: {seed_nodes[0]['score'] if seed_nodes else 0:.4f}).")
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
        Invokes Gemini Pro to synthesize a grounded answer citing exact timestamps and causal reasons,
        incorporating representative multimodal visual frames.
        """
        system_prompt = """
You are a Multimodal Causal Video RAG Answering Agent. Your role is to synthesize a natural language response to user queries by analyzing BOTH the temporal State-Event-State (SES) graph text context AND the accompanying video frame images.

CRITICAL ANSWERING RULES:
1. VISUAL IDENTIFICATION: You are expected and permitted to use your background visual knowledge to identify specific people, famous faces (e.g. Zayn Malik, One Direction members, etc.), brands, or objects present in the video frames. If the text context refers to a generic entity like "a person" but you can visually identify them in the corresponding image frame, identify them by name!
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

        # 1. Try Groq Answering first if GROQ_API_KEY is present and no images are passed (multimodal requires Gemini)
        if not images and self.groq_api_key and ChatGroq:
            print("[Answering] Attempting to generate grounded answer using Groq (llama-3.3-70b-versatile)...")
            try:
                llm = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    temperature=0.2,
                    api_key=self.groq_api_key
                )
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                res = llm.invoke(messages)
                return res.content
            except Exception as e:
                print(f"[Answering Warning] Groq failed: {e}. Falling back to Gemini...")

        # 2. Gemini Answering (supporting multimodal input)
        print(f"[Answering] Generating grounded answer using Gemini (sending {len(images) if images else 0} retrieved frames)...")
        
        contents = []
        if images:
            for item in images:
                if isinstance(item, tuple) and len(item) >= 2:
                    contents.append(item[0])  # Text label (e.g. [Video Frame at 00:10])
                    contents.append(item[1])  # PIL.Image
                else:
                    contents.append(item)
        contents.append(user_prompt)

        try:
            response = self.client.models.generate_content(
                model=self.gemini_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt
                )
            )
            return response.text
        except Exception as e:
            err_str = str(e).upper()
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "QUOTA" in err_str:
                print(f"[Fallback] Gemini rate limit hit ({e}). Routing to Groq Vision API...")
                
                if ChatGroq is None or HumanMessage is None:
                    raise ImportError("langchain-groq and langchain-core are required to run the Groq Vision fallback.")

                if not self.groq_api_key:
                    raise ValueError("GROQ_API_KEY environment variable is not set. Cannot run fallback.")
                
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
                                    print(f"[Fallback Warning] Failed to convert image to base64: {img_err}")
                
                # Execute Groq Qwen Vision API
                try:
                    llm = ChatGroq(
                        model="qwen/qwen3.6-27b",
                        temperature=0.2,
                        api_key=self.groq_api_key
                    )
                    res = llm.invoke([HumanMessage(content=message_content)])
                    return res.content
                except Exception as groq_err:
                    print(f"[Fallback Error] Groq Vision model failed: {groq_err}")
                    raise e
            else:
                raise e

    def query_video_rag(self, query: str, top_k_seeds: int = 3) -> Dict[str, Any]:
        """End-to-end execution of retrieval & grounded generation pipeline."""
        print(f"\n[RAG Pipeline] Processing Query: \"{query}\"")

        # Step 1: Embed query
        query_vector = self.embed_query(query)

        # Step 2: Vector Search seed nodes
        seed_nodes = self.vector_search_seed_nodes(query_vector, top_k=top_k_seeds)
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



        images = []
        for path in sorted(list(image_paths)):
            if os.path.exists(path):
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
            "image_paths": sorted(list(image_paths))
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3 & 4 Causal-Transient Video RAG Retrieval & Q&A")
    parser.add_argument("--query", type=str, required=True, help="User query string")
    parser.add_argument("--neo4j_uri", type=str, default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"), help="Neo4j Bolt URI")
    parser.add_argument("--neo4j_user", type=str, default=os.environ.get("NEO4J_USER", os.environ.get("NEO4J_USERNAME", "neo4j")), help="Neo4j username")
    parser.add_argument("--neo4j_password", type=str, default=os.environ.get("NEO4J_PASSWORD", "password"), help="Neo4j password")
    parser.add_argument("--gemini_model", type=str, default="gemini-2.0-flash", help="Gemini model for synthesis")

    args = parser.parse_args()

    retriever = CausalVideoRetriever(
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        gemini_model=args.gemini_model
    )

    try:
        result = retriever.query_video_rag(args.query)
        print("\n" + "=" * 60)
        print("RETRIEVED GRAPH CONTEXT:")
        print("=" * 60)
        print(result["graph_context"])
        print("\n" + "=" * 60)
        print("GEMINI GROUNDED ANSWER:")
        print("=" * 60)
        print(result["answer"])
    finally:
        retriever.close()
