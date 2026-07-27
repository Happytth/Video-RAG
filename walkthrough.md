# Walkthrough: Causal-Transient Video RAG System

We have fully reviewed, optimized, and implemented the **Causal-Transient Video RAG System**. This architecture overcomes the limitations of standard semantic video RAG by catching split-second visual object detections ("transient") and mapping cause-and-effect across time ("causal") using a **State-Event-State (SES) Knowledge Graph** in **Neo4j** powered by **Gemini 2.0 Pro / Flash** and **Google text-embedding-004**.

---

## 📁 Repository Structure & Modules

| File | Role / Stage | Optimization Details |
|---|---|---|
| [`requirements.txt`](file:///c:/Users/SOUBHAGYA%20NAYAK/OneDrive/Desktop/New%20folder/requirements.txt) | Dependencies | Package versions (`google-genai`, `neo4j`, `ultralytics`, `openai-whisper`, `opencv-python`, `pydantic`) |
| [`docker-compose.yml`](file:///c:/Users/SOUBHAGYA%20NAYAK/OneDrive/Desktop/New%20folder/docker-compose.yml) | Neo4j Container | Local Neo4j v5.12 container config with APOC and native vector index support |
| [`ingestion.py`](file:///c:/Users/SOUBHAGYA%20NAYAK/OneDrive/Desktop/New%20folder/ingestion.py) | Stage 1: Dual-Stream Ingestion | 3–5 FPS OpenCV sampling + YOLO / YOLO-World open-vocabulary mode + Whisper audio transcript merger |
| [`graph_builder.py`](file:///c:/Users/SOUBHAGYA%20NAYAK/OneDrive/Desktop/New%20folder/graph_builder.py) | Stage 2: SES Graph Construction | **15s window / 5s overlap sliding chunker** + **Gemini Pydantic Structured Output** + `text-embedding-004` + Neo4j vector index |
| [`retrieval_app.py`](file:///c:/Users/SOUBHAGYA%20NAYAK/OneDrive/Desktop/New%20folder/retrieval_app.py) | Stage 3 & 4: Retrieval & QA | Neo4j Vector Search + **Directed Cypher traversal with bidirectional subqueries** + Gemini grounded synthesis |
| [`main.py`](file:///c:/Users/SOUBHAGYA%20NAYAK/OneDrive/Desktop/New%20folder/main.py) | Pipeline Orchestrator | Safe CLI entrypoint with graceful optional dependency handling |
| [`test_pipeline.py`](file:///c:/Users/SOUBHAGYA%20NAYAK/OneDrive/Desktop/New%20folder/test_pipeline.py) | Unit Test Suite | Offline test suite verifying sliding window chunker logic and Pydantic schema serialization |

---

## 🛠️ Key Architectural Optimizations Implemented

### 1. Sliding Window Timeline Chunking (`graph_builder.py`)
To prevent splitting events that span across fixed chunk boundaries (e.g. action starting at `00:14` and ending at `00:18`), `graph_builder.py` uses a **15-second sliding window with a 5-second overlap**:
- Chunk 1: `00:00 - 00:15`
- Chunk 2: `00:10 - 00:25`
- Chunk 3: `00:20 - 00:35`

Extracted State/Event nodes and directed relationships across overlapping windows are automatically deduplicated by unique keys before writing to Neo4j.

### 2. Open-Vocabulary YOLO & Speed Configuration (`ingestion.py`)
Supports standard `yolo11n.pt` for maximum inference speed on standard COCO classes, and seamlessly toggles to zero-shot models like `yolov8s-world.pt` / `YOLO-World` when custom open-vocabulary classes (`--custom_classes`) are provided.

### 3. Gemini Pydantic Structured Output (`graph_builder.py`)
Leverages the official `google-genai` SDK by passing Pydantic models directly into `response_schema` in `GenerateContentConfig`:
```python
class NodeExtraction(BaseModel):
    id: str
    node_type: Literal["State", "Event"]
    description: str
    timestamp: str
    time_seconds: float
    objects: List[str]

class RelationshipExtraction(BaseModel):
    source_id: str
    target_id: str
    relation_type: Literal["PRECEDES", "CAUSES", "DURING"]
    explanation: str

class GraphExtractionResult(BaseModel):
    nodes: List[NodeExtraction]
    relationships: List[RelationshipExtraction]
```

### 4. Directed Cypher Traversal & Directionality (`retrieval_app.py`)
To eliminate relationship direction flipping during graph traversal, `retrieval_app.py` enforces directed edge matching `(a:Entity)-[r:PRECEDES|CAUSES|DURING]->(b:Entity)` paired with bidirectional `EXISTS` subqueries:
```cypher
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
                type(r) AS relation_type,
                coalesce(r.explanation, '') AS explanation,
                b.id AS target_id,
                b.node_type AS target_type,
                b.description AS target_desc,
                b.timestamp AS target_time
ORDER BY a.time_seconds ASC
```

---

## 🧪 Verification Results

1. **Unit Test Suite (`test_pipeline.py`)**:
   - Verified that `chunk_timeline_sliding_window` generates correctly overlapping 15s/5s chunks.
   - Verified that `GraphExtractionResult` Pydantic schemas serialize and validate JSON accurately.
   - Output: `All unit tests passed cleanly!`

### Dependency Integration
* Successfully installed `langchain-groq` and `langchain-core` inside the python virtual environment using `uv pip`.

### Schemas Migration in [graph_builder.py](file:///c:/Users/SOUBHAGYA%20NAYAK/OneDrive/Desktop/New%20folder/graph_builder.py)
* Replaced the older extraction schemas with the strictly requested schemas:
  * `Node` (fields: `id`, `type`, `description`, `timestamp`)
  * `Edge` (fields: `source_id`, `target_id`, `relationship`)
  * `SESGraph` (fields: `nodes`, `edges`)

### Answering Migration in [retrieval_app.py](file:///c:/Users/SOUBHAGYA%20NAYAK/OneDrive/Desktop/New%20folder/retrieval_app.py)
* Integrated `langchain_groq` into the answer synthesis phase (`generate_answer` method) of the RAG pipeline.
* When `GROQ_API_KEY` is present in the environment, the retriever will **generate the grounded answer using `llama-3.3-70b-versatile` on Groq**, bypassing the Gemini generation rate limits entirely.
* If the Groq key is absent or a call fails, it automatically falls back to Gemini.

### Dynamic Model Routing (Stage 2)
* Calculated video duration and chunk count inside `run_graph_builder`.
* Implemented the routing rules:
  * **Duration <= 20 minutes (<= 80 chunks):** Selects `llama-3.3-70b-versatile`.
  * **Duration > 20 minutes (> 80 chunks):** Selects `llama-3.1-8b-instant`.

---

## 2. Test Verification Results

We successfully resolved the Cypher syntax error (`42N44`) caused by sorting by `a.time_seconds` when `DISTINCT` was present in the `RETURN` projection of `traverse_causal_graph` in [retrieval_app.py](file:///c:/Users/SOUBHAGYA%20NAYAK/OneDrive/Desktop/New%20folder/retrieval_app.py). By projecting the variable as `source_time_seconds` and ordering on the projected variable, the query compiles and executes cleanly.

### Verification Run
We ran:
```powershell
python main.py --skip_ingest --query "What objects are present in the video and what is the sequence of events?"
```

**Output:**
```
============================================================
STAGE 2: SES KNOWLEDGE GRAPH CONSTRUCTION (GROQ)
============================================================
[Chunker] Created 3 overlapping windows (15.0s window, 5.0s overlap).
[Model Routing] Video Duration: 0.50 mins, Chunks: 3.
[Model Routing] Selected Groq model: 'llama-3.3-70b-versatile'
[Groq SES] Extracting graph triples from 3 chunks...
 -> Processing Chunk 1/3 (00:00 - 00:15)...
 -> Processing Chunk 2/3 (00:10 - 00:25)...
 -> Processing Chunk 3/3 (00:20 - 00:35)...
[Deduplication] Extracted 18 unique nodes and 15 unique relationships.
[Embedder] Generating embeddings using 'models/gemini-embedding-2' for 18 unique nodes...
[Neo4j Schema] Setting up uniqueness constraints...
[Neo4j Vector Index] Vector index 'ses_node_vector_index' already exists with correct dimensions (3072). Skipping creation.
[Neo4j Schema] Schema setup completed.
[Neo4j Ingest] Writing 18 nodes and 15 relationships...
[Neo4j Ingest Debug] Nodes query consumed. Counters: SummaryCounters{labels_added: 0, nodes_created: 0, properties_set: 108, contains_updates: True, contains_system_updates: False}
[Neo4j Ingest Debug] Rels query consumed (APOC). Counters: SummaryCounters{contains_updates: False, contains_system_updates: False}
[Neo4j Ingest Debug] Fallback rels consumed. Counters: SummaryCounters{relationships_created: 1, properties_set: 1, contains_updates: True, contains_system_updates: False}
[Neo4j Ingest] Graph batch insertion finished successfully.
[Neo4j Verification] Verified inserted nodes count in database '2cf7cc93': 24
[Neo4j Verification] Label counts: [[['Entity', 'State'], 20], [['Entity', 'Event'], 4]]
[Stage 2 Complete] State-Event-State graph successfully populated in Neo4j.

============================================================
STAGE 3 & 4: RETRIEVAL & ANSWER SYNTHESIS
============================================================
[RAG Pipeline] Processing Query: "What objects are present in the video and what is the sequence of events?"
[Vector Search Debug] Connecting to database: '2cf7cc93'
[Vector Search Debug] Available indexes: ['entity_id_unique', 'index_1b9dcc97', 'index_460996c0', 'ses_node_vector_index']
[Vector Search] Retrieved 3 seed nodes (top score: 0.8478).

--- RETRIEVED GRAPH CONTEXT ---
### SEED NODES MATCHED FROM VECTOR DB:
* STATE: state_bowl_and_wine_glass_on_table_00m00s (Timestamp: 00:00)
  Description: A bowl and a wine glass are sitting on a table.
* STATE: state_tennis_racket_and_banana_on_table_00m10s (Timestamp: 00:10)
  Description: A tennis racket and a banana are placed on a table.
* STATE: state_cup_on_table_00m00s (Timestamp: 00:00)
  Description: A cup is sitting on a table.

### CAUSAL & TEMPORAL NEIGHBORHOOD IN GRAPH:
- DURING: event_person_gesturing_00m00s (Event: A person is gesturing with their hands.) -> state_bowl_and_wine_glass_on_table_00m00s
- PRECEDES: state_bowl_and_wine_glass_on_table_00m00s -> event_person_waving_hand_00m05s (Event: A person waves their hand.)
- DURING: event_person_gesturing_00m00s (Event: A person is gesturing with their hands.) -> state_cup_on_table_00m00s
- PRECEDES: state_cup_on_table_00m00s -> event_person_waving_hand_00m05s (Event: A person waves their hand.)
- PRECEDES: event_person_waving_hand_00m05s (Event: A person waves their hand.) -> state_tennis_racket_and_banana_on_table_00m10s
- DURING: event_person_holding_tennis_racket_00m10s (Event: A person is holding a tennis racket.) -> state_tennis_racket_and_banana_on_table_00m10s

[Answering] Attempting to generate grounded answer using Groq (llama-3.3-70b-versatile)...

--- GEMINI GROUNDED ANSWER ---
Based on the provided Knowledge Graph context, here are the objects present and the chronological sequence of events:

### Objects Present in the Video
*   Bowl and Wine glass (sitting on a table at 00:00)
*   Cup (sitting on a table at 00:00)
*   Tennis racket and Banana (placed on a table at 00:10)

### Sequence of Events
1.  **Initial States & Concurrent Activity (00:00)**:
    *   States: A bowl, a wine glass, and a cup are sitting on a table.
    *   Event: During these states, a person is gesturing with their hands.
2.  **Transition Event (00:05)**:
    *   Event: Following the initial states, the person waves their hand.
3.  **Subsequent States & Concurrent Activity (00:10)**:
    *   State: The previous action precedes a state where a tennis racket and a banana are placed on a table.
    *   Event: During this subsequent state, the person is holding the tennis racket.
```

2. **CLI Orchestration (`main.py --help`)**:
   - Verified CLI argument parsing and modular imports without crash risks.
   - Output: Clean help text listing all ingestion, graph building, and retrieval options.

---

## 🚀 How to Run the Pipeline

### Step 1: Launch Neo4j Container
```bash
docker compose up -d
```
Neo4j Browser UI will be live at `http://localhost:7474` (User: `neo4j`, Password: `password`).

### Step 2: Set Environment Variable
```bash
set GEMINI_API_KEY=your_gemini_api_key_here
```

### Step 3: Run Full End-to-End RAG Pipeline
```bash
python main.py --video_path sample_video.mp4 --query "Why was the apple missing at 02:40?"
```

### Step 4: Run Open-Vocabulary Mode (Optional)
```bash
python ingestion.py --video_path surgical_video.mp4 --yolo_model yolov8s-world.pt --custom_classes scalpel forceps retractor
```

---

## 🖥️ Streamlit Web Application (`app.py`)

We built a state-of-the-art interactive **Streamlit web application** in [app.py](file:///c:/Users/SOUBHAGYA%20NAYAK/OneDrive/Desktop/New%20folder/app.py) featuring:
- **Connection Center Sidebar**: Real-time status indicator and manual configuration of API keys and Neo4j endpoints.
- **📁 1. Dual-Stream Ingestion Tab**: Video drag-and-drop, interactive ingestion param sliders (FPS, model types), and chronological timeline log viewer.
- **🕸️ 2. Neo4j Knowledge Graph Tab**: Single-click State-Event-State Knowledge Graph creation + live database statistics metrics showing node/relationship distributions.
- **💬 3. Grounded Q&A Engine Tab**: Premium chat container showing grounding answers paired with traversed graph citation cards.

### How to Run Streamlit:
```bash
streamlit run app.py
```
This will open the application in your local browser (typically at `http://localhost:8501`).

---

## 🖼️ Multimodal Frame-Passing RAG Upgrade

We upgraded the RAG pipeline to be multimodal:
1. **Frame Saving**: During ingestion, representative video frames are captured and written to a workspace directory: `./saved_frames/`.
2. **Neo4j Storage**: The `image_path` attribute is saved directly to State/Event nodes (`node.image_path`).
3. **Retrieval**: The retrieval queries return the `image_path` matching seed nodes and traversed nodes in the causal chain.
4. **Multimodal Answering**: The retrieved `.jpg` images are loaded using PIL and sent in a list to Gemini 2.0 Flash (`contents=[image_1, image_2, ..., user_query]`), leveraging full visual frame context to answer queries.
5. **UI Display**: The Streamlit interface displays the actual retrieved video frames under the Grounded Answer section.

---

## ⚡ Primary/Secondary Multimodal Fallback Architecture

To bypass Gemini Free Tier 429 quota exceptions, we added a robust failover architecture inside `retrieval_app.py`:
- **Primary Route**: Multimodal query goes directly to `gemini-2.0-flash`.
- **Secondary Route (Groq Vision)**: If a `429`, `RESOURCE_EXHAUSTED`, or `QUOTA` exception is raised by the Gemini API client, the retriever:
  1. Catches the error and outputs a warning: `[Fallback] Gemini rate limit hit. Routing to Groq Vision API...`.
  2. Encodes local keyframes to base64 strings using a helper function.
  3. Builds an OpenAI-compatible multimodal content list using `HumanMessage`.
  4. Dispatches the query to Groq's `qwen/qwen3.6-27b` Vision model.
  5. Instantly returns the fallback response without pipeline crashing or slow time sleeps.

---

## 🎨 Multimodal Visual Vector Search (CLIP Upgrade)

We implemented CLIP visual-text shared space indexing to allow direct visual query matching:
1. **Ingestion Embedding (`ingestion.py`)**: Extracted video frames are passed through `openai/clip-vit-base-patch32` to produce a normalized **512-dimensional** visual vector stored in `timeline.json`.
2. **Graph Construction (`graph_builder.py`)**: Inherits visual embedding vectors from matching chunks. The Neo4j `ses_node_vector_index` is configured with 512 dimensions. Pre-generated CLIP visual vectors are written directly to the `embedding` node attribute in Neo4j (bypassing Google embedding APIs).
3. **Query Retrieval (`retrieval_app.py`)**: The natural language user query is vectorized via CLIP's text embedding encoder (512-dims). Neo4j matches this text vector directly against visual node embeddings, identifying the exact matching frames.
4. **Targeted Multimodal Payload**: Passes only the matching frames to the answering model, removing the need for global chronological keyframe sampling.
