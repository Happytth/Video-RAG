# Implementation Plan: Causal-Transient Video RAG System

Standard semantic video RAG fails to capture split-second visual events ("transient") and cannot map cause-and-effect across time ("causal"). This plan outlines the technical implementation of a **Causal-Transient Video RAG** system using a **Dual-Stream Visual/Audio Encoder** paired with a **State-Event-State (SES) Knowledge Graph** in **Neo4j** powered by **Gemini 1.5/2.0 Pro** and **Google text-embedding-004**.

---

## Technical Architecture & Workflow

```mermaid
graph TD
    subgraph Stage 1: Dual-Stream Ingestion (ingestion.py)
        V[Video .mp4] --> OpenCV[OpenCV Frame Sampler 3-5 FPS]
        V --> Whisper[Whisper Audio Transcriber]
        OpenCV --> YOLO[YOLOv10/11 Detection]
        YOLO --> StreamA[Stream A: Visual Object Timestamps]
        Whisper --> StreamB[Stream B: Timestamped Transcripts]
        StreamA --> Merger[Chronological Timeline Merger]
        StreamB --> Merger
        Merger --> Timeline[timeline.json]
    end

    subgraph Stage 2: SES Knowledge Graph Construction (graph_builder.py)
        Timeline --> Chunker[15s Window Chunker]
        Chunker --> GeminiSES[Gemini Pro SES Extraction Prompt]
        GeminiSES --> GraphJSON[State-Event-State Triples]
        GraphJSON --> Embedder[Google text-embedding-004]
        Embedder --> Neo4jWriter[Neo4j Cypher Batch Writer]
        Neo4jWriter --> Neo4j[(Neo4j DB: Graph + Vector Index)]
    end

    subgraph Stage 3 & 4: Retrieval & Grounded Generation (retrieval_app.py)
        Query[User Query] --> QEmbed[Google text-embedding-004]
        QEmbed --> VecSearch[Neo4j Vector Search]
        VecSearch --> SeedNode[Seed State/Event Node]
        SeedNode --> CypherTraverse[Cypher 2-Hop Backward / 1-Hop Forward Traversal]
        CypherTraverse --> Context[Structured Causal Context]
        Context --> GeminiRAG[Gemini Pro RAG Synthesizer]
        GeminiRAG --> Answer[Causal & Timestamp-Grounded Answer]
    end
```

---

## User Review Required

> [!IMPORTANT]
> **Prerequisites & Dependencies**:
> - **Neo4j DB**: Running local Neo4j instance (Community Edition / Docker or Desktop) on `bolt://localhost:7687` with Vector Search support (Neo4j v5.11+).
> - **Gemini API Key**: Set as environment variable `GEMINI_API_KEY`.
> - **Local GPU/CPU**: `ultralytics` (YOLO) and `openai-whisper` run locally. ffmpeg must be available on system PATH for audio extraction.

---

## Open Questions

> [!NOTE]
> 1. Do you have a running Neo4j instance locally via Docker or Neo4j Desktop, or would you like docker-compose / instructions included?
> 2. For YOLO model size, we default to `yolo11n.pt` / `yolov8n.pt` for fast execution; would you like configurable model options (e.g. `yolo11x.pt` for higher precision)?

---

## Proposed Modules & File Structure

### 1. `requirements.txt`
Dependencies list: `ultralytics`, `openai-whisper`, `opencv-python`, `google-genai`, `neo4j`, `langchain-neo4j`, `pydantic`, `python-dotenv`, `numpy`.

---

### 2. [NEW] [ingestion.py](file:///c:/Users/SOUBHAGYA%20NAYAK/OneDrive/Desktop/New%20folder/ingestion.py)
Dual-stream processing script:
- **Visual Stream (Stream A)**: Extracts video frames using OpenCV at a configurable frame rate (3–5 FPS). Runs frames through YOLO (`yolo11n.pt`/`yolov8n.pt`) to capture transient object detections with timestamps.
- **Audio Stream (Stream B)**: Uses OpenAI Whisper (`base`/`small`) to generate word-level / segment-level timestamped transcripts.
- **Timeline Synchronization**: Merges visual object logs and spoken transcript segments into chronologically aligned 15-second windows in `timeline.json`.

---

### 3. [NEW] [graph_builder.py](file:///c:/Users/SOUBHAGYA%20NAYAK/OneDrive/Desktop/New%20folder/graph_builder.py)
SES Knowledge Graph extractor and Neo4j ingester:
- **SES Extraction Prompt**: Leverages Gemini 1.5/2.0 Pro with structured Pydantic output schema to extract `State` nodes, `Event` nodes, and relationships (`PRECEDES`, `CAUSES`, `DURING`).
  - *State Node*: Static situations (e.g., "Apple is on table", timestamp "01:40").
  - *Event Node*: Actions/Transitions (e.g., "Man picks up apple", timestamp "02:10").
  - *Relationships*: Causal and chronological directed links.
- **Vector Embedding**: Vectorizes all state and event descriptions using Google's `text-embedding-004` (768 dimensions).
- **Neo4j Storage & Indexing**: Executes Cypher schema setup (Uniqueness constraints + Vector index `ses_node_vector_index`) and inserts nodes/edges into Neo4j.

---

### 4. [NEW] [retrieval_app.py](file:///c:/Users/SOUBHAGYA%20NAYAK/OneDrive/Desktop/New%20folder/retrieval_app.py)
Bidirectional retriever and Gemini generation pipeline:
- **Query Embedding**: Embeds user natural language queries with `text-embedding-004`.
- **Vector Search**: Finds nearest seed nodes in Neo4j vector index.
- **Bidirectional Cypher Traversal**:
  ```cypher
  MATCH (seed:Entity) WHERE seed.id = $seed_id
  MATCH path = (cause:Entity)-[:PRECEDES|CAUSES|DURING*1..2]->(seed)-[:PRECEDES|CAUSES|DURING*0..1]->(effect:Entity)
  RETURN path
  ```
- **Grounded Answer Synthesis**: Formats causal pathways into structured text and prompts Gemini Pro to produce precise answers citing exact timestamps and cause-and-effect chains.

---

### 5. [NEW] [main.py](file:///c:/Users/SOUBHAGYA%20NAYAK/OneDrive/Desktop/New%20folder/main.py)
Convenience CLI pipeline orchestrator to run end-to-end ingestion, graph building, and Q&A query benchmarking.

---

## Verification Plan

### Automated / Code Verification
1. Verify Python syntax and typing of all created files.
2. Unit testing mockup verifying timeline JSON schema, graph node schema, Cypher query generation, and RAG context formatting.

### Manual / Integration Verification
- Execute `ingestion.py` on sample video to produce `timeline.json`.
- Execute `graph_builder.py` to populate local Neo4j database and create vector index.
- Execute `retrieval_app.py` with complex causal queries (e.g., "Why did X happen at timestamp Y?") to verify timestamp-backed causal retrieval.
