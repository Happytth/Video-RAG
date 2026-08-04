import sys
import os
import json
import tempfile
import warnings
import streamlit as st

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

warnings.filterwarnings("ignore", category=UserWarning)

# Import pipeline components
from ingestion import run_ingestion
from graph_builder import run_graph_builder
from retrieval_app import CausalVideoRetriever

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Causal-Transient Video RAG",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling (Glassmorphism, Sleek Dark Palette, Animations)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;600&display=swap');
    
    /* Font overrides */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    /* Global Background and Style */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b;
    }
    
    /* Premium Header Card */
    .header-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .header-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 60%);
        pointer-events: none;
    }
    .header-title {
        background: linear-gradient(90deg, #818cf8 0%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 8px;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 300;
    }
    
    /* Status Badge Indicator */
    .status-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 8px;
    }
    .status-active {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-inactive {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* Glassmorphic Panel Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.3);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0f172a;
        padding: 6px;
        border-radius: 8px;
        border: 1px solid #1e293b;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 16px;
        color: #94a3b8;
        font-weight: 600;
        transition: background-color 0.2s ease, color 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #312e81 !important;
        color: #c7d2fe !important;
    }
    
    /* Micro-Animations & Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3) !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button:hover {
        transform: scale(1.03) !important;
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.5) !important;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "video_processed" not in st.session_state:
    st.session_state.video_processed = False
if "timeline_data" not in st.session_state:
    st.session_state.timeline_data = []
if "graph_populated" not in st.session_state:
    st.session_state.graph_populated = False
if "uploaded_video_path" not in st.session_state:
    st.session_state.uploaded_video_path = ""

# Sidebar: Credentials Setup
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/graph.png", width=64)
    st.title("Connection Center")
    st.markdown("Set up LLM provider keys and Neo4j database configurations.")

    st.subheader("LLM API Keys")
    groq_key = st.text_input("GROQ API KEY", value=os.environ.get("GROQ_API_KEY", ""), type="password")

    st.subheader("Neo4j Database Config")
    neo4j_uri = st.text_input("Neo4j URI", value=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user = st.text_input("Neo4j Username", value=os.environ.get("NEO4J_USERNAME", os.environ.get("NEO4J_USER", "neo4j")))
    neo4j_pwd = st.text_input("Neo4j Password", value=os.environ.get("NEO4J_PASSWORD", "password"), type="password")

    # Export credentials to environment on change
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
    if neo4j_uri:
        os.environ["NEO4J_URI"] = neo4j_uri
    if neo4j_user:
        os.environ["NEO4J_USERNAME"] = neo4j_user
        os.environ["NEO4J_USER"] = neo4j_user
    if neo4j_pwd:
        os.environ["NEO4J_PASSWORD"] = neo4j_pwd

    # Check key statuses
    st.subheader("Services Status")
    
    status_groq = "status-active" if groq_key else "status-inactive"
    label_groq = "Connected (Groq Vision & LLM)" if groq_key else "Missing GROQ_API_KEY"
    st.markdown(f"**Groq Cloud API**: <span class='status-badge {status_groq}'>{label_groq}</span>", unsafe_allow_html=True)

# Main Page Header Layout
st.markdown("""
<div class='header-card'>
    <div class='header-title'>🎬 Causal-Transient Video RAG</div>
    <div class='header-subtitle'>Multi-modal Causal State-Event-State Knowledge Graphs & Grounded Reasoning</div>
</div>
""", unsafe_allow_html=True)

# Main Application Tabs
tab1, tab2, tab3 = st.tabs(["📁 1. Dual-Stream Ingestion", "🕸️ 2. Neo4j Knowledge Graph", "💬 3. Grounded Q&A Engine"])

# ============================================================================
# Tab 1: Video Ingestion & Timeline
# ============================================================================
with tab1:
    col1, col2 = st.columns([1, 1.5], gap="medium")
    
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Video File Uploader")
        uploaded_file = st.file_uploader("Choose an MP4 video file...", type=["mp4"])
        
        if uploaded_file:
            # Save file to a temporary location
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            
            st.session_state.uploaded_video_path = temp_path
            st.video(temp_path)
            st.success(f"Video uploaded: {uploaded_file.name}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Ingestion Configuration")
        
        fps = st.slider("YOLO Frame Extraction Rate (FPS)", min_value=1.0, max_value=10.0, value=4.0, step=1.0)
        yolo_model = st.selectbox("YOLO Model Type", ["yolo11n.pt", "yolov8s-world.pt"], index=0)
        whisper_model = st.selectbox("Whisper Audio Model", ["base", "tiny", "small", "Skip Audio Ingestion"], index=0)
        
        # Action Button for Stage 1 Ingestion
        if st.button("Start Stage 1 Ingestion", type="primary"):
            if not st.session_state.uploaded_video_path:
                st.error("Please upload a video file first.")
            else:
                with st.spinner("Processing video (Extracting keyframe images, running YOLO, Whisper audio, and Shazam)..."):
                    try:
                        timeline_json = "timeline.json"
                        w_model = None if whisper_model == "Skip Audio Ingestion" else whisper_model
                        
                        run_ingestion(
                            video_path=st.session_state.uploaded_video_path,
                            output_json=timeline_json,
                            target_fps=fps,
                            yolo_model=yolo_model,
                            whisper_model=w_model
                        )
                        
                        if os.path.exists(timeline_json):
                            with open(timeline_json, "r", encoding="utf-8") as f:
                                st.session_state.timeline_data = json.load(f)
                            st.session_state.video_processed = True
                            st.success("Stage 1 Ingestion completed! Timeline generated. Switch to Tab 2 to build the Knowledge Graph.")
                        else:
                            st.error("Ingestion failed: timeline.json was not created.")
                    except Exception as e:
                        st.error(f"Ingestion crashed: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Chronological Timeline Log")
        
        if st.session_state.video_processed and st.session_state.timeline_data:
            st.info("Synchronized visual logs and spoken transcripts at 5.0s granularity:")
            
            # Render timeline as formatted Markdown cards
            for block in st.session_state.timeline_data:
                start = block.get("window_start", 0.0)
                end = block.get("window_end", 0.0)
                objects = block.get("visual_objects", [])
                transcript = block.get("transcript_text", "No spoken audio.")
                
                st.markdown(f"""
                <div style='background-color: rgba(30, 41, 59, 0.6); padding: 12px; border-left: 4px solid #818cf8; border-radius: 4px; margin-bottom: 8px;'>
                    <strong style='color:#c084fc;'>🕒 {start:.1f}s - {end:.1f}s</strong><br/>
                    <strong>Visual Objects:</strong> {', '.join(objects) if objects else 'None detected'}<br/>
                    <strong>Audio Transcript:</strong> <em>"{transcript}"</em>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No timeline logs loaded. Process a video file to visualize the ingestion results.")
        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================================
# Tab 2: Knowledge Graph Building & Database View
# ============================================================================
with tab2:
    st.subheader("Build Knowledge Graph (Stage 2)")
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Graph Builder Execution")
        st.markdown("""
        Translates chronological visual & audio blocks into a causal **State-Event-State (SES)** knowledge graph using Pydantic structured schemas.
        * **Duration-based Model Routing** is triggered dynamically to stay strictly within rate limits.
        * Node descriptions are vectorized using Google embeddings and stored in a native Neo4j index.
        """)
        
        window_size = st.number_input("Sliding Window Size (Seconds)", min_value=5.0, max_value=60.0, value=15.0, step=1.0)
        overlap = st.number_input("Sliding Window Overlap (Seconds)", min_value=0.0, max_value=30.0, value=5.0, step=1.0)
        
        if st.button("Construct Knowledge Graph"):
            if not st.session_state.video_processed:
                st.error("Please run Stage 1 Ingestion first.")
            elif not groq_key:
                st.error("GROQ_API_KEY must be provided in the Connection Center.")
            else:
                with st.spinner("Extracting causal relations using Groq and generating embeddings..."):
                    try:
                        run_graph_builder(
                            timeline_json_path="timeline.json",
                            groq_api_key=groq_key,
                            neo4j_uri=neo4j_uri,
                            neo4j_user=neo4j_user,
                            neo4j_password=neo4j_pwd,
                            window_size_sec=window_size,
                            overlap_sec=overlap
                        )
                        st.session_state.graph_populated = True
                        st.success("State-Event-State Knowledge Graph successfully populated in Neo4j!")
                    except Exception as e:
                        st.error(f"Graph builder failed: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Neo4j Live Database Stats")
        
        if st.button("Fetch Current DB Stats") or st.session_state.graph_populated:
            try:
                from neo4j import GraphDatabase
                driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pwd))
                db_name = os.environ.get("NEO4J_DATABASE", "neo4j")
                
                with driver.session(database=db_name) as session:
                    # Retrieve total counts
                    total_cnt = session.run("MATCH (n) RETURN count(n)").single()[0]
                    state_cnt = session.run("MATCH (n:State) RETURN count(n)").single()[0]
                    event_cnt = session.run("MATCH (n:Event) RETURN count(n)").single()[0]
                    rel_cnt = session.run("MATCH ()-[r]->() RETURN count(r)").single()[0]
                    
                    # Check vector index
                    indexes = [r.data() for r in session.run("SHOW INDEXES YIELD name, type, options")]
                    vector_idx_opts = None
                    for idx in indexes:
                        if idx['name'] == 'ses_node_vector_index':
                            vector_idx_opts = idx['options'].get('indexConfig', {})
                            break
                driver.close()
                
                # Render Metric Cards
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.metric("Total Nodes", total_cnt)
                    st.metric("State Nodes", state_cnt)
                with m_col2:
                    st.metric("Relationship Edges", rel_cnt)
                    st.metric("Event Nodes", event_cnt)
                
                if vector_idx_opts:
                    st.success("ses_node_vector_index is ONLINE")
                    st.json(vector_idx_opts)
                else:
                    st.warning("ses_node_vector_index not found in database.")
            except Exception as e:
                st.error(f"Could not connect to database to fetch stats: {e}")
        else:
            st.info("Populate graph or click button to inspect active database.")
        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================================
# Tab 3: Causal Q&A Chat Engine
# ============================================================================
with tab3:
    st.subheader("Conversational Video Reasoning Engine")
    
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    query = st.text_input("Ask a question about temporal actions, object persistency, or cause-and-effect in the video:", 
                         placeholder="e.g., What objects are present in the video and what is the sequence of events?")
    
    if st.button("Execute Causal Query") and query:
        if not groq_key:
            st.error("GROQ_API_KEY must be provided in the Connection Center or .env file.")
        else:
            with st.spinner("Searching Neo4j vector space & traversing causal chains..."):
                try:
                    retriever = CausalVideoRetriever(
                        neo4j_uri=neo4j_uri,
                        neo4j_user=neo4j_user,
                        neo4j_password=neo4j_pwd,
                        groq_api_key=groq_key
                    )
                    
                    # Run RAG Query
                    res = retriever.query_video_rag(query)
                    retriever.close()
                    
                    # Render Grounded Answer Card
                    st.markdown("### 📝 Grounded Synthesis Answer")
                    st.markdown(res["answer"])
                    
                    # Display retrieved visual frames passed to LLM
                    image_paths = res.get("image_paths", [])
                    if image_paths:
                        st.markdown("### 🖼️ Passed Visual Frames Context")
                        cols = st.columns(min(len(image_paths), 4))
                        for idx, img_path in enumerate(image_paths):
                            if os.path.exists(img_path):
                                with cols[idx % len(cols)]:
                                    # Format label with timestamp for display
                                    basename = os.path.basename(img_path)
                                    try:
                                        seconds = int(basename.replace("frame_", "").replace(".jpg", ""))
                                        mins = seconds // 60
                                        secs = seconds % 60
                                        caption = f"Frame at {mins:02d}:{secs:02d}"
                                    except Exception:
                                        caption = basename
                                    st.image(img_path, caption=caption)
                    
                    # Expandable Accordion for Traversed Causal Subgraph
                    with st.expander("🕸️ Traversed Graph Causal Context (Citations)", expanded=True):
                        st.markdown(res["graph_context"])
                except Exception as e:
                    st.error(f"Retrieval RAG Query failed: {e}")
    st.markdown("</div>", unsafe_allow_html=True)
