#!/usr/bin/env python3
"""
Simple but Complete Youtu-GraphRAG Backend
Integrates real GraphRAG functionality with a simple interface
"""

import os
import re
import sys
import json
import asyncio
import glob
import shutil
from collections import Counter
from typing import List, Dict, Optional
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# FastAPI imports
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from utils.logger import logger
from utils.dataset_audit import audit_dataset
from utils.paths import repo_path, repo_str
from utils.process_control import install_interrupt_guard, terminate_process_tree
import ast

# Import document parser
try:
    from utils.document_parser import get_parser

    DOCUMENT_PARSER_AVAILABLE = True
except ImportError as e:
    DOCUMENT_PARSER_AVAILABLE = False
    logger.warning(f"Document parser not available: {e}")

# Try to import GraphRAG components
try:
    from models.constructor import kt_gen as constructor
    from models.retriever import agentic_decomposer as decomposer, enhanced_kt_retriever as retriever, agent_runner
    from config import get_config, ConfigManager

    GRAPHRAG_AVAILABLE = True
    logger.info("GraphRAG components loaded successfully")
except ImportError as e:
    GRAPHRAG_AVAILABLE = False
    logger.error(f"GraphRAG components not available: {e}")

ASSETS_DIR = repo_path("assets")
FRONTEND_DIR = repo_path("frontend")
UPLOADED_DATA_DIR = repo_path("data", "uploaded")
OUTPUT_GRAPHS_DIR = repo_path("output", "graphs")
OUTPUT_CHUNKS_DIR = repo_path("output", "chunks")
OUTPUT_LOGS_DIR = repo_path("output", "logs")
SCHEMAS_DIR = repo_path("schemas")
FAISS_CACHE_ROOT = repo_path("retriever", "faiss_cache_new")
BASE_CONFIG_PATH = repo_str("config", "base_config.yaml")
FRONTEND_INDEX_PATH = repo_str("frontend", "index_new.html")
FRONTEND_STYLE_PATH = repo_str("frontend", "style.css")
FRONTEND_SCRIPT_PATH = repo_str("frontend", "script.js")
FAVICON_PATH = repo_str("assets", "SYSU.png")
DEMO_SCHEMA_PATH = repo_str("schemas", "demo.json")
DEMO_CORPUS_PATH = repo_str("data", "demo", "demo_corpus.json")
DEMO_GRAPH_PATH = repo_str("output", "graphs", "demo_new.json")


def uploaded_dataset_dir(dataset_name: str) -> str:
    return repo_str("data", "uploaded", dataset_name)


def dataset_corpus_path(dataset_name: str) -> str:
    return repo_str("data", "uploaded", dataset_name, "corpus.json")


def dataset_schema_path(dataset_name: str) -> str:
    return repo_str("schemas", f"{dataset_name}.json")


def dataset_graph_path(dataset_name: str) -> str:
    return repo_str("output", "graphs", f"{dataset_name}_new.json")


def dataset_chunk_path(dataset_name: str) -> str:
    return repo_str("output", "chunks", f"{dataset_name}.txt")


app = FastAPI(title="Youtu-GraphRAG Unified Interface", version="1.0.0")

# Mount static files (assets directory)
app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")
# Mount frontend directory for frontend assets
app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(FAVICON_PATH)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
active_connections: Dict[str, WebSocket] = {}
config = None


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)


manager = ConnectionManager()


LLM_SCOPE_ENV = {
    "kg": ("KG_LLM_MODEL", "KG_LLM_BASE_URL", "KG_LLM_API_KEY"),
    "rag": ("RAG_LLM_MODEL", "RAG_LLM_BASE_URL", "RAG_LLM_API_KEY"),
    "default": ("LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY"),
}

PAPER_VISUAL_CATEGORY = "\u8bba\u6587"
AUTHOR_VISUAL_CATEGORY = "\u4f5c\u8005"
ORG_VISUAL_CATEGORY = "\u673a\u6784"
TECH_VISUAL_CATEGORY = "\u6280\u672f"
FIELD_VISUAL_CATEGORY = "\u6559\u80b2\u9886\u57df"
SCENARIO_VISUAL_CATEGORY = "\u6559\u5b66\u573a\u666f"
ATTRIBUTE_VISUAL_CATEGORY = "\u5c5e\u6027"
COMMUNITY_VISUAL_CATEGORY = "\u4e3b\u9898\u793e\u533a"
TOPIC_VISUAL_CATEGORY = "\u7814\u7a76\u4e3b\u9898"
METHOD_VISUAL_CATEGORY = "\u7814\u7a76\u65b9\u6cd5"
JOURNAL_VISUAL_CATEGORY = "\u671f\u520a"
KEYWORD_VISUAL_CATEGORY = "\u5173\u952e\u8bcd"
ENTITY_VISUAL_CATEGORY = "entity"

VISUAL_CATEGORY_ALIASES = {
    "community": COMMUNITY_VISUAL_CATEGORY,
    "keyword": KEYWORD_VISUAL_CATEGORY,
    "attribute": ATTRIBUTE_VISUAL_CATEGORY,
    "paper": PAPER_VISUAL_CATEGORY,
    "author": AUTHOR_VISUAL_CATEGORY,
    "organization": ORG_VISUAL_CATEGORY,
    "institution": ORG_VISUAL_CATEGORY,
    "journal": JOURNAL_VISUAL_CATEGORY,
    "method": METHOD_VISUAL_CATEGORY,
    "framework": METHOD_VISUAL_CATEGORY,
    "topic": TOPIC_VISUAL_CATEGORY,
    "theme": TOPIC_VISUAL_CATEGORY,
    "field": FIELD_VISUAL_CATEGORY,
    "scenario": SCENARIO_VISUAL_CATEGORY,
    "teaching mode": METHOD_VISUAL_CATEGORY,
    "教学模式": METHOD_VISUAL_CATEGORY,
    "人才培养模式": METHOD_VISUAL_CATEGORY,
    "研究理论": TOPIC_VISUAL_CATEGORY,
    "教育理念": TOPIC_VISUAL_CATEGORY,
}

VISUAL_CATEGORY_ORDER = [
    PAPER_VISUAL_CATEGORY,
    TOPIC_VISUAL_CATEGORY,
    ORG_VISUAL_CATEGORY,
    TECH_VISUAL_CATEGORY,
    FIELD_VISUAL_CATEGORY,
    SCENARIO_VISUAL_CATEGORY,
    ATTRIBUTE_VISUAL_CATEGORY,
    COMMUNITY_VISUAL_CATEGORY,
    AUTHOR_VISUAL_CATEGORY,
    METHOD_VISUAL_CATEGORY,
    JOURNAL_VISUAL_CATEGORY,
    KEYWORD_VISUAL_CATEGORY,
    ENTITY_VISUAL_CATEGORY,
]

VISUAL_CATEGORY_COVERAGE_PRIORITY = [
    TOPIC_VISUAL_CATEGORY,
    SCENARIO_VISUAL_CATEGORY,
    COMMUNITY_VISUAL_CATEGORY,
    METHOD_VISUAL_CATEGORY,
    TECH_VISUAL_CATEGORY,
    FIELD_VISUAL_CATEGORY,
    JOURNAL_VISUAL_CATEGORY,
    ATTRIBUTE_VISUAL_CATEGORY,
]


def normalize_visual_category(raw_category: str) -> str:
    category = str(raw_category or "").strip()
    if not category:
        return ENTITY_VISUAL_CATEGORY
    lowered = category.lower()
    return VISUAL_CATEGORY_ALIASES.get(lowered, VISUAL_CATEGORY_ALIASES.get(category, category))


def ordered_visual_categories(raw_categories) -> List[str]:
    seen = set()
    normalized_categories = []
    for category in raw_categories:
        normalized = normalize_visual_category(category)
        if normalized and normalized not in seen:
            normalized_categories.append(normalized)
            seen.add(normalized)

    ordered = [category for category in VISUAL_CATEGORY_ORDER if category in seen]
    extras = sorted(category for category in normalized_categories if category not in ordered)
    return ordered + extras


def build_visual_node_id(node_data: Dict) -> str:
    props = node_data.get("properties", {}) if isinstance(node_data, dict) else {}
    name = str(props.get("name", "")).strip()
    schema_type = normalize_visual_category(props.get("schema_type", node_data.get("label", "entity")))
    if schema_type == PAPER_VISUAL_CATEGORY:
        doc_uid = str(props.get("doc_uid", props.get("chunk_id", props.get("chunk id", "")))).strip()
        if doc_uid:
            return f"paper::{doc_uid}"
    return name


def ensure_llm_scope_config(scope: str):
    env_names = LLM_SCOPE_ENV.get(scope)
    if not env_names:
        raise HTTPException(status_code=500, detail=f"Unsupported LLM scope: {scope}")

    _, _, api_key_env = env_names
    if not os.getenv(api_key_env):
        raise HTTPException(
            status_code=500,
            detail=f"{scope.upper()} LLM API key not found in environment variables ({api_key_env}).",
        )


# Request/Response models
class FileUploadResponse(BaseModel):
    success: bool
    message: str
    dataset_name: Optional[str] = None
    files_count: Optional[int] = None


class GraphConstructionRequest(BaseModel):
    dataset_name: str


class GraphConstructionResponse(BaseModel):
    success: bool
    message: str
    graph_data: Optional[Dict] = None


class QuestionRequest(BaseModel):
    question: str
    dataset_name: str


class QuestionResponse(BaseModel):
    answer: str
    sub_questions: List[Dict]
    retrieved_triples: List[str]
    retrieved_chunks: List[str]
    reasoning_steps: List[Dict]
    visualization_data: Dict
    decompose_fallback: bool = False
    decompose_error: Optional[str] = None
    schema_path_used: Optional[str] = None


def ensure_demo_schema_exists() -> str:
    """Ensure default demo schema exists and return its path."""
    os.makedirs(SCHEMAS_DIR, exist_ok=True)
    schema_path = DEMO_SCHEMA_PATH
    if not os.path.exists(schema_path):
        demo_schema = {
            "Nodes": [
                "person", "location", "organization", "event", "object",
                "concept", "time_period", "creative_work", "biological_entity", "natural_phenomenon"
            ],
            "Relations": [
                "is_a", "part_of", "located_in", "created_by", "used_by", "participates_in",
                "related_to", "belongs_to", "influences", "precedes", "arrives_in", "comparable_to"
            ],
            "Attributes": [
                "name", "date", "size", "type", "description", "status",
                "quantity", "value", "position", "duration", "time"
            ]
        }
        with open(schema_path, 'w') as f:
            json.dump(demo_schema, f, indent=2)
    return schema_path


def get_schema_path_for_dataset(dataset_name: str) -> str:
    """Return dataset-specific schema if present; otherwise fallback to demo schema."""
    if dataset_name and dataset_name != "demo":
        ds_schema = dataset_schema_path(dataset_name)
        if os.path.exists(ds_schema):
            return ds_schema
    return ensure_demo_schema_exists()


async def send_progress_update(client_id: str, stage: str, progress: int, message: str):
    """Send progress update via WebSocket"""
    await manager.send_message({
        "type": "progress",
        "stage": stage,
        "progress": progress,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }, client_id)


# -------- Encoding detection helpers --------
def _detect_encoding_from_bytes(data: bytes) -> Optional[str]:
    """Detect encoding using chardet if available; return lower-cased encoding name or None."""
    try:
        import chardet  # type: ignore
        result = chardet.detect(data) or {}
        enc = result.get("encoding")
        if enc:
            return enc.lower()
    except Exception:
        pass
    return None


def decode_bytes_with_detection(data: bytes) -> str:
    """Decode bytes to string with encoding detection and robust fallbacks.
    Order: detected -> utf-8/utf-8-sig -> common Chinese encodings -> utf-16 variants -> latin-1 -> replace.
    """
    candidates = []
    detected = _detect_encoding_from_bytes(data)
    if detected:
        candidates.append(detected)
    candidates.extend([
        "utf-8", "utf-8-sig", "gb18030", "gbk", "big5",
        "utf-16", "utf-16le", "utf-16be", "latin-1"
    ])
    # De-duplicate while preserving order
    tried = set()
    for enc in candidates:
        if enc in tried or not enc:
            continue
        tried.add(enc)
        try:
            return data.decode(enc)
        except Exception:
            continue
    # Last resort
    return data.decode("utf-8", errors="replace")


async def clear_cache_files(dataset_name: str):
    """Clear all cache files for a dataset before graph construction"""
    try:
        # Clear FAISS cache files
        faiss_cache_dir = str(FAISS_CACHE_ROOT / dataset_name)
        if os.path.exists(faiss_cache_dir):
            shutil.rmtree(faiss_cache_dir)
            logger.info(f"Cleared FAISS cache directory: {faiss_cache_dir}")

        # Clear output chunks
        chunk_file = dataset_chunk_path(dataset_name)
        if os.path.exists(chunk_file):
            os.remove(chunk_file)
            logger.info(f"Cleared chunk file: {chunk_file}")

        # Clear output graphs
        graph_file = dataset_graph_path(dataset_name)
        if os.path.exists(graph_file):
            os.remove(graph_file)
            logger.info(f"Cleared graph file: {graph_file}")

        # Clear any other cache files with dataset name pattern
        cache_patterns = [
            repo_str("output", "logs", f"{dataset_name}_*.log"),
            repo_str("output", "chunks", f"{dataset_name}_*"),
            repo_str("output", "graphs", f"{dataset_name}_*"),
        ]

        for pattern in cache_patterns:
            for file_path in glob.glob(pattern):
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        logger.info(f"Cleared cache file: {file_path}")
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        logger.info(f"Cleared cache directory: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clear {file_path}: {e}")

        logger.info(f"Cache cleanup completed for dataset: {dataset_name}")

    except Exception as e:
        logger.error(f"Error clearing cache files for {dataset_name}: {e}")
        # Don't raise exception, just log the error


@app.get("/api/dataset-audit/{dataset_name}")
async def get_dataset_audit(dataset_name: str):
    """Return a stable audit snapshot for source/chunk/graph consistency checks."""
    try:
        return audit_dataset(dataset_name)
    except Exception as e:
        logger.error(f"Dataset audit failed for {dataset_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Serve frontend HTML

# @app.get("/")
# async def read_root():
#     frontend_path = "frontend/index.html"
#     if os.path.exists(frontend_path):
#         return FileResponse(frontend_path)
#     return {"message": "Youtu-GraphRAG Unified Interface is running!", "status": "ok"}

@app.get("/")
async def read_root():
    # 1. 确保这里指向你新拆分出的 index_new.html
    frontend_path = FRONTEND_INDEX_PATH
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {"message": "Youtu-GraphRAG Unified Interface is running!", "status": "ok"}


@app.get("/style.css")
async def read_css():
    return FileResponse(FRONTEND_STYLE_PATH)


@app.get("/script.js")
async def read_js():
    return FileResponse(FRONTEND_SCRIPT_PATH)


@app.get("/api/status")
async def get_status():
    return {
        "message": "Youtu-GraphRAG Unified Interface is running!",
        "status": "ok",
        "graphrag_available": GRAPHRAG_AVAILABLE
    }


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(client_id)


@app.post("/api/upload", response_model=FileUploadResponse)
async def upload_files(files: List[UploadFile] = File(...), client_id: str = "default"):
    """Upload files and prepare for graph construction"""
    try:
        # Generate dataset name based on file count
        if len(files) == 1:
            # Single file: use its name
            main_file = files[0]
            original_name = os.path.splitext(main_file.filename)[0]
            # Clean filename to be filesystem-safe
            dataset_name = "".join(c for c in original_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            dataset_name = dataset_name.replace(' ', '_')
        else:
            # Multiple files: create a descriptive name with date
            from datetime import datetime
            date_str = datetime.now().strftime("%Y%m%d")
            dataset_name = f"{len(files)}files_{date_str}"

        # Add counter if dataset already exists
        base_name = dataset_name
        counter = 1
        while os.path.exists(uploaded_dataset_dir(dataset_name)):
            dataset_name = f"{base_name}_{counter}"
            counter += 1

        upload_dir = uploaded_dataset_dir(dataset_name)
        os.makedirs(upload_dir, exist_ok=True)

        await send_progress_update(client_id, "upload", 10, "Starting file upload...")

        # Process uploaded files
        corpus_data = []
        skipped_files: List[str] = []
        processed_count = 0
        allowed_extensions = {".txt", ".md", ".json", ".pdf", ".docx", ".doc"}

        # Initialize document parser if needed
        doc_parser = None
        if DOCUMENT_PARSER_AVAILABLE:
            doc_parser = get_parser()

        for i, file in enumerate(files):
            file_path = os.path.join(upload_dir, file.filename)
            with open(file_path, "wb") as buffer:
                content_bytes = await file.read()
                buffer.write(content_bytes)

            # Process file content using encoding detection
            filename_lower = (file.filename or "").lower()
            ext = os.path.splitext(filename_lower)[1]
            if ext not in allowed_extensions:
                # Skip unsupported file types to avoid processing binary files as text
                logger.warning(f"Skipping unsupported file type: {file.filename}")
                skipped_files.append(file.filename)
                progress = 10 + (i + 1) * 80 // len(files)
                await send_progress_update(client_id, "upload", progress, f"Skipped unsupported file: {file.filename}")
                continue

            # Handle PDF and DOCX/DOC files with document parser
            if ext in ['.pdf', '.docx', '.doc']:
                if not doc_parser:
                    logger.warning(f"Document parser not available, skipping {file.filename}")
                    skipped_files.append(file.filename)
                    progress = 10 + (i + 1) * 80 // len(files)
                    await send_progress_update(client_id, "upload", progress,
                                               f"Skipped {file.filename} (parser unavailable)")
                    continue

                try:
                    text = doc_parser.parse_file(file_path, ext)
                    if text and text.strip():
                        corpus_data.append({
                            "title": file.filename,
                            "text": text
                        })
                        processed_count += 1
                        await send_progress_update(client_id, "upload", 10 + (i + 1) * 80 // len(files),
                                                   f"Parsed {file.filename}")
                    else:
                        logger.warning(f"No text extracted from {file.filename}")
                        skipped_files.append(file.filename)
                        await send_progress_update(client_id, "upload", 10 + (i + 1) * 80 // len(files),
                                                   f"No text in {file.filename}")
                except Exception as e:
                    logger.error(f"Error parsing {file.filename}: {e}")
                    skipped_files.append(file.filename)
                    await send_progress_update(client_id, "upload", 10 + (i + 1) * 80 // len(files),
                                               f"Failed to parse {file.filename}")
                continue

            # Treat plain text formats explicitly (.txt and .md)
            if filename_lower.endswith(('.txt', '.md')):
                text = decode_bytes_with_detection(content_bytes)
                corpus_data.append({
                    "title": file.filename,
                    "text": text
                })
                processed_count += 1
            elif filename_lower.endswith('.json'):
                try:
                    json_text = decode_bytes_with_detection(content_bytes)
                    data_obj = json.loads(json_text)
                    if isinstance(data_obj, list):
                        corpus_data.extend(data_obj)
                    elif isinstance(data_obj, dict):
                        corpus_data.append(data_obj)
                    else:
                        logger.warning(f"JSON file {file.filename} content is neither list nor dict")
                    processed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to parse JSON {file.filename}: {e}. Treating as text.")
                    # If JSON parsing fails, treat as text
                    text = decode_bytes_with_detection(content_bytes)
                    corpus_data.append({
                        "title": file.filename,
                        "text": text
                    })

            progress = 10 + (i + 1) * 80 // len(files)
            await send_progress_update(client_id, "upload", progress, f"Processed {file.filename}")

        # Ensure at least one valid file processed
        if processed_count == 0:
            msg = "No supported files were uploaded. Allowed: .txt, .md, .json, .pdf, .docx, .doc"
            if skipped_files:
                msg += f"; skipped: {', '.join(skipped_files)}"
            await send_progress_update(client_id, "upload", 0, msg)
            raise HTTPException(status_code=400, detail=msg)

        # Save corpus data
        corpus_path = dataset_corpus_path(dataset_name)
        with open(corpus_path, 'w', encoding='utf-8') as f:
            json.dump(corpus_data, f, ensure_ascii=False, indent=2)

        # Create dataset configuration
        await create_dataset_config()

        await send_progress_update(client_id, "upload", 100, "Upload completed successfully!")

        msg_ok = "Files uploaded successfully"
        if skipped_files:
            msg_ok += f"; skipped unsupported: {', '.join(skipped_files)}"
        return FileUploadResponse(
            success=True,
            message=msg_ok,
            dataset_name=dataset_name,
            files_count=processed_count
        )

    except Exception as e:
        await send_progress_update(client_id, "upload", 0, f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def create_dataset_config():
    """Create dataset configuration"""
    # Ensure default demo schema exists
    ensure_demo_schema_exists()


@app.post("/api/construct-graph", response_model=GraphConstructionResponse)
async def construct_graph(request: GraphConstructionRequest, client_id: str = "default"):
    """Construct knowledge graph from uploaded data"""
    try:
        if not GRAPHRAG_AVAILABLE:
            raise HTTPException(status_code=503,
                                detail="GraphRAG components not available. Please install or configure them.")
        dataset_name = request.dataset_name

        await send_progress_update(client_id, "construction", 2, "Cleaning old cache files...")

        # Clear all cache files before construction
        await clear_cache_files(dataset_name)

        await send_progress_update(client_id, "construction", 5, "Initializing graph builder...")

        # Get dataset paths
        corpus_path = dataset_corpus_path(dataset_name)
        # Choose schema: dataset-specific or default demo
        schema_path = get_schema_path_for_dataset(dataset_name)

        if not os.path.exists(corpus_path):
            # Try demo dataset
            corpus_path = DEMO_CORPUS_PATH

        if not os.path.exists(corpus_path):
            raise HTTPException(status_code=404, detail="Dataset not found")

        await send_progress_update(client_id, "construction", 10, "Loading configuration and corpus...")

        # Initialize config
        global config
        if config is None:
            config = get_config(BASE_CONFIG_PATH)

        # 【关键修复】确保 LLM API Key 存在，否则这里就会抛出 500
        ensure_llm_scope_config("kg")

        # Initialize KTBuilder
        builder = constructor.KTBuilder(
            dataset_name,
            schema_path,
            mode=config.construction.mode,
            config=config
        )

        await send_progress_update(client_id, "construction", 20, "Starting entity-relation extraction...")

        # Build knowledge graph
        def build_graph_sync():
            return builder.build_knowledge_graph(corpus_path)

        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()

        # Run graph construction without simulated progress updates
        knowledge_graph = await loop.run_in_executor(None, build_graph_sync)

        await send_progress_update(client_id, "construction", 95, "Preparing visualization data...")
        # Load constructed graph for visualization
        graph_path = dataset_graph_path(dataset_name)
        graph_vis_data = await prepare_graph_visualization(graph_path)

        await send_progress_update(client_id, "construction", 100, "Graph construction completed!")
        # Notify completion via WebSocket
        try:
            await manager.send_message({
                "type": "complete",
                "stage": "construction",
                "message": "Graph construction completed!",
                "timestamp": datetime.now().isoformat()
            }, client_id)
        except Exception as _e:
            logger.warning(f"Failed to send completion message: {_e}")

        return GraphConstructionResponse(
            success=True,
            message="Knowledge graph constructed successfully",
            graph_data=graph_vis_data
        )

    except Exception as e:
        await send_progress_update(client_id, "construction", 0, f"Construction failed: {str(e)}")
        try:
            await manager.send_message({
                "type": "error",
                "stage": "construction",
                "message": f"Construction failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }, client_id)
        except Exception as _e:
            logger.warning(f"Failed to send error message: {_e}")
        raise HTTPException(status_code=500, detail=str(e))


async def prepare_graph_visualization(graph_path: str) -> Dict:
    """Prepare graph data for visualization"""
    try:
        if os.path.exists(graph_path):
            with open(graph_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
        else:
            return {"nodes": [], "links": [], "categories": [], "stats": {}}

        # Handle different graph data formats
        if isinstance(graph_data, list):
            # GraphRAG format: list of relationships
            return convert_graphrag_format(graph_data)
        elif isinstance(graph_data, dict) and "nodes" in graph_data:
            # Standard format: {nodes: [], edges: []}
            return convert_standard_format(graph_data)
        else:
            return {"nodes": [], "links": [], "categories": [], "stats": {}}

    except Exception as e:
        logger.error(f"Error preparing visualization: {e}")
        return {"nodes": [], "links": [], "categories": [], "stats": {}}


def convert_graphrag_format(graph_data: List) -> Dict:
    """Convert GraphRAG relationship list to ECharts format"""
    nodes_dict = {}
    links = []
    node_degree = {}

    # Extract nodes and relationships from the list
    for item in graph_data:
        if not isinstance(item, dict):
            continue

        start_node = item.get("start_node", {})
        end_node = item.get("end_node", {})
        relation = item.get("relation", "related_to")

        # Process start node
        start_id = ""
        end_id = ""
        if start_node:
            start_name = start_node.get("properties", {}).get("name", "")
            start_id = build_visual_node_id(start_node)
            if start_id and start_id not in nodes_dict:
                category = normalize_visual_category(
                    start_node.get("properties", {}).get("schema_type", start_node.get("label", "entity"))
                )
                nodes_dict[start_id] = {
                    "id": start_id,
                    "name": str(start_name)[:40],
                    "fullName": start_name,
                    "category": category,
                    "symbolSize": 25,
                    "properties": start_node.get("properties", {})
                }

        # Process end node
        if end_node:
            end_name = end_node.get("properties", {}).get("name", "")
            end_id = build_visual_node_id(end_node)
            if end_id and end_id not in nodes_dict:
                category = normalize_visual_category(
                    end_node.get("properties", {}).get("schema_type", end_node.get("label", "entity"))
                )
                nodes_dict[end_id] = {
                    "id": end_id,
                    "name": str(end_name)[:40],
                    "fullName": end_name,
                    "category": category,
                    "symbolSize": 25,
                    "properties": end_node.get("properties", {})
                }

        # Add relationship
        if start_id and end_id:
            links.append({
                "source": start_id,
                "target": end_id,
                "name": relation,
                "value": 1
            })
            node_degree[start_id] = node_degree.get(start_id, 0) + 1
            node_degree[end_id] = node_degree.get(end_id, 0) + 1

    all_nodes = list(nodes_dict.values())
    for node in all_nodes:
        degree = node_degree.get(node["id"], 0)
        node["value"] = degree
        if node["category"] == PAPER_VISUAL_CATEGORY:
            node["symbolSize"] = 18 if degree <= 2 else min(24, 18 + degree * 0.6)
        else:
            node["symbolSize"] = min(30, 16 + degree * 0.8)
    kept_nodes, kept_links = select_connected_visual_subgraph(
        nodes_dict,
        links,
        node_degree,
        max_nodes=3500,
        max_links=8000,
    )

    ordered_categories = ordered_visual_categories(node["category"] for node in kept_nodes)
    categories = []
    for i, cat_name in enumerate(ordered_categories):
        categories.append({
            "name": cat_name,
            "itemStyle": {
                "color": f"hsl({i * 360 / max(len(ordered_categories), 1)}, 70%, 60%)"
            }
        })

    return {
        "nodes": kept_nodes,
        "links": kept_links,
        "categories": categories,
        "stats": {
            "total_nodes": len(all_nodes),
            "total_edges": len(links),
            "displayed_nodes": len(kept_nodes),
            "displayed_edges": len(kept_links),
            "total_papers": sum(1 for node in all_nodes if node.get("category") == PAPER_VISUAL_CATEGORY),
            "displayed_papers": sum(1 for node in kept_nodes if node.get("category") == PAPER_VISUAL_CATEGORY),
        }
    }


def _link_priority(link: Dict, nodes_dict: Dict[str, Dict], node_degree: Dict[str, int]) -> tuple:
    source_category = normalize_visual_category(nodes_dict.get(link["source"], {}).get("category", ""))
    target_category = normalize_visual_category(nodes_dict.get(link["target"], {}).get("category", ""))
    relation = str(link.get("name", "")).strip()
    categories = {source_category, target_category}

    if source_category == AUTHOR_VISUAL_CATEGORY and relation == "撰写" and target_category == PAPER_VISUAL_CATEGORY:
        tier = 0
    elif source_category == PAPER_VISUAL_CATEGORY and relation == "发表于" and target_category == JOURNAL_VISUAL_CATEGORY:
        tier = 1
    elif source_category == AUTHOR_VISUAL_CATEGORY and relation == "隶属" and target_category == ORG_VISUAL_CATEGORY:
        tier = 2
    elif PAPER_VISUAL_CATEGORY in categories:
        tier = 3
    elif relation == "has_attribute":
        tier = 5
    elif COMMUNITY_VISUAL_CATEGORY in categories:
        tier = 6
    else:
        tier = 4

    combined_degree = node_degree.get(link["source"], 0) + node_degree.get(link["target"], 0)
    return (tier, -combined_degree, link["source"], link["target"], relation)


def select_connected_visual_subgraph(
    nodes_dict: Dict[str, Dict],
    links: List[Dict],
    node_degree: Dict[str, int],
    max_nodes: int = 3500,
    max_links: int = 8000,
) -> tuple[list[Dict], list[Dict]]:
    """Select a connectivity-first visualization subset instead of a paper-only subset."""
    if not nodes_dict or not links:
        return [], []

    category_totals = Counter(
        normalize_visual_category(node.get("category", ENTITY_VISUAL_CATEGORY))
        for node in nodes_dict.values()
    )
    category_caps = {
        PAPER_VISUAL_CATEGORY: min(category_totals.get(PAPER_VISUAL_CATEGORY, 0), 1600),
        AUTHOR_VISUAL_CATEGORY: min(category_totals.get(AUTHOR_VISUAL_CATEGORY, 0), 900),
        ORG_VISUAL_CATEGORY: min(category_totals.get(ORG_VISUAL_CATEGORY, 0), 500),
        JOURNAL_VISUAL_CATEGORY: min(category_totals.get(JOURNAL_VISUAL_CATEGORY, 0), 250),
        COMMUNITY_VISUAL_CATEGORY: min(category_totals.get(COMMUNITY_VISUAL_CATEGORY, 0), 120),
    }

    kept_node_ids = set()
    kept_links: List[Dict] = []
    category_counts: Counter = Counter()
    selected_link_keys = set()
    coverage_targets = [
        category
        for category in VISUAL_CATEGORY_COVERAGE_PRIORITY
        if category_totals.get(category, 0) > 0
    ]
    covered_categories = set()
    coverage_budget = min(24, max_links // 2) if max_links >= 8 and coverage_targets else 0

    def can_add_node(node_id: str) -> bool:
        if node_id in kept_node_ids:
            return True
        if len(kept_node_ids) >= max_nodes:
            return False
        category = normalize_visual_category(nodes_dict.get(node_id, {}).get("category", ENTITY_VISUAL_CATEGORY))
        cap = category_caps.get(category)
        return cap is None or category_counts[category] < cap

    def add_node(node_id: str) -> None:
        if node_id in kept_node_ids:
            return
        kept_node_ids.add(node_id)
        category = normalize_visual_category(nodes_dict.get(node_id, {}).get("category", ENTITY_VISUAL_CATEGORY))
        category_counts[category] += 1

    def add_link(link: Dict) -> bool:
        source = link["source"]
        target = link["target"]
        link_key = (source, target, str(link.get("name", "")))
        if link_key in selected_link_keys:
            return False
        if not can_add_node(source) or not can_add_node(target):
            return False
        add_node(source)
        add_node(target)
        kept_links.append(link)
        selected_link_keys.add(link_key)
        return True

    sorted_links = sorted(links, key=lambda link: _link_priority(link, nodes_dict, node_degree))

    for target_category in coverage_targets:
        if len(kept_links) >= coverage_budget:
            break
        if target_category in covered_categories:
            continue

        for link in sorted_links:
            source_category = normalize_visual_category(
                nodes_dict.get(link["source"], {}).get("category", ENTITY_VISUAL_CATEGORY)
            )
            target_node_category = normalize_visual_category(
                nodes_dict.get(link["target"], {}).get("category", ENTITY_VISUAL_CATEGORY)
            )
            if target_category not in {source_category, target_node_category}:
                continue
            if add_link(link):
                covered_categories.update(
                    category
                    for category in (source_category, target_node_category)
                    if category in coverage_targets
                )
                break

    for link in sorted_links:
        add_link(link)
        if len(kept_node_ids) >= max_nodes or len(kept_links) >= max_links:
            break

    if kept_node_ids and len(kept_links) < max_links:
        for link in sorted_links:
            link_key = (link["source"], link["target"], str(link.get("name", "")))
            if link_key in selected_link_keys:
                continue
            if link["source"] in kept_node_ids and link["target"] in kept_node_ids:
                kept_links.append(link)
                selected_link_keys.add(link_key)
                if len(kept_links) >= max_links:
                    break

    connected_ids = set()
    for link in kept_links:
        connected_ids.add(link["source"])
        connected_ids.add(link["target"])

    kept_nodes = [
        nodes_dict[node_id]
        for node_id in connected_ids
        if node_id in nodes_dict
    ]
    kept_nodes.sort(
        key=lambda node: (
            _link_priority(
                {"source": node["id"], "target": node["id"], "name": ""},
                nodes_dict,
                node_degree,
            )[0],
            -node_degree.get(node["id"], 0),
            node["id"],
        )
    )
    kept_links = [link for link in kept_links if link["source"] in connected_ids and link["target"] in connected_ids]
    return kept_nodes, kept_links


def convert_standard_format(graph_data: Dict) -> Dict:
    """Convert standard {nodes: [], edges: []} format to ECharts format"""
    nodes = []
    links = []
    categories = []

    # Extract unique categories
    node_types = set()
    for node in graph_data.get("nodes", []):
        node_type = normalize_visual_category(node.get("type", "entity"))
        node_types.add(node_type)

    for i, node_type in enumerate(node_types):
        categories.append({
            "name": node_type,
            "itemStyle": {
                "color": f"hsl({i * 360 / len(node_types)}, 70%, 60%)"
            }
        })

    # Process nodes
    for node in graph_data.get("nodes", []):
        nodes.append({
            "id": node.get("id", ""),
            "name": node.get("name", node.get("id", ""))[:30],
            "category": normalize_visual_category(node.get("type", "entity")),
            "value": len(node.get("attributes", [])),
            "symbolSize": min(max(len(node.get("attributes", [])) * 3 + 15, 15), 40),
            "attributes": node.get("attributes", [])
        })

    # Process edges
    for edge in graph_data.get("edges", []):
        links.append({
            "source": edge.get("source", ""),
            "target": edge.get("target", ""),
            "name": edge.get("relation", "related_to"),
            "value": edge.get("weight", 1)
        })

    return {
        "nodes": nodes[:500],  # Limit for performance
        "links": links[:1000],
        "categories": categories,
        "stats": {
            "total_nodes": len(graph_data.get("nodes", [])),
            "total_edges": len(graph_data.get("edges", [])),
            "displayed_nodes": len(nodes[:500]),
            "displayed_edges": len(links[:1000])
        }
    }


@app.post("/api/ask-question", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest, client_id: str = "default"):
    """Process question using agent mode (iterative retrieval + reasoning) and return answer."""
    try:
        if not GRAPHRAG_AVAILABLE:
            raise HTTPException(status_code=503,
                                detail="GraphRAG components not available. Please install or configure them.")
        dataset_name = request.dataset_name
        question = request.question

        await send_progress_update(client_id, "retrieval", 10, "Initializing retrieval system (agent mode)...")

        graph_path = dataset_graph_path(dataset_name)
        schema_path = get_schema_path_for_dataset(dataset_name)
        if not os.path.exists(graph_path):
            graph_path = DEMO_GRAPH_PATH
        if not os.path.exists(graph_path):
            raise HTTPException(status_code=404, detail="Graph not found. Please construct graph first.")

        # Config & components
        global config
        if config is None:
            config = get_config(BASE_CONFIG_PATH)

        ensure_llm_scope_config("rag")

        graphq = decomposer.GraphQ(dataset_name, config=config)
        kt_retriever = retriever.KTRetriever(
            dataset_name,
            graph_path,
            recall_paths=config.retrieval.recall_paths,
            schema_path=schema_path,
            top_k=config.retrieval.top_k_filter,
            mode="agent",  # force agent mode
            config=config
        )

        await send_progress_update(client_id, "retrieval", 40, "Building indices...")
        loop = asyncio.get_running_loop()
        # Offload index building to thread executor to avoid blocking event loop
        await loop.run_in_executor(None, kt_retriever.build_indices)

        # Notify QA start via WS so frontend can show immediate progress
        try:
            await manager.send_message({
                "type": "qa_update",
                "stage": "start",
                "message": "Question processing started",
                "dataset": dataset_name,
                "question": question,
                "timestamp": datetime.now().isoformat()
            }, client_id)
            await asyncio.sleep(0)
        except Exception as _e:
            logger.debug(f"QA start ws send failed: {_e}")


        # Use unified agent runner
        def _sync_callback(event_type, **kwargs):
            try:
                msg = {"type": "qa_update", "stage": event_type, "timestamp": datetime.now().isoformat(), **kwargs}
                if event_type == "retrieval_progress":
                    progress = kwargs.get("progress", 0)
                    msg_text = kwargs.get("message", "Processing...")
                    asyncio.run_coroutine_threadsafe(
                        send_progress_update(client_id, "retrieval", progress, msg_text),
                        loop
                    )
                else:
                    asyncio.run_coroutine_threadsafe(
                        manager.send_message(msg, client_id),
                        loop
                    )
            except Exception as e:
                pass

        max_steps = getattr(getattr(config.retrieval, 'agent', object()), 'max_steps', 3)
        loop = asyncio.get_running_loop()
        
        # Offload logic to thread executor to avoid blocking event loop
        result = await loop.run_in_executor(
            None, 
            lambda: agent_runner.run_agent_retrieval(
                graphq, 
                kt_retriever, 
                question, 
                schema_path, 
                max_steps, 
                config.retrieval.top_k_filter,
                callback=_sync_callback
            )
        )

        final_answer = result['answer']
        sub_questions = result['sub_questions']
        final_triples = result['retrieved_triples']
        final_chunk_contents = result['retrieved_chunks']
        reasoning_steps = result['reasoning_steps']
        decompose_fallback = result['decompose_fallback']
        decompose_error = result['decompose_error']

        await send_progress_update(client_id, "retrieval", 100, "Answer generation completed!")

        try:
            await manager.send_message({
                "type": "qa_complete",
                "answer_preview": (final_answer or "")[:300],
                "sub_questions_count": len(sub_questions),
                "triples_final_count": len(final_triples),
                "chunks_final_count": len(final_chunk_contents),
                "timestamp": datetime.now().isoformat()
            }, client_id)
        except Exception as _e:
            logger.debug(f"QA complete ws send failed: {_e}")

        visualization_data = {
            "subqueries": prepare_subquery_visualization(sub_questions, reasoning_steps),
            "knowledge_graph": prepare_retrieved_graph_visualization(final_triples),
            "reasoning_flow": prepare_reasoning_flow_visualization(reasoning_steps),
            "retrieval_details": {
                "total_triples": len(final_triples),
                "total_chunks": len(final_chunk_contents),
                "sub_questions_count": len(sub_questions),
                "triples_by_subquery": [s.get("triples_count", 0) for s in reasoning_steps if s.get("type") == "sub_question"],
                "diagnostics_by_subquery": [s.get("retrieval_diagnostics", {}) for s in reasoning_steps if s.get("type") == "sub_question"],
            }
        }

        return QuestionResponse(
            answer=final_answer,
            sub_questions=sub_questions,
            retrieved_triples=final_triples,
            retrieved_chunks=final_chunk_contents,
            reasoning_steps=reasoning_steps,
            visualization_data=visualization_data,
            decompose_fallback=decompose_fallback,
            decompose_error=decompose_error,
            schema_path_used=schema_path
        )
    except Exception as e:
        await send_progress_update(client_id, "retrieval", 0, f"Question answering failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def prepare_subquery_visualization(sub_questions: List[Dict], reasoning_steps: List[Dict]) -> Dict:
    """Prepare subquery visualization"""
    nodes = [{"id": "original", "name": "Original Question", "category": "question", "symbolSize": 40}]
    links = []

    for i, sub_q in enumerate(sub_questions):
        sub_id = f"sub_{i}"
        nodes.append({
            "id": sub_id,
            "name": sub_q.get("sub-question", "")[:20] + "...",
            "category": "sub_question",
            "symbolSize": 30
        })
        links.append({"source": "original", "target": sub_id, "name": "decomposed to"})

    return {
        "nodes": nodes,
        "links": links,
        "categories": [
            {"name": "question", "itemStyle": {"color": "#ff6b6b"}},
            {"name": "sub_question", "itemStyle": {"color": "#4ecdc4"}}
        ]
    }


def prepare_retrieved_graph_visualization(triples: List[str]) -> Dict:
    """Prepare retrieved knowledge visualization with main-chain prioritization."""
    fallback_entity_nodes = set()

    def _normalize_category(raw_category: str) -> str:
        category = str(raw_category or "").strip()
        return normalize_visual_category(category) if category else ""

    def _split_triple_content(content: str) -> List[str]:
        parts = []
        current = []
        bracket_depth = 0
        paren_depth = 0

        for ch in content:
            if ch == "[":
                bracket_depth += 1
            elif ch == "]" and bracket_depth > 0:
                bracket_depth -= 1
            elif ch == "(":
                paren_depth += 1
            elif ch == ")" and paren_depth > 0:
                paren_depth -= 1

            if ch == "," and bracket_depth == 0 and paren_depth == 0 and len(parts) < 2:
                parts.append("".join(current).strip())
                current = []
                continue
            current.append(ch)

        if current:
            parts.append("".join(current).strip())
        return parts

    def _parse_entity(raw: str) -> tuple[str, str, str]:
        text_raw = str(raw or "").strip()
        meta_blocks = re.findall(r"\[([^\]]+)\]", text_raw)
        schema_type = ""
        label = ""
        description = ""
        for block in meta_blocks:
            match = re.search(r"schema_type:\s*([^,\]]+)", block)
            if match:
                schema_type = match.group(1).strip()
            match = re.search(r"label:\s*([^,\]]+)", block)
            if match and not label:
                label = match.group(1).strip()
            match = re.search(r"description:\s*(.*?)(?:,\s+[A-Za-z_][A-Za-z0-9_ ]*:|$)", block)
            if match and not description:
                description = match.group(1).strip()

        clean_name = re.sub(r"\s*\[[^\]]*\]\s*", " ", text_raw).strip()
        clean_name = re.sub(r"\s+", " ", clean_name)
        if not schema_type:
            schema_type = _normalize_category(label)
        if not schema_type:
            schema_type = "entity"
            if clean_name:
                fallback_entity_nodes.add(clean_name)
        return clean_name, _normalize_category(schema_type), description

    def _normalize_node_size(category: str) -> int:
        cat = str(category or "").strip().lower()
        if cat in {"\u8bba\u6587", "paper"}:
            return 30
        if cat in {"\u4f5c\u8005", "person"}:
            return 26
        if cat in {"\u673a\u6784", "organization"}:
            return 24
        if cat in {"\u7814\u7a76\u65b9\u6cd5", "method", "\u6280\u672f"}:
            return 24
        if cat in {"\u4e3b\u9898\u793e\u533a", "community"}:
            return 28
        if cat in {"\u5173\u952e\u8bcd", "keyword", "\u5c5e\u6027", "attribute"}:
            return 22
        return 22

    color_pool = [
        "#60a5fa", "#34d399", "#f59e0b", "#a78bfa", "#f472b6",
        "#22d3ee", "#f87171", "#84cc16", "#38bdf8", "#e879f9",
    ]
    anchor_categories = {
        "\u8bba\u6587", "paper", "\u4f5c\u8005", "person", "\u7814\u7a76\u65b9\u6cd5", "method", "\u673a\u6784", "organization",
        "\u6559\u80b2\u9886\u57df", "\u7814\u7a76\u4e3b\u9898", "\u6559\u5b66\u573a\u666f", "\u671f\u520a", "\u6280\u672f",
        "\u4e3b\u9898\u793e\u533a", "\u5173\u952e\u8bcd", "\u5c5e\u6027",
    }

    node_map: Dict[str, Dict] = {}
    links = []
    link_seen = set()

    for triple in triples or []:
        source = relation = target = None
        try:
            if isinstance(triple, str) and triple.startswith("[") and triple.endswith("]"):
                try:
                    parts = ast.literal_eval(triple)
                except Exception:
                    parts = None
                if isinstance(parts, (list, tuple)) and len(parts) == 3:
                    source, relation, target = str(parts[0]), str(parts[1]), str(parts[2])

            if source is None and isinstance(triple, str):
                normalized = re.sub(r"\s*\[score:\s*[-+]?\d*\.?\d+\]\s*$", "", triple.strip())
                if normalized.startswith("(") and normalized.endswith(")"):
                    body = normalized[1:-1].strip()
                    fields = _split_triple_content(body)
                    if len(fields) >= 3:
                        source, relation, target = fields[0], fields[1], fields[2]

            if not (source and relation and target):
                continue

            src_name, src_category, src_description = _parse_entity(source)
            tgt_name, tgt_category, tgt_description = _parse_entity(target)
            rel_name = str(relation).strip()
            if not src_name or not tgt_name or not rel_name:
                continue

            for node_name, node_category, node_description in (
                (src_name, src_category, src_description),
                (tgt_name, tgt_category, tgt_description),
            ):
                existing = node_map.get(node_name)
                if existing is None:
                    node_map[node_name] = {
                        "id": node_name,
                        "name": node_name[:20],
                        "category": node_category,
                        "symbolSize": _normalize_node_size(node_category),
                        "description": node_description,
                        "degree": 0,
                        "component_id": -1,
                        "is_fallback_entity": node_name in fallback_entity_nodes,
                    }
                    existing = node_map[node_name]
                elif existing.get("category") == "entity" and node_category != "entity":
                    existing["category"] = node_category
                    existing["symbolSize"] = _normalize_node_size(node_category)
                if node_description and not existing.get("description"):
                    existing["description"] = node_description

            link_key = (src_name, rel_name, tgt_name)
            if link_key in link_seen:
                continue
            link_seen.add(link_key)
            links.append({"source": src_name, "target": tgt_name, "name": rel_name})
        except Exception:
            continue

    if not node_map or not links:
        return {"nodes": [], "links": [], "categories": []}

    adjacency: Dict[str, set] = {node_id: set() for node_id in node_map}
    for link in links:
        source = str(link["source"])
        target = str(link["target"])
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)

    for node_id, neighbors in adjacency.items():
        if node_id in node_map:
            node_map[node_id]["degree"] = len(neighbors)

    components = []
    visited = set()
    for node_id in node_map:
        if node_id in visited:
            continue
        stack = [node_id]
        component_nodes = []
        visited.add(node_id)
        while stack:
            current = stack.pop()
            component_nodes.append(current)
            for neighbor in adjacency.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(component_nodes)

    component_links_map: Dict[int, List[Dict]] = {}
    for comp_idx, comp_nodes in enumerate(components):
        node_set = set(comp_nodes)
        component_links_map[comp_idx] = [
            link for link in links
            if str(link["source"]) in node_set and str(link["target"]) in node_set
        ]

    best_component_id = None
    best_score = None
    for comp_idx, comp_nodes in enumerate(components):
        comp_links = component_links_map.get(comp_idx, [])
        if not comp_links:
            continue
        comp_node_objs = [node_map[node_id] for node_id in comp_nodes if node_id in node_map]
        non_entity_count = sum(1 for node in comp_node_objs if str(node.get("category", "")).lower() != "entity")
        anchor_count = sum(
            1 for node in comp_node_objs
            if str(node.get("category", "")).strip() in anchor_categories
            or str(node.get("category", "")).strip().lower() in anchor_categories
        )
        fallback_leaf_count = sum(
            1 for node in comp_node_objs
            if node.get("is_fallback_entity") and int(node.get("degree", 0)) <= 1
        )
        score = len(comp_links) * 100 + non_entity_count * 10 + anchor_count * 5 - fallback_leaf_count * 3
        if best_score is None or score > best_score:
            best_score = score
            best_component_id = comp_idx

    # Strategy change: No longer keep only the 'best' component.
    # Keep as many relevant components as possible to form a "continuous" evidence subgraph.
    kept_nodes = []
    kept_links = []
    category_order = []
    
    # Sort components by score descending to prioritize quality
    sorted_component_indices = sorted(
        range(len(components)), 
        key=lambda i: (
            len(component_links_map.get(i, [])) * 100 + 
            sum(1 for nid in components[i] if node_map.get(nid, {}).get("category", "") != "entity") * 10
        ),
        reverse=True
    )

    total_nodes_count = 0
    max_total_nodes = 150 # Reasonable limit for ECharts
    
    for comp_idx in sorted_component_indices:
        comp_nodes = components[comp_idx]
        # If we have at least one good component and the next one pushes us over limit, stop.
        if total_nodes_count > 0 and total_nodes_count + len(comp_nodes) > max_total_nodes:
            continue
            
        for node_id in comp_nodes:
            node = node_map[node_id]
            node["component_id"] = comp_idx
            kept_nodes.append(node)
            cat = node.get("category", "entity")
            if cat not in category_order:
                category_order.append(cat)
        
        kept_links.extend(component_links_map.get(comp_idx, []))
        total_nodes_count += len(comp_nodes)

    if not kept_nodes:
        return {"nodes": [], "links": [], "categories": []}

    categories = [
        {"name": category, "itemStyle": {"color": color_pool[idx % len(color_pool)]}}
        for idx, category in enumerate(category_order)
    ]

    logger.info(
        "Retrieved subgraph parse stats: raw_nodes=%d raw_links=%d components=%d kept_component=%s kept_nodes=%d kept_links=%d fallback_entity_nodes=%d",
        len(node_map),
        len(links),
        len(components),
        best_component_id,
        len(kept_nodes),
        len(kept_links),
        len(fallback_entity_nodes),
    )

    return {
        "nodes": kept_nodes,
        "links": kept_links,
        "categories": categories or [{"name": "entity", "itemStyle": {"color": "#60a5fa"}}],
    }


def prepare_reasoning_flow_visualization(reasoning_steps: List[Dict]) -> Dict:
    """Prepare reasoning flow visualization"""
    steps_data = []
    for i, step in enumerate(reasoning_steps):
        steps_data.append({
            "step": i + 1,
            "type": step.get("type", "unknown"),
            "question": step.get("question", "")[:50],
            "triples_count": step.get("triples_count", 0),
            "chunks_count": step.get("chunks_count", 0),
            "processing_time": step.get("processing_time", 0)
        })

    return {
        "steps": steps_data,
        "timeline": [step["processing_time"] for step in steps_data]
    }


@app.get("/api/datasets")
async def get_datasets():
    """Get list of available datasets"""
    datasets = []

    # Check uploaded datasets
    upload_dir = str(UPLOADED_DATA_DIR)
    if os.path.exists(upload_dir):
        for item in os.listdir(upload_dir):
            item_path = os.path.join(upload_dir, item)
            if os.path.isdir(item_path):
                corpus_path = os.path.join(item_path, "corpus.json")
                if os.path.exists(corpus_path):
                    graph_path = dataset_graph_path(item)
                    status = "ready" if os.path.exists(graph_path) else "needs_construction"
                    has_custom_schema = os.path.exists(dataset_schema_path(item))
                    datasets.append({
                        "name": item,
                        "type": "uploaded",
                        "status": status,
                        "has_custom_schema": has_custom_schema
                    })
                else:
                    logger.warning(
                        "Skipping uploaded dataset '%s' because corpus.json is missing in %s",
                        item,
                        item_path,
                    )

    # Add demo dataset
    demo_corpus = DEMO_CORPUS_PATH
    if os.path.exists(demo_corpus):
        demo_graph = DEMO_GRAPH_PATH
        status = "ready" if os.path.exists(demo_graph) else "needs_construction"
        datasets.append({
            "name": "demo",
            "type": "demo",
            "status": status,
            "has_custom_schema": False
        })

    return {"datasets": datasets}


@app.post("/api/datasets/{dataset_name}/schema")
async def upload_schema(dataset_name: str, schema_file: UploadFile = File(...)):
    """Upload a custom schema JSON for a dataset."""
    try:
        if dataset_name == "demo":
            raise HTTPException(status_code=400, detail="Cannot upload schema for demo dataset")
        if not schema_file.filename.lower().endswith('.json'):
            raise HTTPException(status_code=400, detail="Schema file must be a .json file")

        content = await schema_file.read()
        try:
            schema_text = decode_bytes_with_detection(content)
            data = json.loads(schema_text)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="Schema JSON must be an object")

        os.makedirs(SCHEMAS_DIR, exist_ok=True)
        save_path = dataset_schema_path(dataset_name)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return {"success": True, "message": "Schema uploaded successfully", "dataset_name": dataset_name}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload schema: {str(e)}")


@app.delete("/api/datasets/{dataset_name}")
async def delete_dataset(dataset_name: str):
    """Delete a dataset and all its associated files"""
    try:
        if dataset_name == "demo":
            raise HTTPException(status_code=400, detail="Cannot delete demo dataset")

        deleted_files = []

        # Delete dataset directory
        dataset_dir = uploaded_dataset_dir(dataset_name)
        if os.path.exists(dataset_dir):
            import shutil
            shutil.rmtree(dataset_dir)
            deleted_files.append(dataset_dir)

        # Delete graph file
        graph_path = dataset_graph_path(dataset_name)
        if os.path.exists(graph_path):
            os.remove(graph_path)
            deleted_files.append(graph_path)

        # Delete schema file (if dataset-specific)
        schema_path = dataset_schema_path(dataset_name)
        if os.path.exists(schema_path):
            os.remove(schema_path)
            deleted_files.append(schema_path)

        # Delete cache files
        cache_dir = str(FAISS_CACHE_ROOT / dataset_name)
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
            deleted_files.append(cache_dir)

        # Delete chunk files
        chunk_file = dataset_chunk_path(dataset_name)
        if os.path.exists(chunk_file):
            os.remove(chunk_file)
            deleted_files.append(chunk_file)

        return {
            "success": True,
            "message": f"Dataset '{dataset_name}' deleted successfully",
            "deleted_files": deleted_files
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")


@app.post("/api/datasets/{dataset_name}/reconstruct")
async def reconstruct_dataset(dataset_name: str, client_id: str = "default"):
    """Reconstruct graph for an existing dataset"""
    try:
        if not GRAPHRAG_AVAILABLE:
            raise HTTPException(status_code=503,
                                detail="GraphRAG components not available. Please install or configure them.")
        # Check if dataset exists
        corpus_path = dataset_corpus_path(dataset_name)
        if not os.path.exists(corpus_path):
            if dataset_name == "demo":
                corpus_path = DEMO_CORPUS_PATH
            else:
                raise HTTPException(status_code=404, detail="Dataset not found")

        await send_progress_update(client_id, "reconstruction", 5, "Starting reconstruction...")

        # Delete existing graph file
        graph_path = dataset_graph_path(dataset_name)
        if os.path.exists(graph_path):
            os.remove(graph_path)
            await send_progress_update(client_id, "reconstruction", 15, "Old graph file deleted...")

        # Delete existing cache files
        cache_dir = str(FAISS_CACHE_ROOT / dataset_name)
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
            await send_progress_update(client_id, "reconstruction", 25, "Cache files cleared...")

        await send_progress_update(client_id, "reconstruction", 35, "Reinitializing graph builder...")

        # Initialize config
        global config
        if config is None:
            config = get_config(BASE_CONFIG_PATH)

        ensure_llm_scope_config("kg")

        # Choose schema: dataset-specific or default demo
        schema_path = get_schema_path_for_dataset(dataset_name)

        # Initialize KTBuilder
        builder = constructor.KTBuilder(
            dataset_name,
            schema_path,
            mode=config.construction.mode,
            config=config
        )

        await send_progress_update(client_id, "reconstruction", 50, "Rebuilding knowledge graph...")

        # Build knowledge graph
        def build_graph_sync():
            return builder.build_knowledge_graph(corpus_path)

        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()

        # Run graph reconstruction without simulated progress updates
        knowledge_graph = await loop.run_in_executor(None, build_graph_sync)

        await send_progress_update(client_id, "reconstruction", 100, "Graph reconstruction completed!")
        # Notify completion via WebSocket
        try:
            await manager.send_message({
                "type": "complete",
                "stage": "reconstruction",
                "message": "Graph reconstruction completed!",
                "timestamp": datetime.now().isoformat()
            }, client_id)
        except Exception as _e:
            logger.warning(f"Failed to send completion message: {_e}")

        return {
            "success": True,
            "message": "Dataset reconstructed successfully",
            "dataset_name": dataset_name
        }

    except Exception as e:
        await send_progress_update(client_id, "reconstruction", 0, f"Reconstruction failed: {str(e)}")
        try:
            await manager.send_message({
                "type": "error",
                "stage": "reconstruction",
                "message": f"Reconstruction failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }, client_id)
        except Exception as _e:
            logger.warning(f"Failed to send error message: {_e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph/{dataset_name}")
async def get_graph_data(dataset_name: str):
    """Get graph visualization data"""
    graph_path = dataset_graph_path(dataset_name)

    if not os.path.exists(graph_path):
        # Return demo data
        return {
            "nodes": [
                {"id": "node1", "name": "Example Entity 1", "category": "person", "value": 5, "symbolSize": 25},
                {"id": "node2", "name": "Example Entity 2", "category": "location", "value": 3, "symbolSize": 20},
            ],
            "links": [
                {"source": "node1", "target": "node2", "name": "located_in", "value": 1}
            ],
            "categories": [
                {"name": "person", "itemStyle": {"color": "#ff6b6b"}},
                {"name": "location", "itemStyle": {"color": "#4ecdc4"}},
            ],
            "stats": {"total_nodes": 2, "total_edges": 1, "displayed_nodes": 2, "displayed_edges": 1}
        }

    return await prepare_graph_visualization(graph_path)


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    os.makedirs(UPLOADED_DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_GRAPHS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_CHUNKS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LOGS_DIR, exist_ok=True)
    os.makedirs(SCHEMAS_DIR, exist_ok=True)
    os.makedirs(FAISS_CACHE_ROOT, exist_ok=True)

    logger.info("Youtu-GraphRAG Unified Interface initialized")


if __name__ == "__main__":
    install_interrupt_guard("Youtu-GraphRAG backend")
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        logger.warning("Backend interrupted by user. Forcing shutdown...")
        terminate_process_tree()
