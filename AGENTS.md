# AGENTS.md - Youtu-GraphRAG Development Guide

This file contains essential information for agentic coding agents working in the Youtu-GraphRAG repository.

## Build & Test Commands

### Environment Setup
```bash
# Check environment completeness
python check_environment.py

# Install dependencies
pip install -r requirements.txt

# Download spaCy models (if needed)
python -m spacy download zh_core_web_lg
python -m spacy download en_core_web_lg
```

### Running Tests
```bash
# Run individual test files (no pytest framework found)
python test_glm.py
python test_glm01.py
python test_glm02.py
python test_glm_muti.py
python test_glm_long.py
python test_glmSim.py

# Run main application
python main.py --config config/base_config.yaml --datasets demo

# Run backend server
python backend.py

# Environment check
python check_environment.py
```

### Code Quality Tools
```bash
# Format code (black 25.1.0 is available)
black .

# Sort imports (isort 5.13.2 is available)
isort .

# Type checking (mypy extensions available)
mypy .

# Run all quality tools together
black . && isort . && mypy .
```

## Project Structure

### Core Components
- `main.py` - Primary entry point for GraphRAG pipeline
- `backend.py` - FastAPI web server for UI interface
- `models/` - Core GraphRAG components
  - `agents/` - Multi-agent system (orchestrator, c_agent, r_agent, a_agent, s_agent)
  - `retriever/` - Retrieval components (enhanced_kt_retriever, agentic_decomposer)
  - `constructor/` - Knowledge graph construction (kt_gen)
- `utils/` - Utilities (logger, document_parser, call_llm_api, eval)
- `config/` - Configuration management (config_loader, base_config.yaml)

### Data Flow
1. Document upload → `backend.py` → corpus generation
2. Graph construction → `models/constructor/kt_gen.py` → knowledge graph
3. Question answering → `models/retriever/` → agent-based reasoning
4. Results → Frontend visualization

## Code Style Guidelines

### Import Organization
```python
# Standard library imports first
import os
import sys
import json
from typing import List, Dict, Optional

# Third-party imports
import yaml
import torch
from fastapi import FastAPI
from transformers import AutoModel

# Local imports
from models.agents.orchestrator import GLMOrchestrator
from utils.logger import logger
from config import get_config
```

### Type Hints
- Use type hints consistently (project uses `typing` module)
- Define complex types with `TypedDict` for agent states
- Use `Optional` for nullable parameters
- Return type hints for all public functions

### Naming Conventions
- **Classes**: `PascalCase` (e.g., `GLMOrchestrator`, `KTBuilder`)
- **Functions/Variables**: `snake_case` (e.g., `build_knowledge_graph`, `dataset_name`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_WORKERS`, `DEFAULT_MODE`)
- **Private methods**: prefix with `_` (e.g., `_build_graph`, `_detect_encoding`)

### Error Handling
```python
# Use try-except blocks with specific exceptions
try:
    result = some_operation()
except ImportError as e:
    logger.error(f"Failed to import module: {e}")
    return None
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise

# Use graceful fallbacks
if not os.path.exists(path):
    logger.warning(f"Path not found: {path}, using default")
    path = default_path
```

### Logging
- Use the centralized logger: `from utils.logger import logger`
- Log levels: `logger.info()`, `logger.warning()`, `logger.error()`, `logger.debug()`
- Include context in log messages
- Use emojis for visual clarity (as seen in codebase)

### Configuration Management
- Load configs via `from config import get_config`
- Use `ConfigManager` for structured access
- Override configs with JSON strings when needed
- Dataset-specific configs in `config/base_config.yaml`

### Async/Await Patterns
- Backend uses FastAPI with async endpoints
- Use `await` for I/O operations
- Run CPU-bound tasks in executors: `await loop.run_in_executor(None, func)`

### Agent System Patterns
- Use `TypedDict` for agent states
- Implement state machines with `StateGraph`
- Follow the C-R-A-S pattern (Comprehend-Retrieve-Answer-Summarize)
- Use LangGraph for workflow orchestration

### File Organization
- Keep related functionality in same module
- Use `__init__.py` for package imports
- Separate utilities from core logic
- Configuration files in `config/` directory

### Documentation
- Use docstrings for classes and major functions
- Include parameter types and return values
- Add inline comments for complex logic
- Use Chinese comments where appropriate (project has mixed language)

### Testing Patterns
- Test files named `test_*.py` (no pytest framework detected)
- Use simple assertion patterns
- Test with demo dataset: `cfg.active_dataset = "demo"`
- Include timing measurements for performance

### Linting & Formatting
- Use `black .` for code formatting (25.1.0 available)
- Use `isort .` for import sorting (5.13.2 available)
- Use `mypy .` for type checking (extensions available)
- Run `black . && isort . && mypy .` to apply all formatting

## Development Workflow

1. **Environment Check**: Run `python check_environment.py` first
2. **Code Changes**: Follow style guidelines above
3. **Testing**: Run relevant test files
4. **Integration**: Test with `python main.py --datasets demo`
5. **Backend**: Test UI with `python backend.py`

## Important Notes

- This is a GraphRAG (Graph Retrieval-Augmented Generation) system
- Supports both "agent" and "noagent" modes
- Uses FAISS for vector indexing
- Supports multiple document formats (PDF, DOCX, TXT, JSON)
- Has WebSocket support for real-time progress updates
- Uses spaCy for NLP processing
- Configuration-driven architecture