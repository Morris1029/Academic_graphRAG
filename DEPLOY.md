# Linux Deployment Guide

## 1. System dependencies

### Ubuntu / Debian
```bash
sudo apt-get update
sudo apt-get install -y git default-jre antiword libreoffice-writer libreoffice-calc
```

### CentOS / RHEL / Rocky / Alma
```bash
sudo yum install -y git java-11-openjdk antiword libreoffice
```

## 2. Python environment
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-server.txt
```

Optional parsers:

```bash
pip install -r requirements-optional.txt
```

## 3. spaCy models
```bash
python -m spacy download zh_core_web_lg
python -m spacy download en_core_web_lg
```

## 4. Environment file
Copy `.env.example` to `.env` and set your LLM credentials.

## 5. Start the backend
```bash
python backend.py
```

The backend listens on `0.0.0.0:8000`.
