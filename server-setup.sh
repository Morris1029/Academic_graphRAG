#!/usr/bin/env bash
set -euo pipefail

echo "[1/6] Installing system packages..."
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y git default-jre antiword libreoffice-writer libreoffice-calc
elif command -v yum >/dev/null 2>&1; then
  sudo yum install -y git java-11-openjdk antiword libreoffice
else
  echo "Unsupported package manager. Install git, Java, antiword, and LibreOffice manually."
fi

echo "[2/6] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "[3/6] Upgrading pip..."
python -m pip install --upgrade pip

echo "[4/6] Installing server requirements..."
pip install -r requirements-server.txt

echo "[5/6] Downloading spaCy models..."
python -m spacy download zh_core_web_lg
python -m spacy download en_core_web_lg

echo "[6/6] Setup complete."
echo "Run: source venv/bin/activate && python backend.py"
