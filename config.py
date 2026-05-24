# config.py — all environment variables and constants live here

import os
from dotenv import load_dotenv

load_dotenv()

# LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.3-70b-versatile"

# Database
DATABASE_URL = os.getenv("DATABASE_URL")

# FAISS
FAISS_INDEX_PATH = "floatchat.faiss"
FAISS_DOCS_PATH = "floatchat_docs.json"

# Embeddings
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

# Argopy — Indian Ocean bounding box
DEFAULT_LAT_RANGE = [-40, 30]
DEFAULT_LON_RANGE = [20, 120]
DEFAULT_PRESSURE_RANGE = [0, 200]

# Multi-agent execution controls
MAX_PLAN_STEPS = 6
MAX_REPLAN_ATTEMPTS = 2

# Failure thresholds:
# count == threshold → replan
# count >  threshold → unrecoverable
FAILURE_THRESHOLDS = {
    "execution": 3,
    "dependency": 2,
    "strategy": 1,
    "invalidation": 1,
}