"""Configuration for the AI Revision Coach."""

import os
from dotenv import load_dotenv

load_dotenv()

# LLM Provider: "gemini" or "azure"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

# Google Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_GENERATION_MODEL = "gemini-2.5-flash"
GEMINI_EVALUATION_MODEL = "gemini-2.5-flash"

# Azure OpenAI (fallback)
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

# Embedding
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ChromaDB
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "video_transcripts"

# Chunking
CHUNK_SIZE = 300  # tokens approx (using ~4 chars per token)
CHUNK_OVERLAP = 50

# Retrieval
TOP_K_RETRIEVAL = 5

# Quiz
BASE_QUIZ_LENGTH = 5
MIN_QUIZ_LENGTH = 3
MAX_QUIZ_LENGTH = 5
ADAPTATION_INTERVAL = 2  # adapt after every N questions

# Mastery
DEFAULT_MASTERY = 0.3  # initial mastery for unseen topics/concepts

# Difficulty thresholds
EASY_THRESHOLD = 0.3
MEDIUM_THRESHOLD = 0.6

# Recommendation scoring weights
REC_WEIGHT_SEMANTIC = 0.5
REC_WEIGHT_TOPIC_OVERLAP = 0.3
REC_WEIGHT_FRESHNESS = 0.2

# Freshness
FRESHNESS_RECENCY_DAYS = 3  # topics quizzed within this window get penalized

# Data paths
DATA_DIR = "./data"
VIDEOS_FILE = f"{DATA_DIR}/videos.json"
LEARNER_FILE = f"{DATA_DIR}/learner_profile.json"
METRICS_FILE = f"{DATA_DIR}/metrics.json"
