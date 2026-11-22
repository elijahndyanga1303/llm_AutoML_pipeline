import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

RUNS_DIR = Path(os.getenv("RUNS_DIR", "./runs")).resolve()
RUNS_DIR.mkdir(parents=True, exist_ok=True)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()
_DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-120b",
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-5-5",
}
LLM_MODEL = os.getenv("LLM_MODEL") or _DEFAULT_MODELS.get(LLM_PROVIDER, "openai/gpt-oss-120b")

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
EDA_SAMPLE_ROWS = int(os.getenv("EDA_SAMPLE_ROWS", "50000"))
API_KEY = os.getenv("API_KEY", "").strip()

# How many times the LLM may retry after its output fails validation
MAX_LLM_RETRIES = 3
