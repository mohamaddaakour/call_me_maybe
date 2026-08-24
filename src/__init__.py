"""Function calling application using Qwen LLM"""

import os
from pathlib import Path

# Ignores warning releated to authentication with hugging face
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

os.environ.setdefault(
    "HF_HOME", str(Path(__file__).resolve().parent.parent / ".hf-cache")
)
