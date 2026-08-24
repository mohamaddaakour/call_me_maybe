"""Function calling application using Qwen LLM"""

import os
from pathlib import Path

# `python -m src` runs this module before `src.__main__`, which is the last
# moment these settings can be read: the Hugging Face stack samples them once,
# at import time, and the SDK pulls that stack in.
#
# The weight-loading progress bar and the Hub's advisory headers say nothing
# about the calls this program decodes, and both land on the console next to
# the warnings that do matter.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

# Keep the downloaded weights beside the environment that runs them, so a
# clone carries everything it needs and deleting the clone reclaims all of
# it. Without this the weights land in the home directory of whoever runs
# the program, which is the one disk on a lab machine that has no room.
#
# Every setting here is a default: an HF_HOME already exported by the caller
# wins, so a machine that has the weights cached elsewhere reuses them
# instead of downloading a second copy.
os.environ.setdefault(
    "HF_HOME", str(Path(__file__).resolve().parent.parent / ".hf-cache")
)
