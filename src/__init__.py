"""Function calling application using Qwen LLM"""

import os

# `python -m src` runs this module before `src.__main__`, which is the last
# moment these settings can be read: the Hugging Face stack samples them once,
# at import time, and the SDK pulls that stack in.
#
# The weight-loading progress bar and the Hub's advisory headers say nothing
# about the calls this program decodes, and both land on the console next to
# the warnings that do matter. `setdefault` keeps an explicit choice made by
# the caller's environment.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
