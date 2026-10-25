"""Project-wide settings. Drug list is the scope of the study."""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional at import time
    pass

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"

# Generic names as they appear in openFDA's harmonized `openfda.generic_name` field.
# Brand names are handled by the agent's normalization step in Phase 3.
HER2_DRUGS: list[str] = [
    "trastuzumab",
    "pertuzumab",
    "lapatinib",
    "neratinib",
    "tucatinib",
    "ado-trastuzumab emtansine",
    "fam-trastuzumab deruxtecan-nxki",
]

CTGOV_BASE = "https://clinicaltrials.gov/api/v2"
OPENFDA_EVENT_URL = "https://api.fda.gov/drug/event.json"

OPENFDA_API_KEY = os.getenv("OPENFDA_API_KEY") or None
USER_AGENT = "trial-signal/0.1 (portfolio project)"
