"""Load Chicago Energy Benchmarking data from the bundled CSV.

The skeleton ships with `data/data_full.csv` — a sample of Chicago building
energy benchmarking data. This is enough to verify the pipeline works.

In Tutorial 2 you will REPLACE this with a real download from the
Chicago Open Data Portal's SODA API:
    https://data.cityofchicago.org/resource/xq83-jr8c.json

No authentication required.
"""
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/data_full.csv")

def load_data() -> pd.DataFrame:
    """Load the bundled CSV."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run from the project root directory."
        )
    return pd.read_csv(DATA_PATH)
