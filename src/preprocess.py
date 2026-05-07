"""Preprocessing data.

We use ONE continuous feature: `gross_floor_area_buildings_sq_ft` (the
building's total floor area in square feet). The label is
`total_ghg_emissions_metric_tons_co2e`: the building's total annual
greenhouse gas emissions in metric tons of CO2-equivalent.

Why this feature?
    Larger buildings consume more energy and therefore emit more GHG.
    This gives the model something real to learn, but the R_2 will be modest
    because many other factors matter (building type, age, fuel mix, etc.).

Why GHG emissions as the target?
    Predicting a building's carbon footprint from its characteristics is
    directly relevant to climate policy, ESG reporting, and urban planning.
    Unlike EUI (which is mathematically derived from energy use / floor area),
    GHG emissions depend on the fuel MIX (gas vs. electricity vs. steam),
    so using energy-use columns as features is meaningful.

Hint 1: using more features might improve the predictive performance.
Hint 2: make sure not to use linearly correlated features. Be cautious in analyzing which features to use!
"""
import pandas as pd

FEATURE_COLS = ["gross_floor_area_buildings_sq_ft"]
LABEL_COL = "total_ghg_emissions_metric_tons_co2e"


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Select feature and label columns, drop rows with missing values."""
    out = df.dropna(subset=FEATURE_COLS + [LABEL_COL]).copy()
    out[LABEL_COL] = pd.to_numeric(out[LABEL_COL], errors="coerce")
    out = out[out[LABEL_COL] > 0]
    return out[FEATURE_COLS + [LABEL_COL, "data_year"]].reset_index(drop=True)
