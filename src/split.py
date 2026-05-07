"""Train/test split.

Validation is intentionally LEFT BLANK — implementing it is one of
your first tasks.
"""
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(df: pd.DataFrame, test_frac: float = 0.20, seed: int = 42):
    """Simple random 80/20 split, stratified is not needed for regression."""
    train, test = train_test_split(df, test_size=test_frac, random_state=seed)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def validate(model, df) -> dict:
    """TODO: implement a validation strategy.

    Options to consider:
        - Hold out a separate validation set from the training data.
        - K-fold cross-validation (sklearn.model_selection.KFold).
        - Report mean and std of your metric across folds.

    This is a required part of the project. The stub is here so the
    skeleton runs, but you must replace it with a real implementation.
    """
    raise NotImplementedError(
        "Implement a validation strategy — see project description."
    )
