"""Sanitize labeled data by excising metadata, artifacts, and restricted elements."""

import pandas as pd

from condense import condense


def sanitize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Excise web metadata, aim ii artifacts, subreddit, and post text from labeled dissertation data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw labeled dataframe with metadata columns.

    Returns
    -------
    pd.DataFrame
        Sanitized dataframe with metadata, logits, and restricted elements removed.
    """
    # Apply condense transformations
    df = condense(df)

    # Drop 'text' & 'sbrt'
    df = df.drop(columns=['text', 'sbrt'])

    return df
