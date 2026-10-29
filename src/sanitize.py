"""Sanitize labeled data by excising metadata, artifacts, and restricted elements."""

import pandas as pd

from condense import condense


def sanitize(
    df: pd.DataFrame,
    subreddits: list[str] | None = None,
) -> pd.DataFrame:
    """
    Excise web metadata, aim ii artifacts, subreddit, and post text from labeled dissertation data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw labeled dataframe with metadata columns.
    subreddits : list[str] | None
        If provided, filter to only rows from these subreddits and retain the 'sbrt' column.
        If None (default), drop the 'sbrt' column entirely.

    Returns
    -------
    pd.DataFrame
        Sanitized dataframe with metadata, logits, and restricted elements removed.
    """
    # Apply condense transformations
    df = condense(df)

    # Drop 'text'
    df = df.drop(columns=['text'])

    # Filter by subreddits or drop the column
    if subreddits is not None:
        df = df[df['sbrt'].isin(subreddits)]
    else:
        df = df.drop(columns=['sbrt'])

    return df
