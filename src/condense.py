"""Condense labeled data by excising metadata and artifacts."""

import ast

import pandas as pd


def condense(df: pd.DataFrame) -> pd.DataFrame:
    """
    Excise web metadata and aim ii artifacts from labeled dissertation data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw labeled dataframe with metadata columns.

    Returns
    -------
    pd.DataFrame
        Condensed dataframe with metadata and unbound raw logits removed.
    """
    # Excise web metadata & aim ii artifacts
    df = df.drop(columns=[
        'date',      # datetime in YYY-MM-DD hh:mm:ss
        'p_au',      # post author
        'p_utc',     # datetime in Coordinated Universal Time
        'p_date',    # datetime (dupe)
        'id',        # Reddit post id
        'n_cmnt',    # number of top-level comments
        'not_us',    # 0/1 for mentions of non-U.S. geographies
        'cpnd_pred', # computed indicator for comorbid asp + dep + val
        'Segment',   # concatenation artifact
        'tech',      # LIWC-encoded placebo outcome
        'tech_high', # placebo outcome: quantile split
    ])

    # Ditch deprecated `p_` prefix
    df = df.rename(columns={'p_sbrt': 'sbrt'})

    # Excise RoBERTa-output "raw" (unbound, unnormalized) logits
    df = df.drop(columns=[
        'asp_logit', 'dep_logit', 'val_logit', 'prg_logit',
        'tgd_logit', 'age_logit', 'race_logit', 'dbty_logit',
    ])

    # Unnest normalized `pos(1)` probabilities
    prob_cols = [col for col in df.columns if col.endswith('_prob')]
    for col in prob_cols:
        df[col] = df[col].apply(lambda x: round(ast.literal_eval(x)[1], 2))

    return df
