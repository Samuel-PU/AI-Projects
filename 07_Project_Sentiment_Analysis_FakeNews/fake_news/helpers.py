# fake_news/helpers.py
"""Utility column-selectors kept in their own module so they pickle cleanly."""

def col_text(df):
    return df["text"]

def col_len(df):
    return df[["text_length"]]

def col_domain(df):
    return df[["domain"]]
