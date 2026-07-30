from typing import Dict, Any
import pandas as pd


def summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {
            "total": 0,
            "unread": 0,
            "reading": 0,
            "completed": 0,
            "completion_rate": 0.0,
            "average_rating": None,
            "average_pages": None,
        }
    total = len(df)
    unread = int((df.get("status") == "未読").sum())
    reading = int((df.get("status") == "読書中").sum())
    completed = int((df.get("status") == "読了").sum())
    completion_rate = (completed / total) * 100 if total > 0 else 0.0
    avg_rating = None
    if df.get("rating") is not None and not df["rating"].dropna().empty:
        avg_rating = float(df["rating"].dropna().astype(float).mean())
    avg_pages = None
    if df.get("page_count") is not None and not df["page_count"].dropna().empty:
        avg_pages = float(df["page_count"].dropna().astype(float).mean())
    return {
        "total": total,
        "unread": unread,
        "reading": reading,
        "completed": completed,
        "completion_rate": round(completion_rate, 1),
        "average_rating": avg_rating,
        "average_pages": avg_pages,
    }


def genre_counts(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=int)
    s = df.get("categories").fillna("")
    # split comma-separated categories into multiple rows
    exploded = s.str.split(",").explode().str.strip()
    return exploded[exploded != ""].value_counts()


def state_counts(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=int)
    return df.get("status").fillna("未登録").value_counts()


def rating_distribution(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty or df.get("rating") is None:
        return pd.Series(dtype=int)
    return df["rating"].fillna(0).astype(int).value_counts().sort_index()


def backlog_top(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    # compute days since registered
    def days_since(s):
        try:
            return (pd.Timestamp.now() - pd.to_datetime(s)).days
        except Exception:
            return 0
    df2 = df.copy()
    df2["backlog_days"] = df2["registered_at"].apply(days_since)
    return df2.sort_values("backlog_days", ascending=False).head(n)