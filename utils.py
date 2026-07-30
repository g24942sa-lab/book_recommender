import re
from typing import Optional
from datetime import datetime


def safe_str(x) -> str:
    if x is None:
        return ""
    return str(x)


def normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = safe_str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_isbn(isbn: Optional[str]) -> str:
    if not isbn:
        return ""
    s = re.sub(r"[^0-9Xx]", "", str(isbn))
    return s


def strip_html(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"<[^>]+>", "", s)


def days_since(date_str: Optional[str]) -> int:
    if not date_str:
        return 0
    try:
        dt = datetime.fromisoformat(date_str)
        return (datetime.now() - dt).days
    except Exception:
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return (datetime.now() - dt).days
        except Exception:
            return 0
