import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime


def _days_since(date_str: str | None) -> int:
    if not date_str:
        return 0
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y"):
        try:
            dt = datetime.fromisoformat(date_str)
            return (datetime.now() - dt).days
        except Exception:
            continue
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return (datetime.now() - dt).days
    except Exception:
        return 0


def _split_categories(categories: str | None) -> List[str]:
    if not categories:
        return []
    return [c.strip() for c in str(categories).split(",") if c.strip()]


def _compose_text(df: pd.DataFrame) -> pd.Series:
    return (
        df.get("title", "").fillna("").astype(str)
        + " "
        + df.get("authors", "").fillna("").astype(str)
        + " "
        + df.get("categories", "").fillna("").astype(str)
        + " "
        + df.get("description", "").fillna("").astype(str)
    )


def _content_similarity_scores(read_books: pd.DataFrame, unread_books: pd.DataFrame) -> np.ndarray:
    if unread_books.empty:
        return np.array([])
    if read_books.empty:
        return np.full(len(unread_books), 0.5)

    read_text = _compose_text(read_books)
    unread_text = _compose_text(unread_books)
    try:
        tfidf = TfidfVectorizer().fit_transform(pd.concat([read_text, unread_text], ignore_index=True))
        if tfidf.shape[1] == 0:
            return np.full(len(unread_books), 0.5)
        read_vecs = tfidf[: len(read_text)]
        unread_vecs = tfidf[len(read_text) :]
        sim = cosine_similarity(unread_vecs, read_vecs).mean(axis=1)
        if sim.size == 0:
            return np.full(len(unread_books), 0.5)
        minv, maxv = sim.min(), sim.max()
        if maxv - minv <= 0:
            return np.full(len(unread_books), 0.5)
        return (sim - minv) / (maxv - minv)
    except Exception:
        return np.full(len(unread_books), 0.5)


def _mood_match(categories: str | None, description: str | None, mood: str) -> float:
    text = f"{categories or ''} {description or ''}".lower()
    if not text.strip():
        return 0.0
    mapping = {
        "勉強したい": ["技術", "科学", "歴史", "ビジネス", "自己啓発", "資格", "学習", "実用"],
        "リラックスしたい": ["小説", "漫画", "エッセイ", "旅行", "趣味", "日常"],
        "感動したい": ["文学", "恋愛", "青春", "家族", "ドラマ"],
        "ワクワクしたい": ["冒険", "ファンタジー", "sf", "アクション"],
        "ミステリーを読みたい": ["ミステリー", "推理", "サスペンス", "犯罪"],
    }
    keywords = mapping.get(mood, [])
    matches = sum(1 for k in keywords if k.lower() in text)
    if matches == 0:
        return 0.0
    if mood == "勉強したい":
        return min(1.0, 0.4 + 0.2 * matches)
    return min(1.0, 0.5 + 0.25 * matches)


def _favorite_genre(read_books: pd.DataFrame) -> str | None:
    genre_ratings: dict[str, list[float]] = {}
    for _, row in read_books.iterrows():
        rating = float(row.get("rating") or 0)
        for genre in _split_categories(row.get("categories")):
            genre_ratings.setdefault(genre, []).append(rating)
    if not genre_ratings:
        return None
    avg_ratings = {genre: sum(values) / len(values) for genre, values in genre_ratings.items()}
    return max(avg_ratings, key=avg_ratings.get)


def _policy_weights(policy: str) -> dict[str, float]:
    policies = {
        "好みに合う本": {"content": 0.40, "backlog": 0.15, "time": 0.15, "mood": 0.20, "genre": 0.10, "diversity": 0.0},
        "積読を減らす": {"content": 0.20, "backlog": 0.35, "time": 0.15, "mood": 0.10, "genre": 0.10, "diversity": 0.10},
        "短時間で読める本": {"content": 0.20, "backlog": 0.15, "time": 0.35, "mood": 0.15, "genre": 0.10, "diversity": 0.05},
        "普段と違う本": {"content": 0.20, "backlog": 0.15, "time": 0.15, "mood": 0.15, "genre": 0.10, "diversity": 0.25},
        "バランス重視": {"content": 0.30, "backlog": 0.20, "time": 0.20, "mood": 0.15, "genre": 0.10, "diversity": 0.05},
    }
    return policies.get(policy, policies["バランス重視"])


def recommend_books(books: pd.DataFrame, mood: str, reading_time: str, policy: str = "バランス重視", top_n: int = 3) -> pd.DataFrame:
    """Return top_n recommended unread books with scores and reasons."""
    if books is None or books.empty:
        return pd.DataFrame()

    candidates = books[books.get("status").isin(["未読", "読書中"])].copy()
    if candidates.empty:
        return pd.DataFrame()

    read = books[books.get("status") == "読了"]
    read_high = read[read.get("rating").fillna(0) >= 4] if not read.empty else pd.DataFrame()
    content_scores = _content_similarity_scores(read_high, candidates)
    favorite_genre = _favorite_genre(read) if not read.empty else None
    weights = _policy_weights(policy)

    time_map = {"15分": 15, "30分": 30, "1時間": 60, "2時間以上": 120}
    target_pages = time_map.get(reading_time, 60)

    recent_genres = set()
    if not read.empty:
        for categories in read.get("categories", "").dropna().astype(str):
            recent_genres.update(_split_categories(categories))

    results: List[Dict] = []
    for idx, row in candidates.reset_index(drop=True).iterrows():
        content_score = float(content_scores[idx]) * 100
        days = _days_since(row.get("registered_at"))
        if days >= 365:
            backlog_score = 100.0
        elif days >= 180:
            backlog_score = 80.0
        elif days >= 90:
            backlog_score = 60.0
        elif days >= 30:
            backlog_score = 40.0
        else:
            backlog_score = 10.0

        page = row.get("page_count") or 0
        if page == 0:
            time_fit = 50.0
        else:
            ratio = page / max(1, target_pages)
            if ratio <= 0.5:
                time_fit = 100.0
            elif ratio <= 1.0:
                time_fit = 80.0
            elif ratio <= 2.0:
                time_fit = 40.0
            else:
                time_fit = 10.0

        mood_fit = _mood_match(row.get("categories"), row.get("description"), mood) * 100.0

        genre_pref = 0.0
        book_genres = set(_split_categories(row.get("categories")))
        if favorite_genre and favorite_genre in book_genres:
            genre_pref = 100.0

        diversity = 0.0
        if book_genres and book_genres.isdisjoint(recent_genres):
            diversity = 30.0

        final = (
            content_score * weights["content"]
            + backlog_score * weights["backlog"]
            + time_fit * weights["time"]
            + mood_fit * weights["mood"]
            + genre_pref * weights["genre"]
            + diversity * weights["diversity"]
        )

        rating = row.get("rating") or 0
        final = min(max(final + (float(rating) * 4.0), 0), 100)

        reasons = []
        if backlog_score >= 80:
            reasons.append("長期間積読している")
        if content_score >= 60:
            reasons.append("内容が既存の蔵書に近い")
        if time_fit >= 80:
            reasons.append("今日の時間に合いやすい")
        if mood_fit >= 50:
            reasons.append("気分に合っている")
        if genre_pref >= 90:
            reasons.append("好みのジャンル")
        if diversity >= 20:
            reasons.append("普段と違うジャンル")

        results.append({
            **row.to_dict(),
            "score": float(final),
            "reasons": reasons,
            "content_score": float(content_score),
            "backlog_score": float(backlog_score),
            "time_fit": float(time_fit),
            "mood_fit": float(mood_fit),
            "genre_pref": float(genre_pref),
            "diversity": float(diversity),
            "backlog_days": float(days),
        })

    df = pd.DataFrame(results)
    if df.empty:
        return df
    df = df.sort_values("score", ascending=False).head(top_n)
    return df.reset_index(drop=True)


def create_reason_text(reasons: List[str]) -> str:
    if not reasons:
        return "特に目立った理由はありません。"
    return "。".join(reasons) + "。"
