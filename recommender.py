import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime


def _days_since(date_str: Optional[str]) -> int:
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


def content_based_score(books: pd.DataFrame) -> np.ndarray:
    """Compute content similarity scores (0-1) for each book using TF-IDF.
    Returns zeros if TF-IDF cannot build a vocabulary.
    """
    if len(books) <= 1:
        return np.zeros(len(books))

    text = (
        books.get("categories", "").fillna("") + " " +
        books.get("authors", "").fillna("") + " " +
        books.get("description", "").fillna("")
    )
    try:
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(text)
        if tfidf.shape[1] == 0:
            return np.zeros(len(books))
        sim = cosine_similarity(tfidf)
        scores = sim.mean(axis=1)
        # normalize to 0-1
        minv, maxv = scores.min(), scores.max()
        if maxv - minv <= 0:
            return np.zeros(len(books))
        return (scores - minv) / (maxv - minv)
    except Exception:
        return np.zeros(len(books))


def _mood_match(categories: str, mood: str) -> float:
    if not categories:
        return 0.0
    categories_lower = str(categories).lower()
    mapping = {
        "勉強したい": ["技術", "科学", "歴史", "ビジネス", "自己啓発", "資格", "学習", "実用"],
        "リラックスしたい": ["小説", "漫画", "エッセイ", "旅行", "趣味", "日常"],
        "感動したい": ["文学", "恋愛", "青春", "家族", "ドラマ"],
        "ワクワクしたい": ["冒険", "ファンタジー", "sf", "アクション"],
        "ミステリーを読みたい": ["ミステリー", "推理", "サスペンス", "犯罪"],
    }
    keywords = mapping.get(mood, [])
    matched = [k for k in keywords if k.lower() in categories_lower]
    if not matched:
        return 0.0
    if mood == "勉強したい":
        return 0.8 if len(matched) >= 2 else 0.5
    return 1.0


def recommend_books(books: pd.DataFrame, mood: str, reading_time: str, top_n: int = 3) -> pd.DataFrame:
    """Return top_n recommended unread books with scores and reasons."""
    if books is None or books.empty:
        return pd.DataFrame()

    unread = books[books.get("status") == "未読"].copy()
    if unread.empty:
        return pd.DataFrame()

    read = books[books.get("status") == "読了"]

    favorite_genre = None
    if not read.empty:
        # compute genre preference by average rating
        try:
            tmp = read.copy()
            tmp["categories"] = tmp["categories"].fillna("")
            grouped = tmp.groupby("categories")["rating"].mean()
            if not grouped.empty:
                favorite_genre = grouped.idxmax()
        except Exception:
            favorite_genre = None

    content_scores = content_based_score(unread)

    results: List[Dict] = []

    # reading_time pages estimate
    time_map = {"15分": 15, "30分": 30, "1時間": 60, "2時間以上": 120}
    target_pages = time_map.get(reading_time, 60)

    recent_genres = set(read.get("categories", "").dropna().unique()) if not read.empty else set()

    for idx, row in unread.reset_index(drop=True).iterrows():
        # content similarity (0-100)
        content_score = float(content_scores[idx]) * 100

        # backlog (registered_at -> days)
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

        # time fit based on page_count
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

        # mood fit
        mood_fit = _mood_match(str(row.get("categories") or ""), mood) * 100.0

        # genre preference
        genre_pref = 0.0
        if favorite_genre and favorite_genre in str(row.get("categories") or ""):
            genre_pref = 100.0

        # diversity: small boost if not in recent genres
        diversity = 0.0
        if row.get("categories") and row.get("categories") not in recent_genres:
            diversity = 30.0

        # combine weights per SPEC
        final = (
            content_score * 0.30
            + backlog_score * 0.20
            + time_fit * 0.20
            + mood_fit * 0.15
            + genre_pref * 0.10
            + diversity * 0.05
        )

        # rating contribution (0-5 -> 0-100 scaled small)
        rating = row.get("rating")
        if rating is None:
            rating = 0

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
