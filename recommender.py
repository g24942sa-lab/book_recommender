from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


REQUIRED_COLUMNS = {
    "title": "",
    "authors": "",
    "categories": "",
    "description": "",
    "status": "",
    "rating": 0,
    "registered_at": "",
    "completed_at": "",
    "page_count": 0,
}


MOOD_KEYWORDS: dict[str, list[str]] = {
    "勉強したい": [
        "技術", "科学", "歴史", "ビジネス", "経済", "自己啓発",
        "資格", "学習", "実用", "プログラミング", "コンピュータ",
        "人工知能", "機械学習", "教育", "technology", "science",
        "business", "computer", "education",
    ],
    "リラックスしたい": [
        "小説", "漫画", "コミック", "エッセイ", "旅行", "趣味",
        "日常", "料理", "暮らし", "ユーモア", "癒し",
        "fiction", "comic", "travel", "humor",
    ],
    "感動したい": [
        "文学", "恋愛", "青春", "家族", "友情", "成長",
        "ドラマ", "ヒューマン", "人生", "感動",
        "romance", "family", "drama",
    ],
    "ワクワクしたい": [
        "冒険", "ファンタジー", "sf", "アクション", "異世界",
        "バトル", "魔法", "宇宙", "探索", "ヒーロー",
        "adventure", "fantasy", "science fiction", "action",
    ],
    "ミステリーを読みたい": [
        "ミステリー", "推理", "サスペンス", "犯罪", "探偵",
        "事件", "謎", "スリラー",
        "mystery", "crime", "detective", "thriller",
    ],
}


POLICY_WEIGHTS: dict[str, dict[str, float]] = {
    "好みに合う本": {
        "content": 0.45,
        "backlog": 0.10,
        "time": 0.10,
        "mood": 0.15,
        "genre": 0.20,
        "diversity": 0.00,
    },
    "積読を減らす": {
        "content": 0.15,
        "backlog": 0.45,
        "time": 0.15,
        "mood": 0.10,
        "genre": 0.10,
        "diversity": 0.05,
    },
    "短時間で読める本": {
        "content": 0.15,
        "backlog": 0.10,
        "time": 0.45,
        "mood": 0.15,
        "genre": 0.10,
        "diversity": 0.05,
    },
    "普段と違う本": {
        "content": 0.10,
        "backlog": 0.10,
        "time": 0.10,
        "mood": 0.15,
        "genre": 0.05,
        "diversity": 0.50,
    },
    "バランス重視": {
        "content": 0.30,
        "backlog": 0.20,
        "time": 0.15,
        "mood": 0.15,
        "genre": 0.15,
        "diversity": 0.05,
    },
}


def _prepare_dataframe(books: pd.DataFrame) -> pd.DataFrame:
    """推薦処理に必要な列を補い、安全な形式へ変換する。"""
    df = books.copy()

    for column, default in REQUIRED_COLUMNS.items():
        if column not in df.columns:
            df[column] = default

    text_columns = [
        "title",
        "authors",
        "categories",
        "description",
        "status",
        "registered_at",
        "completed_at",
    ]

    for column in text_columns:
        df[column] = df[column].fillna("").astype(str)

    df["rating"] = pd.to_numeric(
        df["rating"], errors="coerce"
    ).fillna(0).clip(0, 5)

    df["page_count"] = pd.to_numeric(
        df["page_count"], errors="coerce"
    ).fillna(0).clip(lower=0)

    return df


def _parse_date(value: Any) -> datetime | None:
    """複数の日付形式を安全に解析する。"""
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    formats = (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y-%m",
        "%Y/%m",
        "%Y",
    )

    for date_format in formats:
        try:
            return datetime.strptime(text, date_format)
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _days_since(value: Any) -> int:
    date = _parse_date(value)

    if date is None:
        return 0

    return max(0, (datetime.now() - date).days)


def _normalize_genre(genre: str) -> str:
    """ジャンル比較用に表記を簡単に正規化する。"""
    return (
        genre.strip()
        .lower()
        .replace("　", "")
        .replace(" ", "")
        .replace("・", "")
        .replace("/", "")
    )


def _split_categories(categories: Any) -> List[str]:
    if categories is None or pd.isna(categories):
        return []

    text = str(categories)

    for separator in ["、", "／", "/", ";", "・", "|"]:
        text = text.replace(separator, ",")

    return [
        genre.strip()
        for genre in text.split(",")
        if genre.strip()
    ]


def _compose_text(df: pd.DataFrame) -> pd.Series:
    """
    タイトルとジャンルを重ねて配置し、推薦への影響を少し強くする。
    説明文だけが長い場合に、重要情報が埋もれることを防ぐ。
    """
    title = df["title"].fillna("").astype(str)
    authors = df["authors"].fillna("").astype(str)
    categories = df["categories"].fillna("").astype(str)
    description = df["description"].fillna("").astype(str)

    return (
        title
        + " "
        + title
        + " "
        + categories
        + " "
        + categories
        + " "
        + authors
        + " "
        + description
    ).str.lower()


def _content_similarity_scores(
    preferred_books: pd.DataFrame,
    candidates: pd.DataFrame,
) -> np.ndarray:
    """
    高評価本から評価値付きのユーザープロファイルを作り、
    各候補との類似度を求める。

    日本語の分かち書きがなくても比較しやすいように、
    文字n-gramを使用する。
    """
    if candidates.empty:
        return np.array([], dtype=float)

    if preferred_books.empty:
        return np.full(len(candidates), 0.5, dtype=float)

    preferred_text = _compose_text(preferred_books)
    candidate_text = _compose_text(candidates)

    all_text = pd.concat(
        [preferred_text, candidate_text],
        ignore_index=True,
    )

    if all_text.str.strip().eq("").all():
        return np.full(len(candidates), 0.5, dtype=float)

    try:
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=1,
            sublinear_tf=True,
            max_features=8000,
        )

        matrix = vectorizer.fit_transform(all_text)

        preferred_vectors = matrix[: len(preferred_books)]
        candidate_vectors = matrix[len(preferred_books):]

        ratings = preferred_books["rating"].to_numpy(dtype=float)

        # 評価4を1、評価5を2として、評価5の影響を強くする
        rating_weights = np.clip(ratings - 3.0, 0.5, 2.0)
        rating_weights = rating_weights / rating_weights.sum()

        # 高評価本のベクトルを加重平均し、ユーザーの好みを表現する
        user_profile = preferred_vectors.multiply(
            rating_weights[:, np.newaxis]
        ).sum(axis=0)

        similarities = cosine_similarity(
            candidate_vectors,
            user_profile,
        ).ravel()

        # 候補内でmin-max正規化せず、絶対的な類似度を使用する
        return np.clip(similarities, 0.0, 1.0)

    except (ValueError, TypeError):
        return np.full(len(candidates), 0.5, dtype=float)


def _backlog_score(registered_at: Any) -> float:
    """積読日数を滑らかに0〜100点へ変換する。"""
    days = _days_since(registered_at)

    if days <= 0:
        return 20.0

    # 365日前後で十分高得点になり、急な段差を作らない
    score = 20.0 + 80.0 * (
        np.log1p(days) / np.log1p(365)
    )

    return float(np.clip(score, 20.0, 100.0))


def _time_fit_score(
    page_count: Any,
    reading_time: str,
    policy: str,
) -> float:
    """
    読書時間とページ数の相性を評価する。

    1回で読み切れるかではなく、「その時間で読み始めやすいか」を
    基準にする。長編本を極端に不利にしない。
    """
    page = float(page_count or 0)

    if page <= 0:
        return 50.0

    target_pages = {
        "15分": 15,
        "30分": 30,
        "1時間": 60,
        "2時間以上": 120,
    }.get(reading_time, 60)

    # 1回の読書で全体の5〜30％程度進められる本を読み始めやすいとする
    progress_ratio = target_pages / page

    if progress_ratio >= 0.30:
        score = 100.0
    elif progress_ratio >= 0.15:
        score = 85.0
    elif progress_ratio >= 0.08:
        score = 70.0
    elif progress_ratio >= 0.04:
        score = 55.0
    else:
        score = 40.0

    # 「短時間で読める本」の場合のみ短い本を追加で優先する
    if policy == "短時間で読める本":
        if page <= target_pages * 2:
            score += 15
        elif page <= target_pages * 4:
            score += 5

    return float(np.clip(score, 0.0, 100.0))


def _mood_match(
    categories: Any,
    description: Any,
    mood: str,
) -> float:
    """気分とジャンル・説明文の一致度を0〜100点で返す。"""
    if mood in ("", "なんでも", None):
        return 60.0

    text = f"{categories or ''} {description or ''}".lower()

    if not text.strip():
        return 30.0

    keywords = MOOD_KEYWORDS.get(mood, [])

    if not keywords:
        return 50.0

    matches = sum(
        1
        for keyword in keywords
        if keyword.lower() in text
    )

    if matches == 0:
        return 15.0
    if matches == 1:
        return 65.0
    if matches == 2:
        return 85.0

    return 100.0


def _genre_preference_scores(
    read_books: pd.DataFrame,
) -> dict[str, float]:
    """
    読了冊数と評価を考慮して、各ジャンルの好みを0〜100点で算出する。
    1冊だけ評価5のジャンルが過度に優先されないよう冊数も考慮する。
    """
    genre_values: dict[str, list[float]] = {}

    for _, row in read_books.iterrows():
        rating = float(row.get("rating", 0))

        if rating <= 0:
            continue

        for genre in _split_categories(row.get("categories")):
            normalized = _normalize_genre(genre)

            if normalized:
                genre_values.setdefault(normalized, []).append(rating)

    if not genre_values:
        return {}

    scores: dict[str, float] = {}

    for genre, ratings in genre_values.items():
        average_rating = float(np.mean(ratings))
        count_bonus = min(len(ratings), 5) / 5

        rating_score = average_rating / 5.0
        combined = rating_score * 0.8 + count_bonus * 0.2

        scores[genre] = combined * 100.0

    return scores


def _book_genre_preference(
    categories: Any,
    genre_scores: dict[str, float],
) -> float:
    if not genre_scores:
        return 50.0

    book_genres = {
        _normalize_genre(genre)
        for genre in _split_categories(categories)
    }

    matching_scores = [
        score
        for genre, score in genre_scores.items()
        if any(
            genre in book_genre or book_genre in genre
            for book_genre in book_genres
        )
    ]

    if not matching_scores:
        return 20.0

    return float(max(matching_scores))


def _recent_read_genres(
    read_books: pd.DataFrame,
    limit: int = 5,
) -> set[str]:
    """直近に読了した本だけからジャンルを取得する。"""
    if read_books.empty:
        return set()

    recent = read_books.copy()
    recent["_completed_date"] = recent["completed_at"].apply(_parse_date)

    recent = recent.sort_values(
        "_completed_date",
        ascending=False,
        na_position="last",
    ).head(limit)

    genres: set[str] = set()

    for categories in recent["categories"]:
        genres.update(
            _normalize_genre(genre)
            for genre in _split_categories(categories)
        )

    return {genre for genre in genres if genre}


def _diversity_score(
    categories: Any,
    recent_genres: set[str],
) -> float:
    book_genres = {
        _normalize_genre(genre)
        for genre in _split_categories(categories)
    }

    if not book_genres:
        return 40.0

    if not recent_genres:
        return 60.0

    overlaps = book_genres & recent_genres

    if not overlaps:
        return 100.0

    overlap_ratio = len(overlaps) / len(book_genres)

    return float(np.clip(100.0 * (1.0 - overlap_ratio), 10.0, 100.0))


def _policy_weights(policy: str) -> dict[str, float]:
    return POLICY_WEIGHTS.get(
        policy,
        POLICY_WEIGHTS["バランス重視"],
    )


def recommend_books(
    books: pd.DataFrame,
    mood: str,
    reading_time: str,
    policy: str = "バランス重視",
    top_n: int = 3,
) -> pd.DataFrame:
    """未読本から推薦順位、スコア内訳、推薦理由を生成する。"""
    if books is None or books.empty:
        return pd.DataFrame()

    df = _prepare_dataframe(books)

    # 原則として未読本だけを推薦対象にする
    candidates = df[df["status"] == "未読"].copy()

    if candidates.empty:
        return pd.DataFrame()

    read_books = df[df["status"] == "読了"].copy()

    # 評価4以上を優先し、ない場合は評価済みの読了本を使う
    preferred_books = read_books[read_books["rating"] >= 4].copy()

    if preferred_books.empty:
        preferred_books = read_books[read_books["rating"] > 0].copy()

    content_scores = _content_similarity_scores(
        preferred_books,
        candidates,
    )

    genre_scores = _genre_preference_scores(read_books)
    recent_genres = _recent_read_genres(read_books)
    weights = _policy_weights(policy)

    results: List[Dict[str, Any]] = []

    for position, (_, row) in enumerate(
        candidates.reset_index(drop=True).iterrows()
    ):
        content_score = float(content_scores[position] * 100.0)

        backlog_score = _backlog_score(
            row.get("registered_at")
        )

        time_fit = _time_fit_score(
            row.get("page_count"),
            reading_time,
            policy,
        )

        mood_fit = _mood_match(
            row.get("categories"),
            row.get("description"),
            mood,
        )

        genre_pref = _book_genre_preference(
            row.get("categories"),
            genre_scores,
        )

        diversity = _diversity_score(
            row.get("categories"),
            recent_genres,
        )

        final_score = (
            content_score * weights["content"]
            + backlog_score * weights["backlog"]
            + time_fit * weights["time"]
            + mood_fit * weights["mood"]
            + genre_pref * weights["genre"]
            + diversity * weights["diversity"]
        )

        final_score = float(
            np.clip(final_score, 0.0, 100.0)
        )

        reasons: list[str] = []

        if content_score >= 55:
            reasons.append(
                "高く評価した本と内容が似ています"
            )

        if backlog_score >= 75:
            reasons.append(
                "長期間積読になっている本です"
            )

        if time_fit >= 80:
            reasons.append(
                f"{reading_time}の読書時間でも取り組みやすい本です"
            )

        if mood not in ("", "なんでも") and mood_fit >= 65:
            reasons.append(
                f"「{mood}」という今の気分に合っています"
            )

        if genre_pref >= 70:
            reasons.append(
                "過去に高く評価したジャンルです"
            )

        if (
            policy == "普段と違う本"
            and diversity >= 80
        ):
            reasons.append(
                "最近読んでいないジャンルを楽しめます"
            )

        if not reasons:
            highest_component = max(
                {
                    "内容との相性が比較的良い本です": content_score,
                    "積読を減らすきっかけになる本です": backlog_score,
                    "今の読書時間に合わせやすい本です": time_fit,
                    "現在の気分と比較的相性が良い本です": mood_fit,
                },
                key=lambda reason: {
                    "内容との相性が比較的良い本です": content_score,
                    "積読を減らすきっかけになる本です": backlog_score,
                    "今の読書時間に合わせやすい本です": time_fit,
                    "現在の気分と比較的相性が良い本です": mood_fit,
                }[reason],
            )

            reasons.append(highest_component)

        results.append(
            {
                **row.to_dict(),
                "score": round(final_score, 1),
                "reasons": reasons,
                "content_score": round(content_score, 1),
                "backlog_score": round(backlog_score, 1),
                "time_fit": round(time_fit, 1),
                "mood_fit": round(mood_fit, 1),
                "genre_pref": round(genre_pref, 1),
                "diversity": round(diversity, 1),
                "backlog_days": _days_since(
                    row.get("registered_at")
                ),
            }
        )

    result_df = pd.DataFrame(results)

    if result_df.empty:
        return result_df

    # 同点時は積読日数と内容類似度で順位を決定
    result_df = result_df.sort_values(
        by=[
            "score",
            "backlog_score",
            "content_score",
        ],
        ascending=[False, False, False],
    )

    return result_df.head(top_n).reset_index(drop=True)


def create_reason_text(reasons: List[str]) -> str:
    if not reasons:
        return "複数の条件を総合的に判断して選ばれました。"

    return "。".join(
        reason.rstrip("。")
        for reason in reasons
    ) + "。"
