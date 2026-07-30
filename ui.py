from typing import Dict, Any, Optional


def render_book_card(book: Dict[str, Any], key_prefix: str = "book") -> None:
    import streamlit as st
    cols = st.columns([1, 3])
    thumb = book.get("thumbnail_url") or ""
    with cols[0]:
        if thumb:
            try:
                st.image(thumb, width=160)
            except Exception:
                st.empty()
        else:
            st.image(
                "https://via.placeholder.com/128x180.png?text=No+Cover",
                width=160
            )
    with cols[1]:
        st.markdown(f"**{book.get('title','(無題)')}**")
        authors = book.get("authors") or ""
        if authors:
            st.markdown(f"- {authors}")
        st.markdown(f"- ジャンル: {book.get('categories','')} | ページ数: {book.get('page_count','?')}")
        st.markdown(f"- 状態: {book.get('status','未読')} | 評価: {book.get('rating', '')}")
        memo = book.get("memo")
        if memo:
            st.markdown(f"- メモ: {memo}")


def render_rating_stars(rating: Optional[float]) -> None:
    import streamlit as st
    if rating is None:
        st.write("評価: -")
        return
    try:
        r = int(round(float(rating)))
    except Exception:
        r = 0
    stars = "★" * r + "☆" * (5 - r)
    st.write(f"評価: {stars} ({r}/5)")


def render_progress(score: float) -> None:
    import streamlit as st
    pct = max(0, min(score, 100)) / 100.0
    st.progress(pct)
