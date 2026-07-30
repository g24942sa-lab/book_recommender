import streamlit as st
from typing import Optional
import pandas as pd
from database import add_book, to_dataframe, delete_book, update_book, find_books
from google_books import search_by_title, search_by_isbn
from recommender import recommend_books
from utils import normalize_isbn, normalize_text
from ui import render_book_card, render_rating_stars, render_progress


st.set_page_config(
    page_title="AI積読本推薦システム",
    page_icon="📚",
    layout="wide",
)

st.title("📚 AI積読本推薦システム")
st.caption("積読本を管理し、『今読むべき一冊』を推薦します。")

# Load data
books = to_dataframe()

# Sidebar
st.sidebar.header("📖 今日の読書")
mood = st.sidebar.selectbox(
    "今日の気分",
    ["なんでも", "勉強したい", "リラックスしたい", "感動したい", "ワクワクしたい", "ミステリーを読みたい"],
)
reading_time = st.sidebar.selectbox("読める時間", ["15分", "30分", "1時間", "2時間以上"])
st.sidebar.divider()
st.sidebar.metric("登録冊数", books.shape[0] if books is not None else 0)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📚 本を登録", "📖 蔵書一覧", "🤖 おすすめ", "📊 分析"])

with tab1:
    st.header("📚 本を登録")
    col_search, col_manual = st.columns(2)
    with col_search:
        st.subheader("Google Booksから登録")
        q = st.text_input("タイトルまたはISBNで検索", key="search_q")
        if st.button("検索", key="search_button"):
            if not q:
                st.warning("検索語を入力してください。")
            else:
                # if isbn-looking string
                isbn = normalize_isbn(q)
                if isbn:
                    candidates = []
                    item = search_by_isbn(isbn)
                    if item:
                        candidates = [item]
                else:
                    candidates = search_by_title(q, max_results=5)
                if not candidates:
                    st.info("検索結果がありませんでした。手動登録をお試しください。")
                else:
                    for i, c in enumerate(candidates):
                        st.markdown(f"**候補 {i+1}**")
                        render_book_card(c, key_prefix=f"cand_{i}")
                        if st.button("この本を登録する", key=f"reg_{i}"):
                            # check duplicate by ISBN
                            if c.get("isbn"):
                                exists = find_books(query=c.get("isbn"))
                            else:
                                exists = find_books(query=c.get("title"))
                            if exists:
                                st.warning("既に登録済みの可能性があります。")
                            else:
                                book = {
                                    "title": c.get("title"),
                                    "authors": c.get("authors"),
                                    "categories": c.get("categories"),
                                    "description": c.get("description"),
                                    "publisher": c.get("publisher"),
                                    "published_date": c.get("published_date"),
                                    "isbn": normalize_isbn(c.get("isbn")),
                                    "page_count": c.get("page_count") or None,
                                    "thumbnail_url": c.get("thumbnail_url"),
                                    "status": "未読",
                                    "rating": 0,
                                    "started_at": None,
                                    "completed_at": None,
                                    "memo": None,
                                }
                                add_book(book)
                                st.success("登録しました。リロードしてください。")

    with col_manual:
        st.subheader("手動登録")
        m_title = st.text_input("タイトル (必須)", key="m_title")
        m_authors = st.text_input("著者", key="m_authors")
        m_categories = st.text_input("ジャンル", key="m_categories")
        m_pages = st.number_input("ページ数", min_value=0, key="m_pages")
        m_publisher = st.text_input("出版社", key="m_publisher")
        m_published = st.text_input("出版日", key="m_published")
        m_isbn = st.text_input("ISBN", key="m_isbn")
        m_description = st.text_area("説明", key="m_description")
        if st.button("手動で登録", key="manual_register"):
            if not m_title:
                st.warning("タイトルは必須です。")
            else:
                exists = find_books(query=normalize_isbn(m_isbn) or m_title)
                if exists:
                    st.warning("既に登録済みの可能性があります。")
                else:
                    book = {
                        "title": m_title,
                        "authors": m_authors,
                        "categories": m_categories,
                        "description": m_description,
                        "publisher": m_publisher,
                        "published_date": m_published,
                        "isbn": normalize_isbn(m_isbn),
                        "page_count": int(m_pages) if m_pages else None,
                        "thumbnail_url": None,
                        "status": "未読",
                        "rating": 0,
                        "started_at": None,
                        "completed_at": None,
                        "memo": None,
                    }
                    add_book(book)
                    st.success("手動登録しました。リロードしてください。")

with tab2:
    st.header("📖 蔵書一覧")
    df = to_dataframe()
    if df is None or df.empty:
        st.info("まだ本が登録されていません。")
    else:
        k = st.text_input("🔍 タイトル検索 (部分一致)", key="lib_search")
        status_filter = st.selectbox("状態", ["すべて", "未読", "読書中", "読了"], key="lib_status")
        view = df.copy()
        if k:
            view = view[view["title"].str.contains(k, na=False)]
        if status_filter != "すべて":
            view = view[view["status"] == status_filter]
        for _, row in view.iterrows():
            cols = st.columns([1, 4])
            with cols[0]:
                if row.get("thumbnail_url"):
                    st.image(row.get("thumbnail_url"), width=100)
                else:
                    st.image("https://via.placeholder.com/100x150.png?text=No+Cover", width=100)
            with cols[1]:
                st.subheader(row.get("title") or "(無題)")
                st.write(f"👤 {row.get('authors','')}  |  📖 {row.get('categories','')}  |  📄 {row.get('page_count','?')}ページ")
                st.write(f"⭐ {row.get('rating','-')}  |  状態：{row.get('status','未読')}")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("🗑 削除", key=f"del_{int(row.get('id',0))}"):
                        delete_book(int(row.get('id')))
                        st.experimental_rerun()
                with c2:
                    if st.button("✅ 読了にする", key=f"markread_{int(row.get('id',0))}"):
                        update_book(int(row.get('id')), {"status": "読了"})
                        st.experimental_rerun()

with tab3:
    st.header("🤖 AIおすすめ")
    df = to_dataframe()
    rec = recommend_books(df, mood, reading_time) if df is not None else pd.DataFrame()
    if rec is None or rec.empty:
        st.info("おすすめできる未読本がありません。")
    else:
        for i, r in rec.iterrows():
            st.subheader(f"おすすめ {i+1}: {r.get('title')}")
            render_book_card(r, key_prefix=f"rec_{i}")
            st.write("理由:", ", ".join(r.get('reasons', [])))
            render_progress(r.get('score', 0))
            if st.button("この本を今読む (読書中に変更)", key=f"start_{int(r.get('id',0))}"):
                update_book(int(r.get('id')), {"status": "読書中", "started_at": None})
                st.experimental_rerun()

with tab4:
    st.header("📊 分析")
    st.info("分析ダッシュボードは analytics モジュールで集計します。")

        for i, (_, row) in enumerate(top.iterrows(), 1):

            st.subheader(f"🥇 第{i}位")

            col1, col2 = st.columns([1, 3])

            with col1:

                if row["表紙"] != "":

                    st.image(
                        row["表紙"],
                        width=150
                    )

            with col2:

                st.markdown(f"### {row['タイトル']}")

                st.progress(
                    min(row["score"] / 100, 1.0)
                )

                st.write(
                    f"おすすめ度：**{min(row['score'],100):.0f}%**"
                )

                st.markdown("#### おすすめ理由")

                st.success(row["AIコメント"])

            st.divider()