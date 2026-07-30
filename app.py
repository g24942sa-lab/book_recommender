import streamlit as st
from typing import Optional
import pandas as pd
import plotly.express as px
from database import add_book, to_dataframe, delete_book, update_book, find_books, book_exists
from google_books import search_by_title, search_by_isbn
from recommender import recommend_books
from seed_dummy_books import seed_dummy_books
from utils import normalize_isbn, normalize_text
from ui import render_book_card, render_rating_stars, render_progress
from analytics import summary_stats, genre_counts, state_counts, rating_distribution, backlog_top


st.set_page_config(
    page_title="積読本推薦システム",
    page_icon="📚",
    layout="wide",
)

st.title("📚 積読本推薦システム")
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
policy = st.sidebar.selectbox(
    "推薦方針",
    ["好みに合う本", "積読を減らす", "短時間で読める本", "普段と違う本", "バランス重視"],
)
st.sidebar.divider()
st.sidebar.metric("登録冊数", books.shape[0] if books is not None else 0)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📚 本を登録", "📖 蔵書一覧", "🤖 おすすめ", "📊 分析"])

with tab1:
    st.header("📚 本を登録")
    if st.button("🔧 ダミー本を表示する", key="seed_dummy"):
        inserted = seed_dummy_books(count=20, reset=True)
        st.success(f"{inserted}件のダミー本を登録しました。ページを再読み込みしています。")
        st.rerun()
    col_search, col_manual = st.columns(2)
    with col_search:
        st.subheader("Google Booksから登録")
        q = st.text_input("タイトルまたはISBNで検索", key="search_q")
        reg_status = st.selectbox("登録時の状態", ["未読", "読書中", "読了"], key="reg_status")
        reg_rating = st.slider("評価", 0, 5, 0, key="reg_rating")
        reg_memo = st.text_area("メモ", key="reg_memo", height=120)
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
                            isbn_text = normalize_isbn(c.get("isbn"))
                            if isbn_text:
                                exists = book_exists(isbn=isbn_text)
                            else:
                                exists = book_exists(title=c.get("title"), authors=c.get("authors"))
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
                                    "isbn": isbn_text,
                                    "page_count": c.get("page_count") or None,
                                    "thumbnail_url": c.get("thumbnail_url"),
                                    "status": reg_status,
                                    "rating": int(reg_rating),
                                    "started_at": None,
                                    "completed_at": None,
                                    "memo": reg_memo or None,
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
        m_status = st.selectbox("状態", ["未読", "読書中", "読了"], key="m_status")
        m_rating = st.slider("評価", 0, 5, 0, key="m_rating")
        m_memo = st.text_area("メモ", key="m_memo", height=120)
        if st.button("手動で登録", key="manual_register"):
            if not m_title:
                st.warning("タイトルは必須です。")
            else:
                isbn_text = normalize_isbn(m_isbn)
                if isbn_text:
                    exists = book_exists(isbn=isbn_text)
                else:
                    exists = book_exists(title=m_title, authors=m_authors)
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
                        "isbn": isbn_text,
                        "page_count": int(m_pages) if m_pages else None,
                        "thumbnail_url": None,
                        "status": m_status,
                        "rating": int(m_rating),
                        "started_at": None,
                        "completed_at": None,
                        "memo": m_memo or None,
                    }
                    add_book(book)
                    st.success("手動登録しました。リロードしてください。")

with tab2:
    st.header("📖 蔵書一覧")
    df = to_dataframe()
    if df is None or df.empty:
        st.info("まだ本が登録されていません。")
    else:
        k = st.text_input("🔍 検索 (タイトル/著者/ジャンル)", key="lib_search")
        status_filter = st.selectbox("状態", ["すべて", "未読", "読書中", "読了"], key="lib_status")
        rating_filter = st.selectbox("評価", ["すべて", "0", "1", "2", "3", "4", "5"], key="lib_rating")
        sort_option = st.selectbox(
            "並び替え",
            ["登録日順", "積読日数順", "ページ数順", "評価順"],
            key="lib_sort"
        )
        view = df.copy()
        if k:
            mask = (
                view["title"].str.contains(k, na=False)
                | view["authors"].str.contains(k, na=False)
                | view["categories"].str.contains(k, na=False)
            )
            view = view[mask]
        if status_filter != "すべて":
            view = view[view["status"] == status_filter]
        if rating_filter != "すべて":
            view = view[view["rating"].fillna(-1).astype(int) == int(rating_filter)]
        if sort_option == "登録日順":
            view = view.sort_values("registered_at", ascending=False, na_position="last")
        elif sort_option == "積読日数順":
            view = view.assign(
                backlog_days=view["registered_at"].apply(
                    lambda x: (pd.Timestamp.now() - pd.to_datetime(x, errors="coerce")).days
                    if pd.notna(x)
                    else -1
                )
            ).sort_values("backlog_days", ascending=False, na_position="last")
        elif sort_option == "ページ数順":
            view = view.sort_values("page_count", ascending=True, na_position="last")
        elif sort_option == "評価順":
            view = view.sort_values("rating", ascending=False, na_position="last")
        for _, row in view.iterrows():
            cols = st.columns([1, 4])
            with cols[0]:
                thumbnail = row.get("thumbnail_url")
                if isinstance(thumbnail, str) and thumbnail.strip():
                    try:
                        st.image(thumbnail, width=100)
                    except Exception:
                        st.image("https://via.placeholder.com/100x150.png?text=No+Cover", width=100)
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
                        st.rerun()
                with c2:
                    if st.button("✅ 読了にする", key=f"markread_{int(row.get('id',0))}"):
                        update_book(int(row.get('id')), {"status": "読了"})
                        st.rerun()

with tab3:
    st.header("🤖 おすすめ")
    df = to_dataframe()
    if df is None or df.empty:
        st.info("おすすめできる本がありません。")
    else:
        rec = recommend_books(df, mood, reading_time, policy=policy) if df is not None else pd.DataFrame()
        if rec is None or rec.empty:
            st.info("おすすめできる本がありません。")
        else:
            for i, r in rec.iterrows():
                st.subheader(f"おすすめ {i+1}: {r.get('title')}")
                render_book_card(r, key_prefix=f"rec_{i}")
                st.write("理由:", ", ".join(r.get('reasons', [])))
                render_progress(r.get('score', 0))
                if st.button("この本を今読む (読書中に変更)", key=f"start_{int(r.get('id',0))}"):
                    update_book(int(r.get('id')), {"status": "読書中", "started_at": None})
                    st.rerun()

with tab4:
    st.header("📊 分析")
    df = to_dataframe()
    if df is None or df.empty:
        st.info("まだデータがありません。まず本を登録してください。")
    else:
        stats = summary_stats(df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("総登録冊数", stats["total"])
        c2.metric("未読冊数", stats["unread"])
        c3.metric("読書中冊数", stats["reading"])
        c4.metric("読了冊数", stats["completed"])

        st.metric("読了率", f"{stats['completion_rate']:.1f}%")
        st.metric("平均評価", "-" if stats["average_rating"] is None else f"{stats['average_rating']:.1f}/5")
        st.metric("平均ページ数", "-" if stats["average_pages"] is None else f"{stats['average_pages']:.0f}ページ")

        st.subheader("状態別冊数")
        state_df = state_counts(df).reset_index()
        state_df.columns = ["状態", "冊数"]
        if not state_df.empty:
            st.plotly_chart(px.pie(state_df, names="状態", values="冊数", hole=0.4), use_container_width=True)
        else:
            st.info("状態データがありません。")

        st.subheader("ジャンル別冊数")
        genre_df = genre_counts(df).reset_index()
        genre_df.columns = ["ジャンル", "冊数"]
        if not genre_df.empty:
            st.plotly_chart(px.bar(genre_df, x="ジャンル", y="冊数", color="ジャンル"), use_container_width=True)
        else:
            st.info("ジャンルデータがありません。")

        st.subheader("評価分布")
        rating_df = rating_distribution(df).reset_index()
        rating_df.columns = ["評価", "冊数"]
        if not rating_df.empty:
            st.plotly_chart(px.bar(rating_df, x="評価", y="冊数"), use_container_width=True)
        else:
            st.info("評価データがありません。")

        st.subheader("積読日数ランキング")
        backlog_df = backlog_top(df, n=10)
        if backlog_df.empty:
            st.info("積読データがありません。")
        else:
            st.dataframe(backlog_df[["title", "backlog_days", "status"]].rename(columns={"title": "タイトル", "backlog_days": "積読日数", "status": "状態"}), use_container_width=True)