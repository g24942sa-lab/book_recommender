import sqlite3
from typing import List, Optional, Dict, Any
import os
import pandas as pd
from contextlib import contextmanager

DB_FILE = "books.db"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS books (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    authors TEXT,
    categories TEXT,
    description TEXT,
    publisher TEXT,
    published_date TEXT,
    isbn TEXT,
    page_count INTEGER,
    thumbnail_url TEXT,
    status TEXT CHECK(status IN ('未読','読書中','読了')) DEFAULT '未読',
    rating INTEGER CHECK(rating BETWEEN 0 AND 5),
    registered_at TEXT DEFAULT (datetime('now')),
    started_at TEXT,
    completed_at TEXT,
    memo TEXT
);
"""


@contextmanager
def get_connection(db_path: str = DB_FILE):
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    try:
        yield conn
    finally:
        conn.close()


def initialize_database(db_path: str = DB_FILE) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True) if os.path.dirname(db_path) else None
    with get_connection(db_path) as conn:
        cur = conn.cursor()
        cur.execute(CREATE_TABLE_SQL)
        conn.commit()


def row_to_dict(row: sqlite3.Row, columns: List[str]) -> Dict[str, Any]:
    return {col: row[idx] for idx, col in enumerate(columns)}


def add_book(book: Dict[str, Any], db_path: str = DB_FILE) -> int:
    """Insert a book record. Returns inserted row id."""
    initialize_database(db_path)
    cols = [
        'title','authors','categories','description','publisher','published_date',
        'isbn','page_count','thumbnail_url','status','rating','started_at','completed_at','memo'
    ]
    placeholders = ','.join('?' for _ in cols)
    values = [book.get(c) for c in cols]
    with get_connection(db_path) as conn:
        cur = conn.cursor()
        cur.execute(f"INSERT INTO books ({','.join(cols)}) VALUES ({placeholders})", values)
        conn.commit()
        return cur.lastrowid


def get_book(book_id: int, db_path: str = DB_FILE) -> Optional[Dict[str, Any]]:
    initialize_database(db_path)
    with get_connection(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM books WHERE id = ?", (book_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return dict(row)


def find_books(query: Optional[str] = None, status: Optional[str] = None, db_path: str = DB_FILE) -> List[Dict[str, Any]]:
    initialize_database(db_path)
    with get_connection(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        sql = "SELECT * FROM books"
        params: List[Any] = []
        clauses: List[str] = []
        if query:
            clauses.append("(title LIKE ? OR authors LIKE ? OR categories LIKE ? OR isbn LIKE ?)")
            q = f"%{query}%"
            params.extend([q, q, q, q])
        if status:
            clauses.append("status = ?")
            params.append(status)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        cur.execute(sql, params)
        rows = cur.fetchall()
        return [dict(r) for r in rows]


def update_book(book_id: int, updates: Dict[str, Any], db_path: str = DB_FILE) -> bool:
    if not updates:
        return False
    initialize_database(db_path)
    cols = [f"{k} = ?" for k in updates.keys()]
    values = list(updates.values())
    values.append(book_id)
    sql = f"UPDATE books SET {', '.join(cols)} WHERE id = ?"
    with get_connection(db_path) as conn:
        cur = conn.cursor()
        cur.execute(sql, values)
        conn.commit()
        return cur.rowcount > 0


def delete_book(book_id: int, db_path: str = DB_FILE) -> bool:
    initialize_database(db_path)
    with get_connection(db_path) as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM books WHERE id = ?", (book_id,))
        conn.commit()
        return cur.rowcount > 0


def to_dataframe(db_path: str = DB_FILE) -> pd.DataFrame:
    initialize_database(db_path)
    with get_connection(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM books", conn)
        return df
