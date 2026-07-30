import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database import to_dataframe
from seed_dummy_books import seed_dummy_books


def test_seed_dummy_books_creates_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "test_books.db"
    inserted_first = seed_dummy_books(db_path=str(db_path), count=3)
    inserted_second = seed_dummy_books(db_path=str(db_path), count=3)

    assert inserted_first == 3
    assert inserted_second == 3
    df = to_dataframe(db_path=str(db_path))
    assert len(df) == 3
    assert set(df["status"].tolist()) <= {"未読", "読書中", "読了"}
