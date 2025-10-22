from sqlalchemy import text
from ace.models.base import Base
from ace.utils.database import create_db_engine


def test_database_isolation(tmp_path):
    db_path = tmp_path / "isolation.db"
    engine = create_db_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(bind=engine)
    with engine.connect() as conn:
        tables = {row[0] for row in conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))}
    assert "playbook_bullets" in tables
