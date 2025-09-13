from pathlib import Path
from typing import List
import pytest
import hannoy
from hannoy import Metric


@pytest.fixture(scope="function", autouse=False)
def db(tmp_path: Path):
    db = hannoy.Database(tmp_path, Metric.HAMMING)

    with db.writer(3, m=4, ef=10) as writer:
        writer.add_item(0, [1.0, 0.0, 0.0])
        writer.add_item(1, [0.0, 1.0, 0.0])
        writer.add_item(2, [0.0, 0.0, 1.0])

    yield db


def test_exports() -> None:
    assert hannoy.__all__ == ["Metric", "Database", "Writer"]


def test_read(db: hannoy.Database) -> None:
    reader = db.reader(0)
    query = [0.0, 1.0, 0.0]

    res = reader.by_vec(query, n=2)
    print(res)
    assert len(res) == 2

    (item_id, dist) = res[0]
    assert item_id == 1
    assert dist == 0.0


def test_multithreaded_reads(db) -> None:
    import threading

    def _read(db: hannoy.Database, query: List[float]):
        reader = db.reader(0)
        t_id = threading.get_ident()
        print(f"nns from thread {t_id}: {reader.by_vec(query, 1)}")

    threads = []
    for q in [[1.0, 0.0, 0.0,], [0.0, 1.0, 0.0]]:
        t = threading.Thread(target=_read, args=(db, q))
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join()
