import hannoy
from hannoy import Reader


def test_exports() -> None:
    assert hannoy.__all__ == ["Metric", "Database", "Writer", "Reader"]


def test_read(db: hannoy.Database) -> None:
    reader: Reader = db.reader(0)
    query = [0.0, 1.0, 0.0]

    res = reader.by_vec(query, n=2)
    assert len(res) == 2

    (item_id, dist) = res[0]
    assert item_id == 1
    assert dist == 0.0


def test_read_by_item(db: hannoy.Database) -> None:
    reader: Reader = db.reader(0)

    res = reader.by_item(1, n=2)
    assert res is not None
    assert len(res) == 2

    assert {item_id for item_id, _ in res} == {0, 2}
    assert not any(d == 0 for _, d in res)


def test_multithreaded_reads(db) -> None:
    import threading

    def _read(db: hannoy.Database, query: list[float]):
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
