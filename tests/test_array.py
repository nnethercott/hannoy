import re
from pathlib import Path

import numpy as np
import pytest

import hannoy
from hannoy import Metric, Reader


@pytest.fixture(scope="function")
def array_db(tmp_path: Path) -> hannoy.Database:
    db = hannoy.Database(tmp_path, Metric.EUCLIDEAN)
    vectors = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    with db.writer(3, m=4, ef=10) as writer:
        writer.add_items([0, 1, 2], vectors)
    return db


def test_add_items_from_numpy_array(array_db: hannoy.Database) -> None:
    reader: Reader = array_db.reader(0)
    assert reader.by_vec([1.0, 0.0, 0.0], n=1) == [(0, 0.0)]
    assert reader.by_vec([0.0, 1.0, 0.0], n=1) == [(1, 0.0)]
    assert reader.by_vec([0.0, 0.0, 1.0], n=1) == [(2, 0.0)]


@pytest.mark.parametrize(
    ("vectors", "exc"),
    [
        pytest.param(
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            ValueError(
                "add_items requires as many items as the array has rows: got 1 items for 2 rows"
            ),
            id="row-count-mismatch",
        ),
        pytest.param(
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            ValueError("add_items requires a 2D array, got 1D"),
            id="requires-2d",
        ),
    ],
)
def test_add_items_rejects_invalid_input(
    tmp_path: Path, vectors: np.ndarray, exc: Exception
) -> None:
    db = hannoy.Database(tmp_path, Metric.EUCLIDEAN)
    with pytest.raises(type(exc), match=re.escape(str(exc))):
        with db.writer(3, m=4, ef=10) as writer:
            writer.add_items([0], vectors)


def test_by_array_1d_matches_by_vec(array_db: hannoy.Database) -> None:
    reader: Reader = array_db.reader(0)
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    res = reader.by_array(query, n=2, ef_search=10)
    assert res == reader.by_vec([1.0, 0.0, 0.0], n=2, ef_search=10)


def test_by_array_2d_batches_queries(array_db: hannoy.Database) -> None:
    reader: Reader = array_db.reader(0)
    queries = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    res = reader.by_array(queries, n=1, ef_search=10)
    assert res == [[(0, 0.0)], [(1, 0.0)]]


def test_by_array_out_1d(array_db: hannoy.Database) -> None:
    reader: Reader = array_db.reader(0)
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    expected = reader.by_array(query, n=2, ef_search=10)

    ids = np.zeros(2, dtype=np.uint32)
    distances = np.zeros(2, dtype=np.float32)
    result = reader.by_array(query, n=2, ef_search=10, out=(ids, distances))

    assert result is None
    assert list(zip(ids.tolist(), distances.tolist())) == expected


def test_by_array_out_2d(array_db: hannoy.Database) -> None:
    reader: Reader = array_db.reader(0)
    queries = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    expected = reader.by_array(queries, n=1, ef_search=10)

    ids = np.zeros((2, 1), dtype=np.uint32)
    distances = np.zeros((2, 1), dtype=np.float32)
    result = reader.by_array(queries, n=1, ef_search=10, out=(ids, distances))

    assert result is None
    for row, expected_row in zip(range(2), expected):
        assert list(zip(ids[row].tolist(), distances[row].tolist())) == expected_row


def test_by_array_out_pads_missing_hits(array_db: hannoy.Database) -> None:
    reader: Reader = array_db.reader(0)
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    ids = np.zeros(5, dtype=np.uint32)
    distances = np.zeros(5, dtype=np.float32)
    reader.by_array(query, n=5, ef_search=10, out=(ids, distances))

    # only 3 items are indexed, so the last 2 slots are padding
    assert ids[3:].tolist() == [np.iinfo(np.uint32).max] * 2
    assert distances[3:].tolist() == [float("inf")] * 2


@pytest.mark.parametrize(
    ("query", "exc"),
    [
        pytest.param(
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            ValueError("dtype mismatch"),
            id="wrong-dtype",
        ),
        pytest.param(
            np.zeros((2, 2, 2), dtype=np.float32),
            ValueError("by_array requires a 1D or 2D tensor, got 3D"),
            id="wrong-ndim",
        ),
    ],
)
def test_by_array_rejects_invalid_query(
    array_db: hannoy.Database, query: np.ndarray, exc: Exception
) -> None:
    reader: Reader = array_db.reader(0)
    with pytest.raises(type(exc), match=re.escape(str(exc))):
        reader.by_array(query, n=1)


def _out_with_readonly_ids() -> tuple[np.ndarray, np.ndarray]:
    ids = np.zeros(2, dtype=np.uint32)
    ids.flags.writeable = False
    return (ids, np.zeros(2, dtype=np.float32))


@pytest.mark.parametrize(
    ("out", "exc"),
    [
        pytest.param(
            (np.zeros(2, dtype=np.uint32),),
            ValueError("expected tuple of length 2, but got tuple of length 1"),
            id="wrong-tuple-length",
        ),
        pytest.param(
            (np.zeros(2, dtype=np.int32), np.zeros(2, dtype=np.float32)),
            ValueError("dtype mismatch"),
            id="wrong-ids-dtype",
        ),
        pytest.param(
            (np.zeros(3, dtype=np.uint32), np.zeros(2, dtype=np.float32)),
            ValueError("out[0] (ids) must have shape [2], got Ok([3])"),
            id="wrong-ids-shape",
        ),
        pytest.param(
            _out_with_readonly_ids(),
            ValueError("tensor is read-only"),
            id="read-only",
        ),
    ],
)
def test_by_array_out_rejects_invalid_out(
    array_db: hannoy.Database, out: tuple[np.ndarray, np.ndarray], exc: Exception
) -> None:
    reader: Reader = array_db.reader(0)
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    with pytest.raises(type(exc), match=re.escape(str(exc))):
        reader.by_array(query, n=2, out=out)
