from pathlib import Path

import numpy as np
import pytest

import hannoy
from hannoy import Reader

def test_by_items_matches_by_item(array_db: hannoy.Database) -> None:
    reader: Reader = array_db.reader(0)
    batch = reader.by_items([1, 2, 3], n = 2, ef_search = 10)
    per_item = [reader.by_item(i, n = 2, ef_search= 10) for i in (1, 2, 3)]
    assert batch == per_item

def test_by_items_batches_query(array_db: hannoy.Database) -> None:
    reader: Reader = array_db.reader(0)
    res = reader.by_items([0, 1, 2], n = 2, ef_search = 10)
    assert(len(res)) == 3
    assert [sorted(i for i, _ in row) for row in res] == [[1, 2], [0, 2], [0, 1]]



def test_by_items_batches_queries(array_db: hannoy.Database) -> None:
    reader: Reader = array_db.reader(0)
    res = reader.by_items([0, 1], n=2, ef_search=10)
    assert len(res) == 2
    assert [sorted(i for i, _ in row) for row in res] == [[1, 2], [0, 2]]


def test_by_items_none_for_missing_id(array_db: hannoy.Database) -> None:
    reader: Reader = array_db.reader(0)
    res = reader.by_items([0, 999], n=2, ef_search=10)

    assert res[0] is not None
    assert res[1] is None


def test_by_items_preserves_order_with_missing(array_db: hannoy.Database) -> None:
    reader: Reader = array_db.reader(0)
    res = reader.by_items([0, 999, 1], n=2, ef_search=10)
    assert len(res) == 3
    assert res[0] is not None
    assert res[1] is None
    assert res[2] is not None


def test_by_items_duplicate_ids(array_db: hannoy.Database) -> None:
    reader: Reader = array_db.reader(0)
    res = reader.by_items([0, 0, 1], n=2, ef_search=10)
    assert len(res) == 3
    assert res[0] == res[1]
    assert res[0] == reader.by_item(0, n=2, ef_search=10)


def test_by_items_empty(array_db: hannoy.Database) -> None:
    reader: Reader = array_db.reader(0)
    assert reader.by_items([], n=2, ef_search=10) == []