from pathlib import Path

import pytest

import hannoy
from hannoy import Metric


@pytest.fixture(scope="function")
def db(tmp_path: Path):
    db = hannoy.Database(tmp_path, Metric.HAMMING)

    with db.writer(3, m=4, ef=10) as writer:
        writer.add_item(0, [1.0, 0.0, 0.0])
        writer.add_item(1, [0.0, 1.0, 0.0])
        writer.add_item(2, [0.0, 0.0, 1.0])

    yield db
