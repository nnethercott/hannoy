from pathlib import Path
import hannoy


def test_exports() -> None:
    assert hannoy.__all__ == ["Metric", "Database", "Writer"]


def test_create(tmp_path: Path) -> None:
    db = hannoy.Database(tmp_path)

    with db.writer(0, 3) as writer:
        writer.add_item(0, [0.1, 0.2, 0.3])
