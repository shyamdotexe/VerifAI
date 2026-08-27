from pathlib import Path


def remove_file(path: str | Path) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass
