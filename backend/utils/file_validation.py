from app.exceptions import AnalysisError


def require_filename(filename: str | None) -> str:
    if not filename:
        raise AnalysisError("A file is required")
    return filename
