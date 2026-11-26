from pathlib import Path


def resolve_repo_path(
    path_value: str | Path,
    repo_root: Path,
) -> Path:
    path = Path(path_value)

    if not path.is_absolute():
        path = repo_root / path

    return path.resolve()