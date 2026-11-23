"""All writes belong to this checkout; legacy trees are read-only inputs."""
from pathlib import Path
from datetime import datetime, timezone
import json

ROOT = Path(__file__).resolve().parents[1]


def output_path(path: Path) -> Path:
    path = path.resolve()
    if not path.is_relative_to(ROOT / "outputs") or path == ROOT / "outputs":
        raise ValueError("Output must be a child of this project's outputs/; no escaping symlinks")
    return path


def new_run(prefix: str, *, group: tuple[str, ...] = ()) -> Path:
    if not prefix.isidentifier():
        raise ValueError("Run prefix must be an identifier")
    if any(not part.isidentifier() for part in group):
        raise ValueError("Run group components must be identifiers")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    path = output_path(ROOT.joinpath("outputs", *group, f"{prefix}_{stamp}"))
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, value) -> None:
    output_path(path)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def replace_json(path: Path, value) -> None:
    """Atomic progress update in an already isolated run directory."""
    path = output_path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    output_path(temp)
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temp.replace(path)
