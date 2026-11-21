"""Provenance log: every download appends an entry to data/raw/MANIFEST.json."""
import json
from datetime import datetime, timezone
from pathlib import Path


def record(raw_dir: str | Path, source: str, **details) -> None:
    path = Path(raw_dir) / "MANIFEST.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = json.loads(path.read_text()) if path.exists() else []
    entries.append(
        {"source": source, "retrieved_utc": datetime.now(timezone.utc).isoformat(), **details}
    )
    path.write_text(json.dumps(entries, indent=2, default=str))
