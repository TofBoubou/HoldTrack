"""Cache complete pipeline stages with input and output checksums."""

import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path


def file_hash(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(value: dict | list) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path: Path, value: dict | list) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class StageCache:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def locked(self) -> Iterator[None]:
        with (self.directory / ".lock").open("a") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Another process is using this work directory") from error
            yield

    def stage(
        self, name: str, inputs: dict, filename: str, produce: Callable[[Path], None]
    ) -> Path:
        """Reuse a verified stage or commit a complete replacement atomically."""
        inputs = json.loads(json.dumps(inputs, sort_keys=True))
        signature = fingerprint(inputs)
        directory = self.directory / name / signature
        marker = directory / "complete.json"
        if marker.exists():
            saved = json.loads(marker.read_text())
            output = directory / saved["output"]
            if saved["inputs"] != inputs or file_hash(output) != saved["sha256"]:
                raise ValueError(f"Cached output was modified: {name}")
            return output
        directory.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".partial-", dir=directory.parent))
        try:
            output = temporary / filename
            produce(output)
            if not output.is_file():
                raise RuntimeError(f"Stage did not produce its output: {name}")
            atomic_json(
                temporary / "complete.json",
                dict(inputs=inputs, output=filename, sha256=file_hash(output)),
            )
            if directory.exists():
                raise ValueError(f"Uncommitted cache entry already exists: {directory}")
            temporary.replace(directory)
            return directory / filename
        except BaseException:
            raise

    def import_file(self, name: str, source: Path) -> Path:
        return self.stage(
            name,
            dict(source_sha256=file_hash(source)),
            source.name,
            lambda output: shutil.copyfile(source, output),
        )
