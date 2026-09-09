"""Hash-checked removal of the retired indicator implementation.

Default is read-only. Run with --apply in a terminal to remove only reviewed
files. No source backups, data rewrites, Git commits or recursive deletes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


class CleanupBlocked(RuntimeError):
    pass


def planned_files(root: Path, manifest: dict) -> list[tuple[Path, str]]:
    root = root.resolve()
    if manifest.get("schema_version") != 1:
        raise CleanupBlocked("Unsupported cleanup manifest")
    checked: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for item in manifest["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] not in {"backend", "frontend", "docs"}:
            raise CleanupBlocked(f"Unsafe path: {relative}")
        if str(relative) in seen:
            raise CleanupBlocked(f"Duplicate path: {relative}")
        seen.add(str(relative))
        path = root / relative
        if path.is_symlink() or path.resolve() != path.absolute():
            raise CleanupBlocked(f"Symlink refused: {relative}")
        if not path.exists():
            continue
        if not path.is_file():
            raise CleanupBlocked(f"Not a regular file: {relative}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise CleanupBlocked(f"File changed since review; nothing removed: {relative}")
        checked.append((path, digest))
    return checked


def guard_saved_contracts(root: Path) -> None:
    # Only inspect definition/config stores, never market data or run snapshots.
    configured = os.environ.get("CUSTOM_INDICATOR_DATA_DIR")
    if not configured:
        env = root / ".env"
        if env.is_file():
            for line in env.read_text(encoding="utf-8").splitlines():
                key, separator, value = line.partition("=")
                if separator and key.strip() == "CUSTOM_INDICATOR_DATA_DIR":
                    configured = value.strip().strip('\"\'')
    directory = Path(configured).expanduser() if configured else root / "data"
    if not directory.is_absolute():
        directory = root / directory
    for filename in ("custom_indicators.json", "evaluation_plans.json", "snapshot_indicator_config.json"):
        path = directory / filename
        if not path.is_file():
            continue
        pending = [json.loads(path.read_text(encoding="utf-8"))]
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                if value.get("result_kind") == "scalar_bundle" or "scalar_outputs" in value or "output_id" in value:
                    raise CleanupBlocked(f"Saved child-result contract still exists in {filename}; migrate it before removing code")
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)


def finalize(root: Path, manifest: dict, *, apply: bool = False) -> int:
    guard_saved_contracts(root)
    checked = planned_files(root, manifest)  # Validate everything before removal.
    if apply:
        for path, digest in checked:
            if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise CleanupBlocked(f"Concurrent edit detected: {path.name}")
            path.unlink()
    return len(checked)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Physically remove the reviewed retired files")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    git = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=root, text=True, capture_output=True, check=False)
    if git.returncode or Path(git.stdout.strip()).resolve() != root:
        raise CleanupBlocked("Run this tool from its reviewed Git working tree")
    manifest_path = root / "docs/indicator_retirement_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    count = finalize(root, manifest, apply=args.apply)
    if args.apply:
        # This helper and manifest are one-shot cleanup tooling, not runtime
        # source. Remove them only after every reviewed file passed the guards.
        manifest_path.unlink()
        Path(__file__).unlink()
    print(json.dumps({"mode": "applied" if args.apply else "read_only", "files": count, "data_modified": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CleanupBlocked, OSError, ValueError, KeyError) as error:
        print(f"Cleanup blocked: {error}")
        raise SystemExit(1)
