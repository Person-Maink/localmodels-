#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import sys
import tarfile
import tempfile
import zipfile

from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = REPO_ROOT / "model_assets" / "manifest.yaml"
DOWNLOAD_CACHE_DIR = REPO_ROOT / "model_assets" / ".downloads"


def load_manifest(path: Path) -> dict[str, Any]:
    text = path.read_text()
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except Exception:
        return json.loads(text)


def destination_path(entry: dict[str, Any]) -> Path:
    return REPO_ROOT / entry["destination"]


def destination_exists(entry: dict[str, Any]) -> bool:
    dest = destination_path(entry)
    if not dest.exists():
        return False
    if dest.is_dir():
        return any(dest.rglob("*"))
    return True


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def is_model_assets_entry(entry: dict[str, Any]) -> bool:
    return entry["destination"].startswith("model_assets/")


def selected_entries(manifest: dict[str, Any], ids: list[str]) -> list[dict[str, Any]]:
    entries = manifest.get("assets", [])
    if not ids:
        return entries
    wanted = set(ids)
    selected = [entry for entry in entries if entry["id"] in wanted]
    missing = wanted - {entry["id"] for entry in selected}
    if missing:
        raise SystemExit(f"Unknown asset ids: {', '.join(sorted(missing))}")
    return selected


def print_out_of_scope(manifest: dict[str, Any]) -> None:
    for item in manifest.get("out_of_scope", []):
        print(f"OUT_OF_SCOPE {item}")


def require_gdown():
    try:
        import gdown  # type: ignore
    except ImportError as exc:
        raise RuntimeError("gdown is required for Google Drive downloads") from exc
    return gdown


def require_hf_hub():
    try:
        from huggingface_hub import hf_hub_download, snapshot_download  # type: ignore
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for Hugging Face downloads") from exc
    return hf_hub_download, snapshot_download


def cache_filename_for_spec(spec: dict[str, Any]) -> str:
    if spec.get("filename"):
        return spec["filename"]
    if spec["source_type"] == "hf_file":
        return spec["filename"]
    parsed = urlparse(spec["url"])
    name = Path(parsed.path).name
    return name or "download.bin"


def cached_download_path(spec: dict[str, Any]) -> Path:
    DOWNLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return DOWNLOAD_CACHE_DIR / cache_filename_for_spec(spec)


def download_url_to_path(url: str, dest: Path) -> None:
    ensure_parent(dest)
    with urlopen(Request(url, headers={"User-Agent": "model-assets-bootstrap"})) as response:
        with dest.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def resolve_source_to_local(
    spec: dict[str, Any],
    *,
    dry_run: bool,
    local_files_only: bool,
    hf_cache_root: Optional[Path] = None,
) -> Path:
    source_type = spec["source_type"]

    if source_type == "url":
        local_path = cached_download_path(spec)
        if dry_run:
            print(f"DRY_RUN download {spec['url']} -> {local_path}")
            return local_path
        if not local_path.exists():
            download_url_to_path(spec["url"], local_path)
        return local_path

    if source_type == "gdrive":
        local_path = cached_download_path(spec)
        if dry_run:
            print(f"DRY_RUN gdrive {spec['url']} -> {local_path}")
            return local_path
        if not local_path.exists():
            gdown = require_gdown()
            ensure_parent(local_path)
            gdown.download(spec["url"], output=str(local_path), fuzzy=True, quiet=False)
        return local_path

    if source_type == "hf_file":
        hf_hub_download, _ = require_hf_hub()
        kwargs: dict[str, Any] = {
            "repo_id": spec["repo_id"],
            "filename": spec["filename"],
            "local_files_only": local_files_only,
        }
        if hf_cache_root is not None:
            kwargs["cache_dir"] = str(hf_cache_root)
        if dry_run:
            print(f"DRY_RUN hf_file {spec['repo_id']}:{spec['filename']} -> {destination_path(spec)}")
            return destination_path(spec)
        return Path(hf_hub_download(**kwargs))

    if source_type == "hf_snapshot":
        _, snapshot_download = require_hf_hub()
        dest = destination_path(spec)
        kwargs = {
            "repo_id": spec["repo_id"],
            "local_dir": str(dest),
            "local_dir_use_symlinks": False,
            "local_files_only": local_files_only,
        }
        if hf_cache_root is not None:
            kwargs["cache_dir"] = str(hf_cache_root)
        if dry_run:
            print(f"DRY_RUN hf_snapshot {spec['repo_id']} -> {dest}")
            return dest
        snapshot_download(**kwargs)
        return dest

    raise ValueError(f"Unsupported source_type: {source_type}")


def safe_join(base: Path, relative: str) -> Path:
    candidate = (base / relative).resolve()
    if not str(candidate).startswith(str(base.resolve())):
        raise ValueError(f"Refusing to write outside destination root: {relative}")
    return candidate


def extract_archive_entry(entry: dict[str, Any], *, dry_run: bool) -> None:
    archive_path = resolve_source_to_local(
        entry["extract_from"],
        dry_run=dry_run,
        local_files_only=False,
    )
    dest = destination_path(entry)
    archive_member = entry.get("archive_member")
    archive_prefix = entry.get("archive_prefix")

    if archive_member is None and archive_prefix is None:
        raise ValueError(f"{entry['id']} needs archive_member or archive_prefix")

    if dry_run:
        detail = archive_member if archive_member else archive_prefix
        print(f"DRY_RUN extract {detail} from {archive_path} -> {dest}")
        return

    if archive_member:
        ensure_parent(dest)
    else:
        dest.mkdir(parents=True, exist_ok=True)

    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as archive:
            members = archive.getmembers()
            if archive_member:
                match = next((member for member in members if member.name == archive_member), None)
                if match is None:
                    raise FileNotFoundError(f"{archive_member} not found in {archive_path}")
                with archive.extractfile(match) as src, dest.open("wb") as out:
                    assert src is not None
                    shutil.copyfileobj(src, out)
                return

            for member in members:
                if not member.isfile() or not member.name.startswith(archive_prefix):
                    continue
                relative = member.name[len(archive_prefix) :]
                if not relative:
                    continue
                output_path = safe_join(dest, relative)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as src, output_path.open("wb") as out:
                    assert src is not None
                    shutil.copyfileobj(src, out)
            return

    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if archive_member:
            if archive_member not in names:
                raise FileNotFoundError(f"{archive_member} not found in {archive_path}")
            with archive.open(archive_member) as src, dest.open("wb") as out:
                shutil.copyfileobj(src, out)
            return

        for name in names:
            if name.endswith("/") or not name.startswith(archive_prefix):
                continue
            relative = name[len(archive_prefix) :]
            if not relative:
                continue
            output_path = safe_join(dest, relative)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as src, output_path.open("wb") as out:
                shutil.copyfileobj(src, out)


def fetch_entry(entry: dict[str, Any], *, dry_run: bool) -> None:
    status = entry.get("status", "ready")
    if status != "ready":
        print(f"SKIP {entry['id']} status={status}")
        return

    if destination_exists(entry):
        print(f"OK {entry['id']} already present at {destination_path(entry)}")
        return

    if entry["source_type"] == "extract_archive":
        extract_archive_entry(entry, dry_run=dry_run)
        if not dry_run:
            print(f"FETCHED {entry['id']} -> {destination_path(entry)}")
        return

    if entry["source_type"] == "hf_snapshot":
        resolve_source_to_local(entry, dry_run=dry_run, local_files_only=False)
        if not dry_run:
            print(f"FETCHED {entry['id']} -> {destination_path(entry)}")
        return

    local_path = resolve_source_to_local(entry, dry_run=dry_run, local_files_only=False)
    dest = destination_path(entry)
    if dry_run:
        print(f"DRY_RUN stage {local_path} -> {dest}")
        return
    ensure_parent(dest)
    shutil.copy2(local_path, dest)
    print(f"FETCHED {entry['id']} -> {dest}")


def format_cache_path(template: str, cache_root: Path) -> Path:
    return Path(template.format(cache_root=str(cache_root)))


def import_cache_entry(entry: dict[str, Any], *, cache_root: Path, dry_run: bool) -> None:
    if not is_model_assets_entry(entry):
        print(f"SKIP {entry['id']} import-cache only applies to shared model_assets entries")
        return

    if destination_exists(entry):
        print(f"OK {entry['id']} already present at {destination_path(entry)}")
        return

    status = entry.get("status", "ready")
    if status != "ready":
        print(f"SKIP {entry['id']} status={status}")
        return

    if entry["source_type"] == "hf_file":
        try:
            local_path = resolve_source_to_local(
                entry,
                dry_run=dry_run,
                local_files_only=True,
                hf_cache_root=cache_root / "huggingface",
            )
        except Exception:
            print(f"MISSING_CACHE {entry['id']}")
            return
        dest = destination_path(entry)
        if dry_run:
            print(f"DRY_RUN import-cache {local_path} -> {dest}")
            return
        ensure_parent(dest)
        shutil.copy2(local_path, dest)
        print(f"IMPORTED {entry['id']} -> {dest}")
        return

    if entry["source_type"] == "hf_snapshot":
        try:
            resolve_source_to_local(
                entry,
                dry_run=dry_run,
                local_files_only=True,
                hf_cache_root=cache_root / "huggingface",
            )
        except Exception:
            print(f"MISSING_CACHE {entry['id']}")
            return
        if not dry_run:
            print(f"IMPORTED {entry['id']} -> {destination_path(entry)}")
        return

    for template in entry.get("import_cache_paths", []):
        candidate = format_cache_path(template, cache_root)
        if not candidate.exists():
            continue
        dest = destination_path(entry)
        if dry_run:
            print(f"DRY_RUN import-cache {candidate} -> {dest}")
            return
        ensure_parent(dest)
        shutil.copy2(candidate, dest)
        print(f"IMPORTED {entry['id']} -> {dest}")
        return

    print(f"MISSING_CACHE {entry['id']}")


def verify_entries(entries: list[dict[str, Any]]) -> int:
    missing = 0
    for entry in entries:
        dest = destination_path(entry)
        status = entry.get("status", "ready")
        if destination_exists(entry):
            print(f"OK {entry['id']} -> {dest}")
            continue
        if status == "ready":
            print(f"MISSING {entry['id']} -> {dest}")
            missing += 1
        else:
            print(f"UNRESOLVED {entry['id']} status={status} -> {dest}")
    return missing


def list_missing(entries: list[dict[str, Any]]) -> int:
    missing = 0
    for entry in entries:
        if destination_exists(entry):
            continue
        print(f"{entry['id']}: {destination_path(entry)} ({entry.get('status', 'ready')})")
        missing += 1
    return missing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap repo-local model assets.")
    parser.add_argument(
        "command",
        choices=["fetch", "verify", "list-missing", "import-cache", "dry-run"],
        help="Action to run.",
    )
    parser.add_argument("ids", nargs="*", help="Optional asset ids to limit the command.")
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Path to the manifest file.",
    )
    parser.add_argument(
        "--cache-root",
        default=str(Path.home() / ".cache"),
        help="Cache root used by import-cache.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    manifest = load_manifest(Path(args.manifest))
    entries = selected_entries(manifest, args.ids)

    if args.command == "fetch":
        for entry in entries:
            fetch_entry(entry, dry_run=False)
        return 0

    if args.command == "dry-run":
        for entry in entries:
            fetch_entry(entry, dry_run=True)
        return 0

    if args.command == "import-cache":
        cache_root = Path(args.cache_root).expanduser()
        for entry in entries:
            import_cache_entry(entry, cache_root=cache_root, dry_run=False)
        return 0

    if args.command == "verify":
        missing = verify_entries(entries)
        print_out_of_scope(manifest)
        return 1 if missing else 0

    if args.command == "list-missing":
        list_missing(entries)
        print_out_of_scope(manifest)
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
