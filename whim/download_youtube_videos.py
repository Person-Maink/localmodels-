from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import shutil
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


try:
    import yt_dlp
except ImportError as exc:  # pragma: no cover - import error is user-environment dependent
    raise SystemExit("Missing dependency: install yt-dlp with `python -m pip install yt-dlp`.") from exc


@dataclass(frozen=True)
class VideoSpec:
    video_id: str
    height: int
    width: int
    fps: float
    length: int | None

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"


def parse_args() -> argparse.Namespace:
    script_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Download a set of YouTube videos from a WiLoR-style metadata JSON using yt-dlp."
    )
    parser.add_argument("--root", type=Path, default=script_root, help="WiLoR root directory.")
    parser.add_argument("--mode", choices=["train", "test"], default="train", help="Metadata split to use by default.")
    parser.add_argument(
        "--metadata_json",
        type=Path,
        default=None,
        help="Override metadata JSON path. Defaults to <root>/whim/<mode>_video_ids.json.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to store downloaded videos. Defaults to <root>/Videos.",
    )
    parser.add_argument(
        "--video_id",
        action="append",
        default=[],
        help="Download only a specific video ID. Repeat the flag to select multiple videos.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of videos to download.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of videos to download in parallel.",
    )
    parser.add_argument(
        "--concurrent_fragments",
        type=int,
        default=1,
        help="Number of media fragments yt-dlp may fetch in parallel per video.",
    )
    parser.add_argument(
        "--cookies_from_browser",
        type=str,
        default=None,
        help="Browser name for yt-dlp cookies import, for example chrome or firefox.",
    )
    parser.add_argument(
        "--cookies",
        type=Path,
        default=None,
        help="Optional Netscape-format cookies.txt file. Prefer this on HPC if browser profiles are unavailable.",
    )
    parser.add_argument(
        "--allow_cookie_fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retry without browser cookies when yt-dlp cannot access the requested browser profile.",
    )
    parser.add_argument("--retries", type=int, default=10, help="yt-dlp retry count for transient failures.")
    parser.add_argument(
        "--skip_existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip downloads when a file with the same video ID already exists in output_dir.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail a video if no exact width/height/fps match is available instead of using the closest stream.",
    )
    parser.add_argument(
        "--fps_tolerance",
        type=float,
        default=0.25,
        help="Accepted absolute fps difference for an exact match.",
    )
    parser.add_argument(
        "--download_archive",
        type=Path,
        default=None,
        help="Optional yt-dlp download archive file to avoid re-downloading URLs across runs.",
    )
    parser.add_argument(
        "--report_path",
        type=Path,
        default=None,
        help="Where to save the JSON report. Defaults to <output_dir>/download_report_<mode>.json.",
    )
    parser.add_argument(
        "--request_delay",
        type=float,
        default=10.0,
        help="Seconds to wait between starting video requests.",
    )
    return parser.parse_args()


def load_video_specs(metadata_path: Path, selected_ids: set[str] | None = None) -> list[VideoSpec]:
    raw = json.loads(metadata_path.read_text())
    specs: list[VideoSpec] = []
    for video_id, info in raw.items():
        if selected_ids and video_id not in selected_ids:
            continue

        res = info.get("res")
        if not isinstance(res, list) or len(res) != 2:
            raise ValueError(f"Invalid res field for {video_id}: {res!r}")

        specs.append(
            VideoSpec(
                video_id=video_id,
                height=int(res[0]),
                width=int(res[1]),
                fps=float(info.get("fps", 0.0)),
                length=int(info["length"]) if info.get("length") is not None else None,
            )
        )
    return specs


def existing_download(output_dir: Path, video_id: str) -> Path | None:
    matches = sorted(output_dir.glob(f"{video_id}.*"))
    for match in matches:
        if match.is_file():
            return match
    return None


def score_format(fmt: dict[str, Any], spec: VideoSpec, fps_tolerance: float) -> tuple[Any, ...] | None:
    if fmt.get("vcodec") in (None, "none"):
        return None

    height = fmt.get("height")
    width = fmt.get("width")
    fps = fmt.get("fps")
    if height is None or width is None:
        return None

    height = int(height)
    width = int(width)
    fps_value = float(fps) if fps is not None else math.inf
    target_aspect = spec.width / spec.height
    aspect = width / height
    ext = fmt.get("ext") or ""
    protocol = fmt.get("protocol") or ""
    has_audio = fmt.get("acodec") not in (None, "none")

    return (
        0 if ext == "mp4" else 1,
        0 if not protocol.startswith("m3u8") else 1,
        0 if height == spec.height else 1,
        abs(height - spec.height),
        0 if width == spec.width else 1,
        abs(width - spec.width),
        0 if math.isfinite(fps_value) and abs(fps_value - spec.fps) <= fps_tolerance else 1,
        abs(fps_value - spec.fps) if math.isfinite(fps_value) else math.inf,
        abs(aspect - target_aspect),
        0 if has_audio else 1,
        -(fmt.get("tbr") or 0.0),
    )


def rank_formats(formats: list[dict[str, Any]], spec: VideoSpec, fps_tolerance: float) -> list[dict[str, Any]]:
    ranked: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for fmt in formats:
        score = score_format(fmt, spec, fps_tolerance)
        if score is not None:
            ranked.append((score, fmt))

    if not ranked:
        raise RuntimeError("No downloadable video formats were found.")

    ranked.sort(key=lambda item: item[0])
    return [fmt for _, fmt in ranked]


def choose_format(formats: list[dict[str, Any]], spec: VideoSpec, fps_tolerance: float) -> dict[str, Any]:
    return rank_formats(formats, spec, fps_tolerance)[0]


def is_exact_match(selected: dict[str, Any], spec: VideoSpec, fps_tolerance: float) -> bool:
    height = selected.get("height")
    width = selected.get("width")
    fps = selected.get("fps")
    if height is None or width is None or fps is None:
        return False
    return (
        int(height) == spec.height
        and int(width) == spec.width
        and abs(float(fps) - spec.fps) <= fps_tolerance
    )


def format_summary(selected: dict[str, Any]) -> dict[str, Any]:
    return {
        "format_id": selected.get("format_id"),
        "ext": selected.get("ext"),
        "width": selected.get("width"),
        "height": selected.get("height"),
        "fps": selected.get("fps"),
        "vcodec": selected.get("vcodec"),
        "acodec": selected.get("acodec"),
        "protocol": selected.get("protocol"),
        "tbr": selected.get("tbr"),
    }


def should_retry_with_next_format(exc: Exception) -> bool:
    message = str(exc).lower()
    retry_markers = (
        "http error 403",
        "forbidden",
        "requested format is not available",
        "unable to download video data",
        "requested format not available",
    )
    return any(marker in message for marker in retry_markers)


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def fallback_format_selectors(spec: VideoSpec, *, allow_merge: bool) -> list[str]:
    selectors = [
        "best[ext=mp4]"
        f"[height<={spec.height}]"
        f"[width<={spec.width}]",
        "best[ext=mp4]",
        "best",
    ]
    if not allow_merge:
        return selectors

    bounded_mp4_merge = (
        "bestvideo[ext=mp4]"
        f"[height<={spec.height}]"
        f"[width<={spec.width}]"
        "+bestaudio[ext=m4a]/"
        "best[ext=mp4]"
        f"[height<={spec.height}]"
        f"[width<={spec.width}]"
    )
    return selectors + [
        bounded_mp4_merge,
        "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
        "bestvideo+bestaudio/best",
    ]


def available_formats_summary(formats: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for fmt in formats:
        if fmt.get("vcodec") in (None, "none"):
            continue
        lines.append(
            "    - format_id={format_id} ext={ext} size={width}x{height} fps={fps} "
            "vcodec={vcodec} acodec={acodec} protocol={protocol}".format(
                format_id=fmt.get("format_id"),
                ext=fmt.get("ext"),
                width=fmt.get("width"),
                height=fmt.get("height"),
                fps=fmt.get("fps"),
                vcodec=fmt.get("vcodec"),
                acodec=fmt.get("acodec"),
                protocol=fmt.get("protocol"),
            )
        )
    if not lines:
        return "    (no downloadable video formats reported)"
    return "\n".join(lines)


def log_message(message: str, *, print_lock: threading.Lock | None = None) -> None:
    if print_lock is None:
        print(message, file=sys.stderr, flush=True)
        return
    with print_lock:
        print(message, file=sys.stderr, flush=True)


def should_retry_without_browser_cookies(exc: Exception) -> bool:
    message = str(exc).lower()
    return "could not find" in message and "cookies" in message and "database" in message


def extract_info_with_fallback(
    url: str,
    ydl_opts: dict[str, Any],
    *,
    download: bool,
    allow_cookie_fallback: bool,
) -> Any:
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(url, download=download)
    except Exception as exc:
        if allow_cookie_fallback and "cookiesfrombrowser" in ydl_opts and should_retry_without_browser_cookies(exc):
            browser = ydl_opts["cookiesfrombrowser"][0]
            print(f"  warning: browser cookies for {browser} are unavailable here, retrying without them")
            fallback_opts = dict(ydl_opts)
            fallback_opts.pop("cookiesfrombrowser", None)
            with yt_dlp.YoutubeDL(fallback_opts) as ydl:
                return ydl.extract_info(url, download=download)
        raise


def download_video(
    spec: VideoSpec,
    output_dir: Path,
    cookies_from_browser: str | None,
    cookie_file: Path | None,
    allow_cookie_fallback: bool,
    concurrent_fragments: int,
    retries: int,
    skip_existing: bool,
    strict: bool,
    fps_tolerance: float,
    download_archive: Path | None,
    print_lock: threading.Lock | None = None,
) -> dict[str, Any]:
    existing = existing_download(output_dir, spec.video_id)
    if existing is not None and skip_existing:
        return {
            "status": "skipped_existing",
            "output_path": str(existing),
            "spec": asdict(spec),
        }

    shared_opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "retries": retries,
    }
    if concurrent_fragments > 1:
        shared_opts["concurrent_fragment_downloads"] = concurrent_fragments
    if cookies_from_browser:
        shared_opts["cookiesfrombrowser"] = (cookies_from_browser,)
    if cookie_file is not None:
        shared_opts["cookiefile"] = str(cookie_file)
    if download_archive is not None:
        shared_opts["download_archive"] = str(download_archive)
    ffmpeg_available = has_ffmpeg()

    info = extract_info_with_fallback(
        spec.url,
        shared_opts,
        download=False,
        allow_cookie_fallback=allow_cookie_fallback,
    )
    formats_text = available_formats_summary(info.get("formats") or [])
    log_message(f"  available formats for {spec.video_id}:\n{formats_text}", print_lock=print_lock)

    ranked_formats = rank_formats(info.get("formats") or [], spec, fps_tolerance)
    selected = ranked_formats[0]
    exact_match = is_exact_match(selected, spec, fps_tolerance)
    if strict and not exact_match:
        raise RuntimeError(
            f"No exact stream match found. Closest was {selected.get('width')}x{selected.get('height')} @ "
            f"{selected.get('fps')} fps."
        )

    last_error: Exception | None = None
    candidate_formats = ranked_formats[: min(8, len(ranked_formats))]
    for attempt_idx, candidate in enumerate(candidate_formats, start=1):
        download_opts = dict(shared_opts)
        download_opts.update(
            {
                "format": candidate["format_id"],
                "outtmpl": str(output_dir / f"{spec.video_id}.%(ext)s"),
            }
        )
        try:
            if attempt_idx > 1:
                log_message(
                    f"  retrying {spec.video_id} with alternate format {candidate.get('format_id')}",
                    print_lock=print_lock,
                )
            extract_info_with_fallback(
                spec.url,
                download_opts,
                download=True,
                allow_cookie_fallback=allow_cookie_fallback,
            )
            selected = candidate
            break
        except Exception as exc:
            last_error = exc
            if not should_retry_with_next_format(exc):
                raise
    else:
        for selector in fallback_format_selectors(spec, allow_merge=ffmpeg_available):
            download_opts = dict(shared_opts)
            download_opts.update(
                {
                    "format": selector,
                    "outtmpl": str(output_dir / f"{spec.video_id}.%(ext)s"),
                }
            )
            try:
                log_message(
                    f"  retrying {spec.video_id} with selector fallback: {selector}",
                    print_lock=print_lock,
                )
                extract_info_with_fallback(
                    spec.url,
                    download_opts,
                    download=True,
                    allow_cookie_fallback=allow_cookie_fallback,
                )
                selected = candidate_formats[0]
                break
            except Exception as exc:
                last_error = exc
        else:
            if last_error is not None:
                raise last_error

    downloaded = existing_download(output_dir, spec.video_id)
    if downloaded is None:
        raise RuntimeError("Download completed but no output file was found.")

    return {
        "status": "downloaded",
        "output_path": str(downloaded),
        "spec": asdict(spec),
        "selected_format": format_summary(selected),
        "exact_match": exact_match,
    }


def main() -> None:
    args = parse_args()
    metadata_path = args.metadata_json or (args.root / "whim" / f"{args.mode}_video_ids.json")
    output_dir = args.output_dir or (args.root / "Videos")
    report_path = args.report_path or (output_dir / f"download_report_{args.mode}.json")

    if not metadata_path.is_file():
        raise SystemExit(f"Metadata JSON not found: {metadata_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if args.download_archive is not None:
        args.download_archive.parent.mkdir(parents=True, exist_ok=True)
    if args.cookies is not None and not args.cookies.is_file():
        raise SystemExit(f"Cookie file not found: {args.cookies}")

    selected_ids = set(args.video_id) if args.video_id else None
    specs = load_video_specs(metadata_path, selected_ids=selected_ids)
    if args.limit > 0:
        specs = specs[: args.limit]
    skipped_existing = 0
    if args.skip_existing:
        pending_specs: list[VideoSpec] = []
        for spec in specs:
            if existing_download(output_dir, spec.video_id) is not None:
                skipped_existing += 1
                continue
            pending_specs.append(spec)
        specs = pending_specs
    if not specs:
        if skipped_existing > 0:
            raise SystemExit(
                f"No pending videos matched the requested selection. "
                f"Skipped {skipped_existing} already-downloaded videos."
            )
        raise SystemExit("No videos matched the requested selection.")

    report: dict[str, Any] = {
        "metadata_path": str(metadata_path),
        "output_dir": str(output_dir),
        "mode": args.mode,
        "strict": args.strict,
        "fps_tolerance": args.fps_tolerance,
        "jobs": args.jobs,
        "concurrent_fragments": args.concurrent_fragments,
        "results": {},
    }

    total = len(specs)
    failures = 0
    jobs = max(1, args.jobs)
    print(
        f"Starting downloads with jobs={jobs}, concurrent_fragments={args.concurrent_fragments}, "
        f"pending_videos={total}"
    )
    if skipped_existing > 0:
        print(f"Skipping {skipped_existing} already-downloaded videos before scheduling requests.")

    print_lock = threading.Lock()
    report_lock = threading.Lock()
    request_delay = max(0.0, args.request_delay)

    def process_one(index: int, spec: VideoSpec) -> tuple[int, VideoSpec, dict[str, Any]]:
        with print_lock:
            print(
                f"[{index:04d}/{total:04d}] {spec.video_id} -> target "
                f"{spec.width}x{spec.height} @ {spec.fps:.3f} fps"
            )

        try:
            result = download_video(
                spec=spec,
                output_dir=output_dir,
                cookies_from_browser=args.cookies_from_browser,
                cookie_file=args.cookies,
                allow_cookie_fallback=args.allow_cookie_fallback,
                concurrent_fragments=args.concurrent_fragments,
                retries=args.retries,
                skip_existing=args.skip_existing,
                strict=args.strict,
                fps_tolerance=args.fps_tolerance,
                download_archive=args.download_archive,
                print_lock=print_lock,
            )
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            result = {"status": "failed", "spec": asdict(spec), "error": str(exc)}
            with print_lock:
                print(f"  failed {spec.video_id}: {exc}")
            return index, spec, result

        selected = result.get("selected_format")
        with print_lock:
            if selected:
                print(
                    f"  saved {result['output_path']} using {selected['width']}x{selected['height']} "
                    f"@ {selected['fps']} fps ({selected['ext']})"
                )
            else:
                print(f"  skipped existing file {result['output_path']}")
        return index, spec, result

    if jobs == 1:
        def iter_sequential() -> Any:
            for index, spec in enumerate(specs, start=1):
                if index > 1 and request_delay > 0:
                    time.sleep(request_delay)
                yield process_one(index, spec)

        iterator = iter_sequential()
    else:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=jobs)
        future_to_index = {}
        for index, spec in enumerate(specs, start=1):
            if index > 1 and request_delay > 0:
                time.sleep(request_delay)
            future = executor.submit(process_one, index, spec)
            future_to_index[future] = index
        iterator = (future.result() for future in concurrent.futures.as_completed(future_to_index))

    try:
        for _, spec, result in iterator:
            if result["status"] == "failed":
                failures += 1
            with report_lock:
                report["results"][spec.video_id] = result
                report_path.write_text(json.dumps(report, indent=2))
    finally:
        if jobs != 1:
            executor.shutdown(wait=True)

    print(f"Finished. {total - failures} succeeded, {failures} failed.")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
