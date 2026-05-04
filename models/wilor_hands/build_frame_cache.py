import argparse

from frame_store import DEFAULT_JPEG_QUALITY, build_frame_caches


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sidecar ZIP frame caches for WiLoR videos.")
    parser.add_argument("--input_root", type=str, required=True, help="Directory containing raw videos.")
    parser.add_argument("--video", type=str, default=None, help="Optional video stem to cache.")
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Rebuild an existing cache instead of skipping it.",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=DEFAULT_JPEG_QUALITY,
        help="JPEG quality to use when packing decoded frames into the ZIP archive.",
    )
    args = parser.parse_args()

    results = build_frame_caches(
        args.input_root,
        video_name=args.video,
        overwrite=args.overwrite,
        jpeg_quality=args.jpeg_quality,
    )
    if not results:
        print("No raw videos found.")
        return

    for result in results:
        print(
            f"[{result['status']}] {result['video_name']}: "
            f"{result['frame_count']} frame(s) -> {result['zip_path']}"
        )


if __name__ == "__main__":
    main()
