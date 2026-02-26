#!/usr/bin/env python3
import argparse
import re
import shutil
import zipfile
from pathlib import Path

import requests
import torch


DETECTRON2_URL = (
    "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
    "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
)
HAMBA_ZIP_FILE_ID = "1JRPC11EQGJY5ANiiQf8cr_7CSg1R7Mce"  # official hamba.zip

def download_vitpose_wholebody_pth(
    dst_path="/scratch/mthakur/hamba/_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth",
    url="PUT_THE_EXACT_VITPOSE_WHOLEBODY_URL_HERE",
    force=False,
):
    import re
    from pathlib import Path
    import requests
    import torch

    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and dst.stat().st_size > 0 and not force:
        # Validate existing file
        torch.load(dst, map_location="cpu")
        return str(dst)

    def _save_response(resp, out_path):
        resp.raise_for_status()
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if "text/html" in ctype:
            raise RuntimeError(
                f"Got HTML instead of checkpoint from {resp.url}. "
                "This is usually a bad/blocked link."
            )
        tmp = out_path.with_suffix(out_path.suffix + ".part")
        with tmp.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        tmp.replace(out_path)

    with requests.Session() as s:
        # Normal URL path
        r = s.get(url, stream=True, timeout=180, allow_redirects=True)

        # Basic Google Drive confirm-token handling
        if "drive.google.com" in url:
            token = None
            for k, v in r.cookies.items():
                if k.startswith("download_warning"):
                    token = v
                    break
            if token:
                r = s.get(url, params={"confirm": token}, stream=True, timeout=180)
            else:
                try:
                    html = r.text
                    m = re.search(r"confirm=([0-9A-Za-z_]+)&", html)
                    if m:
                        r = s.get(url, params={"confirm": m.group(1)}, stream=True, timeout=180)
                except Exception:
                    pass

        _save_response(r, dst)

    # Final validation: must be a real torch checkpoint
    torch.load(dst, map_location="cpu")
    return str(dst)


def _stream_to_file(resp: requests.Response, dst: Path) -> None:
    resp.raise_for_status()
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with tmp.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    tmp.replace(dst)


def download_http(url: str, dst: Path, force: bool = False) -> None:
    if dst.exists() and dst.stat().st_size > 0 and not force:
        print(f"[skip] exists: {dst}")
        return
    print(f"[download] {url}\n  -> {dst}")
    with requests.get(url, stream=True, timeout=120) as r:
        _stream_to_file(r, dst)


def download_google_drive(file_id: str, dst: Path, force: bool = False) -> None:
    if dst.exists() and dst.stat().st_size > 0 and not force:
        print(f"[skip] exists: {dst}")
        return

    url = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    print(f"[download] Google Drive file id={file_id}\n  -> {dst}")
    r = session.get(url, params={"id": file_id}, stream=True, timeout=120)

    token = None
    for k, v in r.cookies.items():
        if k.startswith("download_warning"):
            token = v
            break

    if token:
        r = session.get(
            url, params={"id": file_id, "confirm": token}, stream=True, timeout=120
        )
    else:
        ctype = r.headers.get("Content-Type", "").lower()
        if "text/html" in ctype:
            html = r.text
            m = re.search(r"confirm=([0-9A-Za-z_]+)&(?:amp;)?id=", html)
            if m:
                r = session.get(
                    url,
                    params={"id": file_id, "confirm": m.group(1)},
                    stream=True,
                    timeout=120,
                )

    _stream_to_file(r, dst)


def extract_hamba_ckpt(zip_path: Path, target_ckpt: Path) -> None:
    target_ckpt.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = zip_path.parent / "_extract_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_dir)

    candidates = list(tmp_dir.rglob("hamba.ckpt"))
    if not candidates:
        raise FileNotFoundError("Could not find hamba.ckpt inside downloaded zip.")

    src = candidates[0]
    shutil.copy2(src, target_ckpt)
    shutil.rmtree(tmp_dir)
    print(f"[ok] extracted -> {target_ckpt}")


def patch_detectron_url(repo_root: Path, local_detectron_path: Path) -> None:
    replaced = []
    local_str = local_detectron_path.as_posix()

    for pattern in ("*.py", "*.yaml", "*.yml"):
        for p in repo_root.rglob(pattern):
            try:
                txt = p.read_text(encoding="utf-8")
            except Exception:
                continue
            if DETECTRON2_URL in txt:
                p.write_text(txt.replace(DETECTRON2_URL, local_str), encoding="utf-8")
                replaced.append(p)

    if replaced:
        print("[ok] patched detectron2 URL in:")
        for p in replaced:
            print(f"  - {p}")
    else:
        print("[warn] no config files patched (URL string not found).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/home/mthakur/.cache/torch/hub"),
        help="Hamba repo root",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if files already exist",
    )
    parser.add_argument(
        "--no-patch",
        action="store_true",
        help="Do not patch config files to use local detectron2 pkl",
    )
    args = parser.parse_args()

    print(f"torch: {torch.__version__}")

    repo_root = args.repo_root
    detectron_dst = repo_root / "detectron2" / "model_final_f05665.pkl"
    hamba_zip = repo_root / "ckpts" / "hamba" / "hamba.zip"
    hamba_ckpt = repo_root / "ckpts" / "hamba" / "checkpoints" / "hamba.ckpt"

    download_http(DETECTRON2_URL, detectron_dst, force=args.force)
    # download_google_drive(HAMBA_ZIP_FILE_ID, hamba_zip, force=args.force)
    # extract_hamba_ckpt(hamba_zip, hamba_ckpt)
    download_vitpose_wholebody_pth(
    url="YOUR_REAL_WHOLEBODY.PTH_LINK",
    force=True,
)


    if not args.no_patch:
        patch_detectron_url(repo_root, detectron_dst)

    print("\nDone.")
    print(f"Detectron2 weights: {detectron_dst}")
    print(f"Hamba checkpoint:   {hamba_ckpt}")


if __name__ == "__main__":
    main()
