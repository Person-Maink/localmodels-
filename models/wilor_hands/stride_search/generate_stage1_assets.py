#!/usr/bin/env python3
"""Create reproducible STRIDE sweep split manifests and stage-1 HMP configs."""
from __future__ import annotations

import csv
import json
import re
import random
from itertools import product
from collections import Counter
from pathlib import Path

SEARCH_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = SEARCH_ROOT.parent
PROJECT_ROOT = MODEL_ROOT.parent.parent
GROUND_TRUTH_CSV = PROJECT_ROOT / "data" / "new_dataset" / "ground_truth.csv"
VIDEO_ROOT = PROJECT_ROOT / "data" / "new_dataset" / "processed"
WILOR_ROOT = PROJECT_ROOT / "outputs" / "wilor"
BASELINE_CONFIG = MODEL_ROOT / "stride_configs" / "hmp.yaml"
SEED = 20260813
TEST_PERSON = "niti"
ANGLES = ("0", "45", "90", "135", "180", "BE")


def _clip_id(filename: str) -> str:
    return Path(filename).stem


def _parts(clip_id: str) -> tuple[str, str, str]:
    tokens = clip_id.split("_")
    return "_".join(tokens[:-2]), tokens[-2], tokens[-1]


def _load_clips() -> list[str]:
    with GROUND_TRUTH_CSV.open(newline="", encoding="utf-8") as handle:
        clips = sorted(_clip_id(row["video_filename"]) for row in csv.DictReader(handle))
    if len(clips) != 120 or len(set(clips)) != 120:
        raise ValueError(f"Expected 120 unique annotated clips, found {len(clips)}.")
    return clips


def _validate_assets(clips: list[str]) -> None:
    missing_videos = [clip for clip in clips if not (VIDEO_ROOT / f"{clip}.mp4").is_file()]
    missing_wilor = [clip for clip in clips if not (WILOR_ROOT / clip / "meshes").is_dir()]
    if missing_videos or missing_wilor:
        errors = []
        if missing_videos:
            errors.append("videos: " + ", ".join(missing_videos[:10]))
        if missing_wilor:
            errors.append("WiLoR meshes: " + ", ".join(missing_wilor[:10]))
        raise FileNotFoundError("Missing required sweep inputs (" + "; ".join(errors) + ")")


def _stage1_subset(development: list[str]) -> list[str]:
    """Select 18 clips: 3 per angle, 4/4/5/5 conditions, and 5/5/4/4 people."""
    by_key = {(condition, angle, person): clip for clip in development for condition, angle, person in [_parts(clip)]}
    conditions = sorted({key[0] for key in by_key})
    people = ("hari", "mayank", "meenakshi", "soumen")
    if len(conditions) != 4 or len(by_key) != 96:
        raise ValueError("Development inventory is not 4 conditions x 6 angles x 4 people.")
    person_patterns = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3), (0, 1, 2), (0, 1, 3))
    selected = []
    for index, angle in enumerate(ANGLES):
        chosen_conditions = [condition for position, condition in enumerate(conditions) if position != index % 4]
        for condition, person_index in zip(chosen_conditions, person_patterns[index]):
            selected.append(by_key[(condition, angle, people[person_index])])
    return sorted(selected)


def _counts(clips: list[str]) -> dict[str, dict[str, int]]:
    return {
        "condition": dict(sorted(Counter(_parts(clip)[0] for clip in clips).items())),
        "angle": dict(sorted(Counter(_parts(clip)[1] for clip in clips).items())),
        "participant": dict(sorted(Counter(_parts(clip)[2] for clip in clips).items())),
    }


def _write_split(name: str, clips: list[str], description: str) -> None:
    path = SEARCH_ROOT / "splits" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"stage": name, "description": description, "seed": SEED, "video_ids": clips, "stratification_counts": _counts(clips)}, indent=2) + "\n", encoding="utf-8")


def _config_payload(base: str, *, iters: int, lr: float, overlap: int, beta_enabled: bool, beta_iters: int) -> str:
    """Patch only the five swept scalar values in the baseline YAML.

    Keeping this dependency-free lets manifests/configs be generated on login
    nodes that do not have the inference container's PyYAML installed.
    """
    payload = re.sub(r"(?m)^(  overlap:)\s*.*$", rf"\g<1> {overlap}", base)
    payload = re.sub(r"(?m)^(    iters:)\s*.*$", rf"\g<1> {iters}", payload, count=1)
    payload = re.sub(r"(?m)^(    lr:)\s*.*$", rf"\g<1> {lr}", payload, count=1)
    beta_block = "    optimize: " + ("true" if beta_enabled else "false") + f"\n    iters: {beta_iters if beta_enabled else 0}"
    payload = re.sub(r"(?m)^    optimize:\s*.*$\n^    iters:\s*.*$", beta_block, payload, count=1)
    return payload


def _write_configs() -> None:
    baseline = BASELINE_CONFIG.read_text(encoding="utf-8")
    destination = SEARCH_ROOT / "configs" / "stage1"
    destination.mkdir(parents=True, exist_ok=True)
    for path in destination.glob("*.yaml"):
        path.unlink()
    baseline_values = (120, 0.03, 48, True, 120)
    candidates = [
        candidate
        for candidate in product((60, 120, 180), (0.01, 0.03, 0.06), (24, 48, 72), ((False, 0), (True, 60), (True, 120)))
        for candidate in [(candidate[0], candidate[1], candidate[2], candidate[3][0], candidate[3][1])]
        if candidate != baseline_values
    ]
    random.Random(SEED).shuffle(candidates)
    values = [("s01_baseline", *baseline_values)] + [("", *candidate) for candidate in candidates[:47]]
    index = []
    submitters = SEARCH_ROOT / "submitters" / "stage1"
    submitters.mkdir(parents=True, exist_ok=True)
    for path in submitters.glob("*.sh"):
        path.unlink()
    for number, (name, iters, lr, overlap, beta_enabled, beta_iters) in enumerate(values, start=1):
        config_id = name or f"s{number:02d}"
        path = destination / f"{config_id}.yaml"
        path.write_text(_config_payload(baseline, iters=iters, lr=lr, overlap=overlap, beta_enabled=beta_enabled, beta_iters=beta_iters), encoding="utf-8")
        submitter = submitters / f"submit_{config_id}.sh"
        submitter.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "SEARCH_ROOT=$(cd \"$(dirname \"${BASH_SOURCE[0]}\")/../..\" && pwd)\n"
            f"exec env STAGE=stage1 CONFIG_ID={config_id} \"${{SEARCH_ROOT}}/submit_config.sh\" \"$@\"\n",
            encoding="utf-8",
        )
        submitter.chmod(0o755)
        index.append({"config_id": config_id, "config_path": str(path.relative_to(SEARCH_ROOT)), "pose_iters": iters, "pose_lr": lr, "overlap": overlap, "beta_optimize": beta_enabled, "beta_iters": beta_iters if beta_enabled else 0})
    (destination / "config_index.json").write_text(json.dumps({"seed": SEED, "configs": index}, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    clips = _load_clips()
    _validate_assets(clips)
    test = [clip for clip in clips if _parts(clip)[2] == TEST_PERSON]
    development = [clip for clip in clips if _parts(clip)[2] != TEST_PERSON]
    stage1 = _stage1_subset(development)
    if (len(test), len(development), len(stage1)) != (24, 96, 18):
        raise AssertionError("Unexpected split sizes.")
    _write_split("stage1", stage1, "18-video development screening subset; editable after generation.")
    _write_split("stage2", development, "All development clips excluding locked participant Niti.")
    _write_split("test", test, "Locked final test set: all Niti clips. Do not tune on these clips.")
    _write_configs()
    print("Created 3 split manifests and 48 stage-1 HMP configurations.")


if __name__ == "__main__":
    main()
