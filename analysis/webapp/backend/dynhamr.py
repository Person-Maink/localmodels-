from __future__ import annotations

from typing import List

from dynhamr_io import export_latest_dynhamr_runs

from .settings import AppSettings


def normalize_dynhamr_outputs(settings: AppSettings, overwrite: bool = False) -> List[dict]:
    if not settings.dynhamr_logs_root.exists():
        return []
    return export_latest_dynhamr_runs(
        logs_root=settings.dynhamr_logs_root,
        output_root=settings.dynhamr_output_root,
        overwrite=overwrite,
    )
