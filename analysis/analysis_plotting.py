from __future__ import annotations

from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np


MetricStyleResolver = Callable[[int, dict], dict]


def build_time_psd_metrics_figure(
    entries: list[dict],
    fps: float,
    title_time: str,
    title_psd: str,
    figsize_inches=(12, 10),
    dpi: int = 100,
    style_resolver: MetricStyleResolver | None = None,
) -> plt.Figure:
    fig, axes = plt.subplots(3, 1, figsize=figsize_inches, dpi=dpi)
    for index, entry in enumerate(entries):
        result = entry["result"]
        style = _resolve_style(index, entry, style_resolver)
        color = style["color"]
        linestyle = style["linestyle"]
        linewidth = style.get("linewidth", 1.5)
        label = entry.get("plot_label") or f"{entry['label']} ({result['dominant']:.2f} Hz)"
        t = np.arange(len(result["magnitude"]), dtype=np.float32) / float(fps)

        axes[0].plot(
            t,
            result["magnitude"],
            color=color,
            linestyle=linestyle,
            lw=linewidth,
            label=label,
        )
        axes[1].semilogy(
            result["freqs"],
            result["psd"],
            color=color,
            linestyle=linestyle,
            lw=linewidth,
            label=label,
        )
        axes[1].axvline(result["dominant"], color=color, ls=":", alpha=0.45)

    axes[0].set_title(title_time)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Displacement magnitude")
    axes[0].grid(True)
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize="small")

    axes[1].set_title(title_psd)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Power spectral density")
    axes[1].grid(True)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize="small")

    add_metrics_table(axes[2], entries)

    fig.tight_layout()
    return fig


def add_metrics_table(ax, entries: Iterable[dict], title: str = "Metrics summary") -> None:
    ax.axis("off")
    columns = [
        "Label",
        "Dominant Hz",
        "Peak Ratio",
        "Peak Sharpness",
        "Temporal Noise",
        "Spatial Coherence",
        "RMS",
        "Samples",
    ]
    rows = []
    for entry in entries:
        result = entry["result"]
        rows.append(
            [
                entry["label"],
                _format_float(result.get("dominant")),
                _format_float(result.get("peak_ratio"), scientific=True),
                _format_float(result.get("peak_sharpness"), scientific=True),
                _format_float(result.get("temporal_noise"), scientific=True),
                _format_float(result.get("spatial_coherence"), scientific=True, allow_none=True),
                _format_float(result.get("rms"), scientific=True),
                str(int(len(result.get("magnitude", [])))),
            ]
        )

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.3)
    ax.set_title(title)


def _resolve_style(index: int, entry: dict, style_resolver: MetricStyleResolver | None) -> dict:
    if style_resolver is None:
        return {"color": f"C{index % 10}", "linestyle": "-", "linewidth": 1.5}
    resolved = dict(style_resolver(index, entry))
    resolved.setdefault("color", f"C{index % 10}")
    resolved.setdefault("linestyle", "-")
    resolved.setdefault("linewidth", 1.5)
    return resolved


def _format_float(value, scientific: bool = False, allow_none: bool = False) -> str:
    if value is None:
        return "-" if allow_none else "0.000"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(number):
        return "-"
    if scientific:
        return f"{number:.3e}"
    return f"{number:.3f}"
