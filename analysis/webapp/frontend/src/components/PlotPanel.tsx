import Plot from "react-plotly.js";
import type { AnalysisResult } from "../types";

type PlotPanelProps = {
  result: AnalysisResult;
};

export function PlotPanel({ result }: PlotPanelProps) {
  const sweepEntries = result.entries.filter((entry) => Array.isArray(entry.series));
  const timeEntries = result.entries.filter((entry) => entry.plots);
  const colorForIndex = (index: number) => `hsl(${(index * 47) % 360} 70% 38%)`;

  if (sweepEntries.length > 0) {
    return (
      <div className="result-grid">
        <Plot
          data={sweepEntries.map((entry, index) => ({
            x: entry.series?.map((row) => row.point_count) ?? [],
            y: entry.series?.map((row) => row.dominant_hz) ?? [],
            type: "scatter",
            mode: "lines+markers",
            name: entry.label,
            marker: { color: `hsl(${(index * 53) % 360} 75% 42%)` }
          }))}
          layout={{ title: "Dominant Frequency vs Region Size", paper_bgcolor: "transparent", plot_bgcolor: "white" }}
          style={{ width: "100%", height: "100%" }}
        />
        <Plot
          data={sweepEntries.map((entry, index) => ({
            x: entry.series?.map((row) => row.point_count) ?? [],
            y: entry.series?.map((row) => row.rms_amplitude) ?? [],
            type: "scatter",
            mode: "lines+markers",
            name: entry.label,
            marker: { color: `hsl(${(index * 53) % 360} 75% 42%)` }
          }))}
          layout={{ title: "RMS Amplitude vs Region Size", paper_bgcolor: "transparent", plot_bgcolor: "white" }}
          style={{ width: "100%", height: "100%" }}
        />
      </div>
    );
  }

  return (
    <div className="result-grid">
      <Plot
        data={timeEntries.map((entry, index) => ({
          x: entry.plots?.time_s ?? [],
          y: entry.plots?.magnitude ?? [],
          type: "scatter",
          mode: "lines",
          name: entry.label,
          line: { color: colorForIndex(index) }
        }))}
        layout={{ title: "Displacement Magnitude", paper_bgcolor: "transparent", plot_bgcolor: "white" }}
        style={{ width: "100%", height: "100%" }}
      />
      <Plot
        data={timeEntries.flatMap((entry, index) => {
          const color = colorForIndex(index);
          const traces: any[] = [
            {
              x: entry.plots?.freqs_hz ?? [],
              y: entry.plots?.psd ?? [],
              type: "scatter" as const,
              mode: "lines" as const,
              name: `${entry.label} Welch (${entry.dominant_hz?.toFixed(2) ?? "0.00"} Hz)`,
              line: { color, width: 2 }
            }
          ];

          if (entry.plots?.welch_peak_value != null && entry.plots?.welch_peak_hz != null) {
            traces.push({
              x: [entry.plots.welch_peak_hz],
              y: [entry.plots.welch_peak_value],
              type: "scatter" as const,
              mode: "markers" as const,
              name: `${entry.label} Welch peak`,
              showlegend: false,
              marker: { color, size: 8, symbol: "circle" }
            });
          }

          if ((entry.plots?.fft_freqs_hz?.length ?? 0) > 0 && (entry.plots?.fft_spectrum?.length ?? 0) > 0) {
            traces.push({
              x: entry.plots?.fft_freqs_hz ?? [],
              y: entry.plots?.fft_spectrum ?? [],
              type: "scatter" as const,
              mode: "lines" as const,
              name: `${entry.label} FFT (${entry.fft_peak_hz?.toFixed(2) ?? "0.00"} Hz)`,
              line: { color, dash: "dash", width: 1.5 }
            });
          }

          if (entry.plots?.fft_peak_value != null && entry.plots?.fft_peak_hz != null) {
            traces.push({
              x: [entry.plots.fft_peak_hz],
              y: [entry.plots.fft_peak_value],
              type: "scatter" as const,
              mode: "markers" as const,
              name: `${entry.label} FFT peak`,
              showlegend: false,
              marker: { color, size: 8, symbol: "x" }
            });
          }

          return traces;
        })}
        layout={{
          title: "Welch PSD and FFT Spectrum",
          yaxis: { type: "log" },
          paper_bgcolor: "transparent",
          plot_bgcolor: "white"
        }}
        style={{ width: "100%", height: "100%" }}
      />
    </div>
  );
}
