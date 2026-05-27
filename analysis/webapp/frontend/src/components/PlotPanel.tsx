import Plot from "react-plotly.js";
import type { AnalysisResult } from "../types";

type PlotPanelProps = {
  result: AnalysisResult;
};

export function PlotPanel({ result }: PlotPanelProps) {
  const sweepEntries = result.entries.filter((entry) => Array.isArray(entry.series));
  const timeEntries = result.entries.filter((entry) => entry.plots);

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
          line: { color: `hsl(${(index * 47) % 360} 70% 38%)` }
        }))}
        layout={{ title: "Displacement Magnitude", paper_bgcolor: "transparent", plot_bgcolor: "white" }}
        style={{ width: "100%", height: "100%" }}
      />
      <Plot
        data={timeEntries.map((entry, index) => ({
          x: entry.plots?.freqs_hz ?? [],
          y: entry.plots?.psd ?? [],
          type: "scatter",
          mode: "lines",
          name: entry.label,
          line: { color: `hsl(${(index * 47) % 360} 70% 38%)` }
        }))}
        layout={{
          title: "Power Spectral Density",
          yaxis: { type: "log" },
          paper_bgcolor: "transparent",
          plot_bgcolor: "white"
        }}
        style={{ width: "100%", height: "100%" }}
      />
    </div>
  );
}
