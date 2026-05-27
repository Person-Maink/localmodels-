import { useEffect, useMemo, useState } from "react";

import { Sidebar } from "./components/Sidebar";
import { api } from "./lib/api";
import { AnalysisPage } from "./pages/AnalysisPage";
import { VisualizationPage } from "./pages/VisualizationPage";
import type { ModeSpec, SourceAsset, TreeViewPayload } from "./types";

export default function App() {
  const [tree, setTree] = useState<TreeViewPayload | null>(null);
  const [visualizationModes, setVisualizationModes] = useState<ModeSpec[]>([]);
  const [analysisModes, setAnalysisModes] = useState<ModeSpec[]>([]);
  const [activeTab, setActiveTab] = useState<"visualization" | "analysis">("visualization");
  const [organizationMode, setOrganizationMode] = useState("by_clip");
  const [search, setSearch] = useState("");
  const [selectedSourceIds, setSelectedSourceIds] = useState<Set<string>>(new Set());

  useEffect(() => {
    api.getLibraryTree().then(setTree).catch((error) => console.error(error));
    api.getVisualizationModes().then(setVisualizationModes).catch((error) => console.error(error));
    api.getAnalysisModes().then(setAnalysisModes).catch((error) => console.error(error));
  }, []);

  const selectedSources = useMemo<SourceAsset[]>(() => {
    if (!tree) {
      return [];
    }
    return [...selectedSourceIds]
      .map((sourceId) => tree.sources[sourceId])
      .filter(Boolean);
  }, [selectedSourceIds, tree]);

  return (
    <div className="app-shell">
      <Sidebar
        tree={tree}
        organizationMode={organizationMode}
        search={search}
        selectedSourceIds={selectedSourceIds}
        activeTab={activeTab}
        onOrganizationModeChange={setOrganizationMode}
        onSearchChange={setSearch}
        onClearSelection={() => setSelectedSourceIds(new Set())}
        onToggleSource={(sourceId) =>
          setSelectedSourceIds((current) => {
            const next = new Set(current);
            if (next.has(sourceId)) {
              next.delete(sourceId);
            } else {
              next.add(sourceId);
            }
            return next;
          })
        }
      />

      <main className="main-shell">
        <div className="top-nav">
          <button
            className={activeTab === "visualization" ? "active" : ""}
            onClick={() => setActiveTab("visualization")}
          >
            Visualization
          </button>
          <button
            className={activeTab === "analysis" ? "active" : ""}
            onClick={() => setActiveTab("analysis")}
          >
            Frequency Analysis
          </button>
          <button
            className="rescan-button"
            onClick={() => api.rescanLibrary().then(() => api.getLibraryTree().then(setTree))}
          >
            Rescan Library
          </button>
        </div>

        {!tree ? (
          <div className="loading-shell">Loading project library…</div>
        ) : activeTab === "visualization" ? (
          <VisualizationPage tree={tree} modes={visualizationModes} selectedSources={selectedSources} />
        ) : (
          <AnalysisPage modes={analysisModes} selectedSources={selectedSources} />
        )}
      </main>
    </div>
  );
}
