import { useMemo, useState } from "react";

import type { SourceAsset, TreeViewPayload } from "../types";

type SidebarProps = {
  tree: TreeViewPayload | null;
  organizationMode: string;
  search: string;
  selectedSourceIds: Set<string>;
  activeTab: "visualization" | "analysis";
  onOrganizationModeChange: (mode: string) => void;
  onSearchChange: (value: string) => void;
  onToggleSource: (sourceId: string) => void;
  onClearSelection: () => void;
};

function matchesSearch(source: SourceAsset, search: string) {
  const value = search.trim().toLowerCase();
  if (!value) {
    return true;
  }
  return [source.id, source.family, source.clip_id, source.family_id, source.experiment ?? "", source.video_id ?? ""]
    .join(" ")
    .toLowerCase()
    .includes(value);
}

function capabilityLabels(source: SourceAsset) {
  const labels = [source.family];
  if (source.video_id) {
    labels.push("video");
  }
  if (source.capabilities.camera) {
    labels.push("camera");
  }
  if (source.capabilities.bounding_boxes) {
    labels.push("bbox");
  }
  if (source.capabilities.beta) {
    labels.push("beta");
  }
  if (source.experiment) {
    labels.push(source.experiment);
  }
  return labels;
}

function SourceLeaf({
  source,
  selected,
  disabled,
  activeTab,
  onToggle,
}: {
  source: SourceAsset;
  selected: boolean;
  disabled: boolean;
  activeTab: "visualization" | "analysis";
  onToggle: (sourceId: string) => void;
}) {
  const disabledReason = disabled && activeTab === "analysis" ? "No frequency analysis" : "";

  return (
    <label className={`source-leaf ${selected ? "selected" : ""} ${disabled ? "disabled" : ""}`}>
      <input
        type="checkbox"
        checked={selected}
        disabled={disabled}
        onChange={() => onToggle(source.id)}
      />
      <div className="source-leaf-body">
        <div className="source-leaf-head">
          <strong>{source.clip_id}</strong>
          {disabledReason ? <span className="source-disabled-reason">{disabledReason}</span> : null}
        </div>
        <div className="source-leaf-subtitle">{source.id}</div>
        <div className="source-chip-row">
          {capabilityLabels(source).map((label) => (
            <span className="source-chip" key={`${source.id}:${label}`}>
              {label}
            </span>
          ))}
        </div>
      </div>
    </label>
  );
}

function SectionToggle({
  title,
  meta,
  open,
  onToggle,
}: {
  title: string;
  meta: string;
  open: boolean;
  onToggle: () => void;
}) {
  return (
    <button className="tree-toggle" type="button" onClick={onToggle}>
      <div>
        <strong>{title}</strong>
        <span>{meta}</span>
      </div>
      <span>{open ? "Hide" : "Show"}</span>
    </button>
  );
}

export function Sidebar({
  tree,
  organizationMode,
  search,
  selectedSourceIds,
  activeTab,
  onOrganizationModeChange,
  onSearchChange,
  onToggleSource,
  onClearSelection,
}: SidebarProps) {
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({});
  const [expandedVariants, setExpandedVariants] = useState<Record<string, boolean>>({});

  if (!tree) {
    return <aside className="sidebar">Loading library…</aside>;
  }

  const searchActive = search.trim().length > 0;
  const selectedCount = selectedSourceIds.size;

  const visibleSourceIds = useMemo(() => {
    return new Set(
      Object.values(tree.sources)
        .filter((source) => matchesSearch(source, search))
        .map((source) => source.id),
    );
  }, [search, tree.sources]);

  const renderSource = (sourceId: string) => {
    if (!visibleSourceIds.has(sourceId)) {
      return null;
    }
    const source = tree.sources[sourceId];
    if (!source) {
      return null;
    }
    const disabled = activeTab === "analysis" && !source.capabilities.frequency;
    return (
      <SourceLeaf
        key={source.id}
        source={source}
        selected={selectedSourceIds.has(source.id)}
        disabled={disabled}
        activeTab={activeTab}
        onToggle={onToggleSource}
      />
    );
  };

  const isExpanded = (bucket: Record<string, boolean>, key: string, fallback: boolean) =>
    bucket[key] ?? fallback;

  const toggleSection = (key: string) =>
    setExpandedSections((current) => ({ ...current, [key]: !(current[key] ?? false) }));

  const toggleVariant = (key: string) =>
    setExpandedVariants((current) => ({ ...current, [key]: !(current[key] ?? false) }));

  const activeTree =
    organizationMode === "by_source"
      ? tree.views.by_source
      : organizationMode === "by_experiment"
        ? tree.views.by_experiment
        : tree.views.by_clip;

  return (
    <aside className="sidebar">
      <div className="sidebar-sticky">
        <div className="sidebar-header">
          <h1>Tremor Lab</h1>
          <p>Browse clips, source families, and experiments without flooding the main view.</p>
        </div>

        <div className="sidebar-controls">
          <label>
            <span>Organization</span>
            <select
              value={organizationMode}
              onChange={(event) => onOrganizationModeChange(event.target.value)}
            >
              <option value="by_clip">By Clip</option>
              <option value="by_source">By Source</option>
              <option value="by_experiment">By Experiment</option>
            </select>
          </label>
          <label>
            <span>Search</span>
            <input
              type="text"
              value={search}
              placeholder="Filter clips, variants, experiments"
              onChange={(event) => onSearchChange(event.target.value)}
            />
          </label>
        </div>

        <div className="sidebar-summary">
          <span>{Object.keys(tree.sources).length} primary sources</span>
          <span>{selectedCount} selected</span>
          <button type="button" onClick={onClearSelection} disabled={selectedCount === 0}>
            Clear
          </button>
        </div>
      </div>

      <div className="tree-list">
        {organizationMode === "by_clip" &&
          activeTree.map((family: any) => {
            const hasSelected = family.variants.some((variant: any) =>
              variant.source_ids.some((sourceId: string) => selectedSourceIds.has(sourceId)),
            );
            const visibleVariants = family.variants.filter((variant: any) =>
              variant.source_ids.some((sourceId: string) => visibleSourceIds.has(sourceId)),
            );
            if (visibleVariants.length === 0) {
              return null;
            }
            const sectionKey = `clip:${family.id}`;
            const open = isExpanded(expandedSections, sectionKey, searchActive || hasSelected);
            return (
              <section className="tree-section" key={family.id}>
                <SectionToggle
                  title={family.label}
                  meta={`${visibleVariants.length} variants`}
                  open={open}
                  onToggle={() => toggleSection(sectionKey)}
                />
                {open &&
                  visibleVariants.map((variant: any) => {
                    const variantKey = `${sectionKey}:${variant.id}`;
                    const variantSelected = variant.source_ids.some((sourceId: string) =>
                      selectedSourceIds.has(sourceId),
                    );
                    const leaves = variant.source_ids.map(renderSource).filter(Boolean);
                    if (leaves.length === 0) {
                      return null;
                    }
                    const variantOpen = isExpanded(expandedVariants, variantKey, searchActive || variantSelected);
                    return (
                      <div className="tree-subsection" key={variant.id}>
                        <SectionToggle
                          title={variant.label}
                          meta={`${variant.source_ids.length} sources${variant.video_id ? " · raw video" : ""}`}
                          open={variantOpen}
                          onToggle={() => toggleVariant(variantKey)}
                        />
                        {variantOpen ? <div className="leaf-stack">{leaves}</div> : null}
                      </div>
                    );
                  })}
              </section>
            );
          })}

        {organizationMode === "by_source" &&
          activeTree.map((family: any) => {
            const hasSelected = family.experiments.some((group: any) =>
              group.source_ids.some((sourceId: string) => selectedSourceIds.has(sourceId)),
            );
            const groups = family.experiments.filter((group: any) =>
              group.source_ids.some((sourceId: string) => visibleSourceIds.has(sourceId)),
            );
            if (groups.length === 0) {
              return null;
            }
            const sectionKey = `source:${family.id}`;
            const open = isExpanded(expandedSections, sectionKey, searchActive || hasSelected);
            return (
              <section className="tree-section" key={family.id}>
                <SectionToggle
                  title={family.label}
                  meta={`${groups.length} groups`}
                  open={open}
                  onToggle={() => toggleSection(sectionKey)}
                />
                {open &&
                  groups.map((group: any) => {
                    const groupKey = `${sectionKey}:${group.id}`;
                    const groupSelected = group.source_ids.some((sourceId: string) =>
                      selectedSourceIds.has(sourceId),
                    );
                    const groupOpen = isExpanded(expandedVariants, groupKey, searchActive || groupSelected);
                    return (
                      <div className="tree-subsection" key={group.id}>
                        <SectionToggle
                          title={group.label}
                          meta={`${group.source_ids.length} clips`}
                          open={groupOpen}
                          onToggle={() => toggleVariant(groupKey)}
                        />
                        {groupOpen ? (
                          <div className="leaf-stack">{group.source_ids.map(renderSource).filter(Boolean)}</div>
                        ) : null}
                      </div>
                    );
                  })}
              </section>
            );
          })}

        {organizationMode === "by_experiment" &&
          activeTree.map((group: any) => {
            const visibleGroupSources = group.source_ids.filter((sourceId: string) =>
              visibleSourceIds.has(sourceId),
            );
            if (visibleGroupSources.length === 0) {
              return null;
            }
            const sectionKey = `experiment:${group.id}`;
            const hasSelected = group.source_ids.some((sourceId: string) => selectedSourceIds.has(sourceId));
            const open = isExpanded(expandedSections, sectionKey, searchActive || hasSelected);
            return (
              <section className="tree-section" key={group.id}>
                <SectionToggle
                  title={group.label}
                  meta={`${visibleGroupSources.length} sources`}
                  open={open}
                  onToggle={() => toggleSection(sectionKey)}
                />
                {open ? <div className="leaf-stack">{visibleGroupSources.map(renderSource).filter(Boolean)}</div> : null}
              </section>
            );
          })}
      </div>
    </aside>
  );
}
