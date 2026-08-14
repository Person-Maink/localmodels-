import type { ControlSpec } from "../types";

type ModeControlsProps = {
  controls: ControlSpec[];
  values: Record<string, unknown>;
  onChange: (id: string, value: unknown) => void;
};

export function ModeControls({ controls, values, onChange }: ModeControlsProps) {
  if (controls.length === 0) {
    return <div className="empty-note">No extra controls for this mode.</div>;
  }

  return (
    <div className="mode-controls">
      {controls.map((control) => {
        const value = values[control.id] ?? control.default ?? "";

        if (control.type === "boolean") {
          return (
            <label className="control-card control-inline" key={control.id}>
              <span className="control-label">{control.label}</span>
              <input
                type="checkbox"
                checked={Boolean(value)}
                onChange={(event) => onChange(control.id, event.target.checked)}
              />
            </label>
          );
        }

        if (control.type === "select") {
          return (
            <label className="control-card" key={control.id}>
              <span className="control-label">{control.label}</span>
              <select
                value={String(value)}
                onChange={(event) => onChange(control.id, event.target.value)}
              >
                {(control.options ?? []).map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
              {control.help ? <span className="control-help">{control.help}</span> : null}
            </label>
          );
        }

        return (
          <label className="control-card" key={control.id}>
            <span className="control-label">{control.label}</span>
            <input
              type={control.type === "number" ? "number" : "text"}
              min={control.min}
              max={control.max}
              step={control.step}
              placeholder={control.placeholder}
              value={String(value)}
              onChange={(event) =>
                onChange(
                  control.id,
                  control.type === "number" ? Number(event.target.value) : event.target.value
                )
              }
            />
            {control.help ? <span className="control-help">{control.help}</span> : null}
          </label>
        );
      })}
    </div>
  );
}
