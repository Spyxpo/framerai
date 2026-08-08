import { useEffect, useRef } from "react";
import { X, RotateCcw } from "lucide-react";
import { SETTING_FIELDS, RESOLUTIONS } from "../../hooks/useSettings";

export default function SettingsPanel({ open, settings, model, onChange, onReset, onClose }) {
  const panelRef = useRef(null);
  const closeRef = useRef(null);

  useEffect(() => {
    if (!open) return undefined;
    closeRef.current?.focus();

    const onKeyDown = (e) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="settings-overlay" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div
        className="settings-panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-title"
        ref={panelRef}
      >
        <div className="settings-header">
          <h2 id="settings-title">Generation settings</h2>
          <button className="icon-btn" onClick={onClose} aria-label="Close settings" ref={closeRef}>
            <X size={18} aria-hidden="true" />
          </button>
        </div>

        <div className="settings-body">
          <p className="settings-model">
            Model
            <span className="model-badge">{model || "unknown"}</span>
          </p>
          <p className="settings-note">
            The backend serves one checkpoint at a time. Point <code>MODEL_PATH</code> at a
            different export and restart it to change size.
          </p>

          {SETTING_FIELDS.map(({ key, label, min, max, step, help }) => (
            <div className="setting" key={key}>
              <label htmlFor={`setting-${key}`}>
                {label}
                <output htmlFor={`setting-${key}`}>{settings[key]}</output>
              </label>
              <input
                id={`setting-${key}`}
                type="range"
                min={min}
                max={max}
                step={step}
                value={settings[key]}
                onChange={(e) => onChange(key, Number(e.target.value))}
                aria-describedby={`setting-${key}-help`}
              />
              <p className="setting-help" id={`setting-${key}-help`}>{help}</p>
            </div>
          ))}

          <div className="setting">
            <label htmlFor="setting-resolution">Image resolution</label>
            <select
              id="setting-resolution"
              value={settings.resolution}
              onChange={(e) => onChange("resolution", Number(e.target.value))}
              aria-describedby="setting-resolution-help"
            >
              {RESOLUTIONS.map((size) => (
                <option key={size} value={size}>
                  {size} x {size}
                </option>
              ))}
            </select>
            <p className="setting-help" id="setting-resolution-help">
              Larger images take longer to generate.
            </p>
          </div>
        </div>

        <div className="settings-footer">
          <button className="settings-reset" onClick={onReset}>
            <RotateCcw size={14} aria-hidden="true" />
            <span>Reset to defaults</span>
          </button>
          <span className="settings-saved">Saved on this device</span>
        </div>
      </div>
    </div>
  );
}
