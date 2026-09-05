import { useEffect, useRef } from "react";
import { X, RotateCcw } from "lucide-react";
import { SETTING_FIELDS, ASPECT_RATIOS, SIZE_TIERS, resolveSize } from "../../hooks/useSettings";

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

  const resolved = resolveSize(settings.aspect, settings.tier ?? settings.sizeTier);

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
            <label htmlFor="setting-aspect">
              Aspect ratio
              <output htmlFor="setting-aspect">{resolved.width} x {resolved.height}</output>
            </label>
            <select
              id="setting-aspect"
              value={settings.aspect}
              onChange={(e) => onChange("aspect", e.target.value)}
              aria-describedby="setting-aspect-help"
            >
              {ASPECT_RATIOS.map(({ value, label }) => (
                <option key={value} value={value}>
                  {label} ({value})
                </option>
              ))}
            </select>
            <p className="setting-help" id="setting-aspect-help">
              A prompt asking for a shape, such as &quot;make it 16:9&quot; or
              &quot;portrait&quot;, overrides this.
            </p>
          </div>

          <div className="setting">
            <label htmlFor="setting-size-tier">Image size</label>
            <select
              id="setting-size-tier"
              value={settings.tier ?? settings.sizeTier}
              onChange={(e) => onChange("tier", Number(e.target.value))}
              aria-describedby="setting-size-tier-help"
            >
              {SIZE_TIERS.map((tier) => (
                <option key={tier} value={tier}>
                  {tier}px
                </option>
              ))}
            </select>
            <p className="setting-help" id="setting-size-tier-help">
              Every aspect ratio holds roughly the same pixel count at a given size, so
              larger sizes take longer regardless of shape.
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
