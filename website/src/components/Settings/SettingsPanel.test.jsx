import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import SettingsPanel from "./SettingsPanel";
import {
  ASPECT_RATIOS,
  DEFAULT_SETTINGS,
  SETTING_FIELDS,
  SIZE_TIERS,
  resolveSize,
} from "../../hooks/useSettings";

function renderPanel(overrides = {}) {
  const props = {
    open: true,
    settings: { ...DEFAULT_SETTINGS },
    model: "framer-tiny",
    onChange: vi.fn(),
    onReset: vi.fn(),
    onClose: vi.fn(),
    ...overrides,
  };
  return { ...render(<SettingsPanel {...props} />), props };
}

describe("SettingsPanel", () => {
  it("renders nothing when closed", () => {
    const { container } = renderPanel({ open: false });
    expect(container).toBeEmptyDOMElement();
  });

  it("renders a slider per setting field, each with its help text", () => {
    const { container } = renderPanel();
    const sliders = screen.getAllByRole("slider");
    expect(sliders).toHaveLength(SETTING_FIELDS.length);

    for (const { key } of SETTING_FIELDS) {
      const input = container.querySelector(`#setting-${key}`);
      expect(input).toBeInTheDocument();
      expect(input).toHaveAttribute("aria-describedby", `setting-${key}-help`);
      expect(container.querySelector(`#setting-${key}-help`)).toBeInTheDocument();
    }
  });

  it("offers every aspect ratio and size tier", () => {
    const { container } = renderPanel();
    const aspect = container.querySelector("#setting-aspect");
    expect(aspect.options).toHaveLength(ASPECT_RATIOS.length);
    expect(aspect.value).toBe(DEFAULT_SETTINGS.aspect);

    const tier = container.querySelector("#setting-size-tier");
    expect(tier.options).toHaveLength(SIZE_TIERS.length);
    expect(tier.value).toBe(String(DEFAULT_SETTINGS.sizeTier));
  });

  it("previews the resolved pixel size for the chosen ratio", () => {
    renderPanel({ settings: { ...DEFAULT_SETTINGS, aspect: "16:9", sizeTier: 512 } });
    expect(screen.getByText("688 x 384")).toBeInTheDocument();
  });

  it("resolves every ratio to multiple-of-16 dimensions", () => {
    for (const { value } of ASPECT_RATIOS) {
      for (const tier of SIZE_TIERS) {
        const { width, height } = resolveSize(value, tier);
        expect(width % 16).toBe(0);
        expect(height % 16).toBe(0);
      }
    }
  });

  it("reports the served model", () => {
    renderPanel({ model: "framer-1t-a32b" });
    expect(screen.getByText("framer-1t-a32b")).toBeInTheDocument();
  });

  it("emits the ratio as a string and the tier as a number", async () => {
    const { container, props } = renderPanel();
    await userEvent.selectOptions(container.querySelector("#setting-aspect"), "16:9");
    expect(props.onChange).toHaveBeenCalledWith("aspect", "16:9");

    await userEvent.selectOptions(container.querySelector("#setting-size-tier"), "1024");
    expect(props.onChange).toHaveBeenCalledWith("sizeTier", 1024);
  });

  it("closes on Escape", async () => {
    const { props } = renderPanel();
    await userEvent.keyboard("{Escape}");
    expect(props.onClose).toHaveBeenCalled();
  });

  it("is a labelled modal dialog", () => {
    renderPanel();
    const dialog = screen.getByRole("dialog");
    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(dialog).toHaveAccessibleName("Generation settings");
  });
});
