import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import SettingsPanel from "./SettingsPanel";
import { DEFAULT_SETTINGS, RESOLUTIONS, SETTING_FIELDS } from "../../hooks/useSettings";

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

  it("offers every supported resolution", () => {
    renderPanel();
    const select = screen.getByLabelText(/image resolution/i);
    expect(select.options).toHaveLength(RESOLUTIONS.length);
    expect(select.value).toBe(String(DEFAULT_SETTINGS.resolution));
  });

  it("reports the served model", () => {
    renderPanel({ model: "framer-1t-a32b" });
    expect(screen.getByText("framer-1t-a32b")).toBeInTheDocument();
  });

  it("emits a numeric value when the resolution changes", async () => {
    const { props } = renderPanel();
    await userEvent.selectOptions(screen.getByLabelText(/image resolution/i), "512");
    expect(props.onChange).toHaveBeenCalledWith("resolution", 512);
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
