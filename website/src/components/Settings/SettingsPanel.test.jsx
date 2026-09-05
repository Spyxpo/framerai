import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import SettingsPanel from "./SettingsPanel";
import {
  ASPECT_RATIOS,
  DEFAULT_SETTINGS,
  SETTING_FIELDS,
  SIZE_TIERS,
  resolveSize,
  useSettings,
} from "../../hooks/useSettings";
import { useChat } from "../../hooks/useChat";

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

afterEach(() => {
  vi.restoreAllMocks();
});

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
    expect(tier.value).toBe(String(DEFAULT_SETTINGS.tier));
  });

  it("previews the resolved pixel size for the chosen ratio", () => {
    renderPanel({ settings: { ...DEFAULT_SETTINGS, aspect: "16:9", tier: 512 } });
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
    expect(props.onChange).toHaveBeenCalledWith("tier", 1024);
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

  describe("regression: chat request serialization includes settings.tier and not sizeTier", () => {
    function SettingsAndChatHarness() {
      const { settings, updateSetting } = useSettings();
      const { sendMessage } = useChat(settings);

      return (
        <div>
          <SettingsPanel
            open={true}
            settings={settings}
            onChange={updateSetting}
            onClose={() => {}}
          />
          <button onClick={() => sendMessage("generate an image of an astronaut", "image")}>
            Send Message
          </button>
        </div>
      );
    }

    it("serializes settings.tier and not sizeTier over REST when user selects a non-default tier", async () => {
      localStorage.clear();

      let serializedPayload = null;
      vi.spyOn(global, "fetch").mockImplementation((url, options) => {
        if (options?.body && typeof options.body === "string") {
          try {
            const parsed = JSON.parse(options.body);
            if (parsed.settings) serializedPayload = parsed;
          } catch {
            // ignore
          }
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ id: "msg-1", role: "assistant", content: "reply", type: "text" }),
        });
      });

      const { container } = render(<SettingsAndChatHarness />);

      const tierSelect = container.querySelector("#setting-size-tier");
      expect(tierSelect.value).toBe(String(DEFAULT_SETTINGS.tier));

      // Select aspect ratio 16:9
      const aspectSelect = container.querySelector("#setting-aspect");
      await userEvent.selectOptions(aspectSelect, "16:9");
      expect(screen.getByText("688 x 384")).toBeInTheDocument();

      // User selects non-default size tier 1024
      await userEvent.selectOptions(tierSelect, "1024");
      expect(tierSelect.value).toBe("1024");

      // Verify dimensions preview reflects the selected ratio and tier
      expect(screen.getByText("1360 x 768")).toBeInTheDocument();

      // User sends the chat message
      await userEvent.click(screen.getByRole("button", { name: /send message/i }));

      await waitFor(() => {
        expect(serializedPayload).not.toBeNull();
      });

      // Regression assertions:
      expect(serializedPayload.settings.tier).toBe(1024);
      expect(serializedPayload.settings.sizeTier).toBeUndefined();
      expect(serializedPayload.settings).not.toHaveProperty("sizeTier");
      expect(serializedPayload.settings.aspect).toBe("16:9");
      expect(serializedPayload.settings.temperature).toBe(DEFAULT_SETTINGS.temperature);
    });

    it("serializes settings.tier and not sizeTier over WebSocket when user selects a non-default tier", async () => {
      localStorage.clear();

      let wsInstance = null;
      let serializedPayload = null;

      const MockWS = class {
        constructor(url) {
          this.url = url;
          this.readyState = 1; // WebSocket.OPEN
          wsInstance = this;
          setTimeout(() => {
            if (this.onopen) this.onopen();
          }, 0);
        }
        send(data) {
          try {
            const parsed = JSON.parse(data);
            if (parsed.settings) serializedPayload = parsed;
          } catch {
            // ignore
          }
        }
        close() {
          this.readyState = 3;
        }
      };
      MockWS.OPEN = 1;
      MockWS.CLOSED = 3;

      const origWS = global.WebSocket;
      global.WebSocket = MockWS;

      try {
        const { container } = render(<SettingsAndChatHarness />);

        await waitFor(() => {
          expect(wsInstance).not.toBeNull();
        });

        const tierSelect = container.querySelector("#setting-size-tier");
        expect(tierSelect.value).toBe(String(DEFAULT_SETTINGS.tier));

        // Select aspect ratio 16:9
        const aspectSelect = container.querySelector("#setting-aspect");
        await userEvent.selectOptions(aspectSelect, "16:9");
        expect(screen.getByText("688 x 384")).toBeInTheDocument();

        // User selects non-default tier 1024
        await userEvent.selectOptions(tierSelect, "1024");
        expect(tierSelect.value).toBe("1024");
        expect(screen.getByText("1360 x 768")).toBeInTheDocument();

        // Send message
        await userEvent.click(screen.getByRole("button", { name: /send message/i }));

        await waitFor(() => {
          expect(serializedPayload).not.toBeNull();
        });

        // Regression assertions:
        expect(serializedPayload.settings.tier).toBe(1024);
        expect(serializedPayload.settings.sizeTier).toBeUndefined();
        expect(serializedPayload.settings).not.toHaveProperty("sizeTier");
        expect(serializedPayload.settings.aspect).toBe("16:9");
        expect(serializedPayload.settings.temperature).toBe(DEFAULT_SETTINGS.temperature);
      } finally {
        global.WebSocket = origWS;
      }
    });
  });
});
