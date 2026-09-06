import { useCallback, useEffect, useState } from "react";

const STORAGE_KEY = "framerai.settings";

// Mirrors the bounds the backend validates against.
export const SETTING_FIELDS = [
  {
    key: "temperature",
    label: "Temperature",
    min: 0.1,
    max: 2,
    step: 0.05,
    help: "Higher is more varied, lower is more predictable.",
  },
  {
    key: "top_p",
    label: "Top-p",
    min: 0.1,
    max: 1,
    step: 0.05,
    help: "Sample from the smallest set of tokens with this combined probability.",
  },
  {
    key: "top_k",
    label: "Top-k",
    min: 0,
    max: 200,
    step: 1,
    help: "Consider only this many candidates per token. 0 turns the cutoff off.",
  },
  {
    key: "max_new_tokens",
    label: "Max new tokens",
    min: 16,
    max: 2048,
    step: 16,
    help: "Upper bound on the length of a response.",
  },
  {
    key: "num_frames",
    label: "Video frames",
    min: 1,
    max: 64,
    step: 1,
    help: "Frames generated for a video request.",
  },
];

// Square-only sizes, kept for compatibility with older saved settings.
export const RESOLUTIONS = [64, 128, 256, 512];

// Aspect ratios the backend accepts, with the size they resolve to at each tier.
export const ASPECT_RATIOS = [
  { value: "1:1", label: "Square" },
  { value: "4:3", label: "Standard" },
  { value: "3:4", label: "Standard tall" },
  { value: "3:2", label: "Landscape" },
  { value: "2:3", label: "Portrait" },
  { value: "16:9", label: "Widescreen" },
  { value: "9:16", label: "Vertical" },
  { value: "21:9", label: "Ultrawide" },
];

export const SIZE_TIERS = [256, 512, 768, 1024];

const MULTIPLE = 16;

/** Mirrors model/utils/image_sizing.py so the panel can preview the real size. */
export function resolveSize(aspect, tier) {
  const entry = ASPECT_RATIOS.find((r) => r.value === aspect) || ASPECT_RATIOS[0];
  const [w, h] = entry.value.split(":").map(Number);
  const ratio = w / h;
  const area = tier * tier;
  const round = (value) => Math.max(MULTIPLE, Math.round(value / MULTIPLE) * MULTIPLE);
  return { width: round(Math.sqrt(area * ratio)), height: round(Math.sqrt(area / ratio)) };
}

export const DEFAULT_SETTINGS = {
  temperature: 0.7,
  top_p: 0.9,
  top_k: 50,
  max_new_tokens: 512,
  aspect: "1:1",
  tier: 512,
  num_frames: 16,
};

function load() {
  try {
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    if (stored.sizeTier !== undefined && stored.tier === undefined) {
      stored.tier = stored.sizeTier;
    }
    delete stored.sizeTier;
    // Merge over the defaults so a partial or outdated payload still works.
    return { ...DEFAULT_SETTINGS, ...stored };
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

export function useSettings() {
  const [settings, setSettings] = useState(load);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    } catch {
      // Storage can be unavailable (private mode, quota). Settings still
      // apply for this session.
    }
  }, [settings]);

  const updateSetting = useCallback((key, value) => {
    setSettings((prev) => ({ ...prev, [key]: value }));
  }, []);

  const resetSettings = useCallback(() => setSettings({ ...DEFAULT_SETTINGS }), []);

  return { settings, updateSetting, resetSettings };
}
