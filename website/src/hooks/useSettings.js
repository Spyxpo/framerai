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

export const RESOLUTIONS = [64, 128, 256, 512];

export const DEFAULT_SETTINGS = {
  temperature: 0.7,
  top_p: 0.9,
  top_k: 50,
  max_new_tokens: 512,
  resolution: 256,
  num_frames: 16,
};

function load() {
  try {
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
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
