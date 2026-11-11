/*
  Test-set results of the deployed setup (ResNet50 + colour/texture model voting),
  from Food_Sure_Vote_Test.ipynb, November 2026. "type" = pure or the right filler,
  "exact" = right filler and right level.
*/
export const SPICES = [
  {
    id: "black_pepper_powder",
    name: "Black pepper",
    adulterants: ["charcoal", "millet"],
    test: { photos: 56, type: 54, exact: 49 },
    thumb: "samples/black_pepper_powder__15_pct_millet_thumb.jpg",
  },
  {
    id: "red_chilli_powder",
    name: "Red chilli",
    adulterants: ["sawdust", "semolina"],
    test: { photos: 56, type: 45, exact: 34 },
    thumb: "samples/red_chilli_powder__10_pct_sawdust_thumb.jpg",
  },
  {
    id: "turmeric",
    name: "Turmeric",
    adulterants: ["rice flour", "semolina"],
    test: { photos: 57, type: 55, exact: 47 },
    thumb: "samples/turmeric__5_pct_semolina_thumb.jpg",
  },
];

export const SEVERITY_META = {
  PURE: { label: "Pure", level: "none", status: "good" },
  LOW: { label: "Adulterated", level: "low", status: "warning" },
  MEDIUM: { label: "Adulterated", level: "medium", status: "serious" },
  HIGH: { label: "Adulterated", level: "high", status: "critical" },
  UNKNOWN: { label: "Unclear", level: "unknown", status: "unknown" },
};

export const LEVEL_BANDS = [
  { severity: "LOW", name: "Low", range: "under 7.5%" },
  { severity: "MEDIUM", name: "Medium", range: "7.5–12.5%" },
  { severity: "HIGH", name: "High", range: "above 12.5%" },
];

export function pct(n, d) {
  return `${((n / d) * 100).toFixed(1)}%`;
}

export function nice(text) {
  return String(text ?? "").replace(/_/g, " ");
}

/** '15_pct_sawdust' -> "15% sawdust", 'pure' -> "pure" */
export function formatClassLabel(label) {
  const name = String(label).trim().toLowerCase();
  const match = name.match(/^(\d+)_pct_(.+)$/);
  if (match) return `${match[1]}% ${nice(match[2])}`;
  return nice(name);
}

/** '15_pct_sawdust' -> "sawdust", 'pure' -> "pure" */
export function typeOfClass(label) {
  const match = String(label).match(/^\d+_pct_(.+)$/);
  return match ? match[1] : String(label);
}

export function spiceName(id) {
  return SPICES.find((s) => s.id === id)?.name || id;
}

/** The band an adulterated result falls in, e.g. "7.5–12.5%". */
export function levelText(result) {
  if (result.severity === "PURE") return "none";
  if (result.concentration_range) return result.concentration_range.replace("-", "–");
  if (result.concentration_pct || result.concentration_pct === 0) return `about ${result.concentration_pct}%`;
  return "unknown";
}

export function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
