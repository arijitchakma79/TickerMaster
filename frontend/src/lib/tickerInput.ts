import type { TickerLookup } from "./types";

export function normalizeTickerCandidate(value: string) {
  const trimmed = value.trim();
  if (!trimmed) return "";
  if (!/^[A-Za-z]{1,5}(?:[-.][A-Za-z]{1,2})?$/.test(trimmed)) return "";
  return trimmed.toUpperCase().replace(/\./g, "-");
}

export function resolveTickerCandidate(value: string, suggestions: TickerLookup[]) {
  const explicitTicker = normalizeTickerCandidate(value);
  if (explicitTicker) return explicitTicker;

  const normalized = value.trim().toLowerCase();
  if (!normalized) return "";

  const exactMatch = suggestions.find(
    (item) => item.ticker.toLowerCase() === normalized || item.name.toLowerCase() === normalized
  );
  if (exactMatch) return exactMatch.ticker.toUpperCase();

  return suggestions[0]?.ticker.toUpperCase() ?? "";
}
