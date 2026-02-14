import type { TickerLookup } from "./types";

type RankedTicker = TickerLookup & {
  _score: number;
  _isUs: boolean;
};

const US_EXCHANGES = new Set([
  "NASDAQ",
  "NYSE",
  "NYSEARCA",
  "AMEX",
  "BATS",
  "IEX",
]);

function normalizeText(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "");
}

function normalizeTicker(value: string) {
  return value.trim().toUpperCase().replace(/\./g, "-");
}

function isTickerLike(value: string) {
  return /^[A-Z][A-Z0-9-]{0,9}$/.test(value);
}

function scoreCandidate(query: string, item: TickerLookup, isUs: boolean) {
  const cleanQuery = query.trim();
  const queryNorm = normalizeText(cleanQuery);
  const symbol = normalizeTicker(item.ticker);
  const symbolNorm = normalizeText(symbol);
  const name = item.name.trim();
  const nameNorm = normalizeText(name);

  let score = 0;
  if (cleanQuery.toUpperCase() === symbol) score = 100;
  else if (symbol.startsWith(cleanQuery.toUpperCase())) score = 92;
  else if (symbolNorm === queryNorm) score = 88;
  else if (symbolNorm.startsWith(queryNorm)) score = 84;
  else if (nameNorm === queryNorm) score = 82;
  else if (nameNorm.startsWith(queryNorm)) score = 76;
  else if (nameNorm.includes(queryNorm)) score = 68;

  if (isUs) score += 4;
  return score;
}

function normalizeLookup(
  query: string,
  ticker: string,
  name: string,
  exchange?: string | null,
  instrumentType?: string | null,
  country?: string | null,
) {
  const symbol = normalizeTicker(ticker);
  if (!isTickerLike(symbol)) return null;

  const normalizedExchange = exchange?.trim() || null;
  const isUs = (country || "").toLowerCase() === "united states" || US_EXCHANGES.has((normalizedExchange || "").toUpperCase());
  const score = scoreCandidate(
    query,
    {
      ticker: symbol,
      name: name.trim() || symbol,
      exchange: normalizedExchange,
      instrument_type: instrumentType?.trim() || null,
    },
    isUs,
  );

  if (score <= 0) return null;

  return {
    ticker: symbol,
    name: name.trim() || symbol,
    exchange: normalizedExchange,
    instrument_type: instrumentType?.trim() || null,
    _score: score,
    _isUs: isUs,
  } satisfies RankedTicker;
}

async function searchTwelveData(query: string, limit: number): Promise<RankedTicker[]> {
  const url = new URL("https://api.twelvedata.com/symbol_search");
  url.searchParams.set("symbol", query);
  url.searchParams.set("outputsize", String(Math.min(Math.max(limit * 4, 12), 60)));
  if (import.meta.env.VITE_TWELVEDATA_API_KEY) {
    url.searchParams.set("apikey", import.meta.env.VITE_TWELVEDATA_API_KEY);
  }

  const response = await fetch(url.toString());
  if (!response.ok) return [];

  const payload = (await response.json()) as {
    data?: Array<{
      symbol?: string;
      instrument_name?: string;
      exchange?: string;
      instrument_type?: string;
      country?: string;
    }>;
  };

  if (!Array.isArray(payload.data)) return [];

  const out: RankedTicker[] = [];
  for (const row of payload.data) {
    if (!row?.symbol) continue;
    const normalized = normalizeLookup(
      query,
      row.symbol,
      row.instrument_name || row.symbol,
      row.exchange,
      row.instrument_type,
      row.country,
    );
    if (normalized) out.push(normalized);
  }

  return out;
}

async function searchFinnhub(query: string): Promise<RankedTicker[]> {
  const token = import.meta.env.VITE_FINNHUB_API_KEY;
  if (!token) return [];

  const url = new URL("https://finnhub.io/api/v1/search");
  url.searchParams.set("q", query);
  url.searchParams.set("token", token);

  const response = await fetch(url.toString());
  if (!response.ok) return [];

  const payload = (await response.json()) as {
    result?: Array<{
      symbol?: string;
      description?: string;
      exchange?: string;
      type?: string;
    }>;
  };

  if (!Array.isArray(payload.result)) return [];

  const out: RankedTicker[] = [];
  for (const row of payload.result) {
    if (!row?.symbol) continue;
    const normalized = normalizeLookup(
      query,
      row.symbol,
      row.description || row.symbol,
      row.exchange,
      row.type,
      null,
    );
    if (normalized) out.push(normalized);
  }

  return out;
}

export async function searchTickerDirectory(query: string, limit = 8): Promise<TickerLookup[]> {
  const cleanQuery = query.trim();
  if (!cleanQuery) return [];

  const [twelveDataResult, finnhubResult] = await Promise.allSettled([
    searchTwelveData(cleanQuery, limit),
    searchFinnhub(cleanQuery),
  ]);

  const merged = new Map<string, RankedTicker>();

  const addAll = (items: RankedTicker[]) => {
    for (const item of items) {
      const key = item.ticker;
      const existing = merged.get(key);
      if (!existing || item._score > existing._score) {
        merged.set(key, item);
      }
    }
  };

  if (twelveDataResult.status === "fulfilled") addAll(twelveDataResult.value);
  if (finnhubResult.status === "fulfilled") addAll(finnhubResult.value);

  return Array.from(merged.values())
    .sort((a, b) => {
      if (b._score !== a._score) return b._score - a._score;
      if (a._isUs !== b._isUs) return a._isUs ? -1 : 1;
      return a.ticker.localeCompare(b.ticker);
    })
    .slice(0, limit)
    .map(({ ticker, name, exchange, instrument_type }) => ({
      ticker,
      name,
      exchange,
      instrument_type,
    }));
}
