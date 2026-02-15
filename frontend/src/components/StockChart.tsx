import { useEffect, useMemo, useRef } from "react";
import {
  CandlestickSeries,
  ColorType,
  HistogramSeries,
  LineSeries,
  createChart,
  type LineData,
  type UTCTimestamp,
} from "lightweight-charts";
import type { CandlePoint } from "../lib/types";

export type ChartOverlayIndicator = "sma20" | "sma50" | "sma200" | "ema21" | "ema50" | "vwap" | "bbands";

interface Props {
  points: CandlePoint[];
  mode: "candles" | "line";
  indicators: Partial<Record<ChartOverlayIndicator, boolean>>;
}

function toTs(value: string): UTCTimestamp {
  return Math.floor(new Date(value).getTime() / 1000) as UTCTimestamp;
}

function normalizePoints(points: CandlePoint[]): CandlePoint[] {
  const sorted = [...points].sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
  const out: CandlePoint[] = [];
  let lastTs = -1;
  for (const point of sorted) {
    const ts = new Date(point.timestamp).getTime();
    if (!Number.isFinite(ts)) continue;
    const safeTs = ts <= lastTs ? lastTs + 1000 : ts;
    lastTs = safeTs;
    out.push({
      ...point,
      timestamp: new Date(safeTs).toISOString(),
    });
  }
  return out;
}

function sma(points: CandlePoint[], period: number): LineData<UTCTimestamp>[] {
  const out: LineData<UTCTimestamp>[] = [];
  for (let i = period - 1; i < points.length; i += 1) {
    const slice = points.slice(i - period + 1, i + 1);
    const avg = slice.reduce((acc, cur) => acc + cur.close, 0) / period;
    out.push({ time: toTs(points[i].timestamp), value: avg });
  }
  return out;
}

function ema(points: CandlePoint[], period: number): LineData<UTCTimestamp>[] {
  const out: LineData<UTCTimestamp>[] = [];
  if (points.length === 0) return out;
  const k = 2 / (period + 1);
  let prev = points[0].close;
  for (let i = 0; i < points.length; i += 1) {
    const close = points[i].close;
    const value = i === 0 ? close : (close * k) + (prev * (1 - k));
    prev = value;
    out.push({ time: toTs(points[i].timestamp), value });
  }
  return out;
}

function vwap(points: CandlePoint[]): LineData<UTCTimestamp>[] {
  const out: LineData<UTCTimestamp>[] = [];
  let cumulativeVolume = 0;
  let cumulativeTpv = 0;
  for (const point of points) {
    const volume = Math.max(0, point.volume ?? 0);
    const typicalPrice = (point.high + point.low + point.close) / 3;
    cumulativeVolume += volume;
    cumulativeTpv += typicalPrice * volume;
    if (cumulativeVolume <= 0) continue;
    out.push({ time: toTs(point.timestamp), value: cumulativeTpv / cumulativeVolume });
  }
  return out;
}

function bollinger(points: CandlePoint[], period = 20, stdDev = 2): {
  upper: LineData<UTCTimestamp>[];
  mid: LineData<UTCTimestamp>[];
  lower: LineData<UTCTimestamp>[];
} {
  const upper: LineData<UTCTimestamp>[] = [];
  const mid: LineData<UTCTimestamp>[] = [];
  const lower: LineData<UTCTimestamp>[] = [];
  if (points.length < period) return { upper, mid, lower };
  for (let i = period - 1; i < points.length; i += 1) {
    const slice = points.slice(i - period + 1, i + 1);
    const mean = slice.reduce((acc, cur) => acc + cur.close, 0) / period;
    const variance = slice.reduce((acc, cur) => acc + ((cur.close - mean) ** 2), 0) / period;
    const sigma = Math.sqrt(variance);
    const time = toTs(points[i].timestamp);
    mid.push({ time, value: mean });
    upper.push({ time, value: mean + (stdDev * sigma) });
    lower.push({ time, value: mean - (stdDev * sigma) });
  }
  return { upper, mid, lower };
}

export default function StockChart({ points, mode, indicators }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const safePoints = useMemo(() => normalizePoints(points), [points]);
  const candles = useMemo(
    () =>
      safePoints.map((point) => ({
        time: toTs(point.timestamp),
        open: point.open,
        high: point.high,
        low: point.low,
        close: point.close,
      })),
    [safePoints]
  );

  const volume = useMemo(
    () =>
      safePoints.map((point) => ({
        time: toTs(point.timestamp),
        value: point.volume,
        color: point.close >= point.open ? "rgba(16, 163, 127, 0.55)" : "rgba(212, 63, 76, 0.55)",
      })),
    [safePoints]
  );

  useEffect(() => {
    if (!containerRef.current) return;
    const chart = createChart(containerRef.current, {
      autoSize: true,
      height: 340,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#9fb0b5",
      },
      grid: {
        vertLines: { color: "rgba(130, 150, 150, 0.14)" },
        horzLines: { color: "rgba(130, 150, 150, 0.14)" },
      },
      rightPriceScale: {
        borderColor: "rgba(130, 150, 150, 0.18)",
      },
      timeScale: {
        borderColor: "rgba(130, 150, 150, 0.18)",
        timeVisible: true,
      },
      crosshair: {
        vertLine: { color: "rgba(16, 163, 127, 0.35)" },
        horzLine: { color: "rgba(16, 163, 127, 0.35)" },
      },
    });

    let mainSeries;
    if (mode === "candles") {
      mainSeries = chart.addSeries(CandlestickSeries, {
        upColor: "#10a37f",
        downColor: "#d43f4c",
        borderVisible: false,
        wickUpColor: "#10a37f",
        wickDownColor: "#d43f4c",
      });
      mainSeries.setData(candles);
    } else {
      mainSeries = chart.addSeries(LineSeries, {
        color: "#10a37f",
        lineWidth: 2,
      });
      mainSeries.setData(candles.map((row) => ({ time: row.time, value: row.close })));
    }

    const volumeSeries = chart.addSeries(HistogramSeries, {
      priceFormat: { type: "volume" },
      priceScaleId: "",
      base: 0,
    });
    volumeSeries.priceScale().applyOptions({
      scaleMargins: { top: 0.78, bottom: 0 },
    });
    volumeSeries.setData(volume);

    if (indicators.sma20) {
      const series = chart.addSeries(LineSeries, { color: "#63a8ff", lineWidth: 1 });
      series.setData(sma(safePoints, 20));
    }
    if (indicators.sma50) {
      const series = chart.addSeries(LineSeries, { color: "#4f84e6", lineWidth: 1 });
      series.setData(sma(safePoints, 50));
    }
    if (indicators.sma200) {
      const series = chart.addSeries(LineSeries, { color: "#2f5fae", lineWidth: 1 });
      series.setData(sma(safePoints, 200));
    }
    if (indicators.ema21) {
      const series = chart.addSeries(LineSeries, { color: "#e5a24b", lineWidth: 1 });
      series.setData(ema(safePoints, 21));
    }
    if (indicators.ema50) {
      const series = chart.addSeries(LineSeries, { color: "#d88726", lineWidth: 1 });
      series.setData(ema(safePoints, 50));
    }
    if (indicators.vwap) {
      const series = chart.addSeries(LineSeries, { color: "#14b8a6", lineWidth: 1 });
      series.setData(vwap(safePoints));
    }
    if (indicators.bbands) {
      const bands = bollinger(safePoints, 20, 2);
      const upper = chart.addSeries(LineSeries, { color: "rgba(163, 173, 186, 0.9)", lineWidth: 1 });
      const mid = chart.addSeries(LineSeries, { color: "rgba(140, 158, 176, 0.8)", lineWidth: 1 });
      const lower = chart.addSeries(LineSeries, { color: "rgba(163, 173, 186, 0.9)", lineWidth: 1 });
      upper.setData(bands.upper);
      mid.setData(bands.mid);
      lower.setData(bands.lower);
    }

    chart.timeScale().fitContent();
    return () => chart.remove();
  }, [candles, indicators, mode, safePoints, volume]);

  return <div ref={containerRef} className="advanced-chart" />;
}
