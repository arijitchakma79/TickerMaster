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

interface Props {
  points: CandlePoint[];
  mode: "candles" | "line";
  showSma: boolean;
  showEma: boolean;
}

function toTs(value: string): UTCTimestamp {
  return Math.floor(new Date(value).getTime() / 1000) as UTCTimestamp;
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

export default function StockChart({ points, mode, showSma, showEma }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const candles = useMemo(
    () =>
      points.map((point) => ({
        time: toTs(point.timestamp),
        open: point.open,
        high: point.high,
        low: point.low,
        close: point.close,
      })),
    [points]
  );

  const volume = useMemo(
    () =>
      points.map((point) => ({
        time: toTs(point.timestamp),
        value: point.volume,
        color: point.close >= point.open ? "rgba(16, 163, 127, 0.55)" : "rgba(212, 63, 76, 0.55)",
      })),
    [points]
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

    if (showSma) {
      const smaSeries = chart.addSeries(LineSeries, { color: "#63a8ff", lineWidth: 1 });
      smaSeries.setData(sma(points, 20));
    }
    if (showEma) {
      const emaSeries = chart.addSeries(LineSeries, { color: "#e5a24b", lineWidth: 1 });
      emaSeries.setData(ema(points, 21));
    }

    chart.timeScale().fitContent();
    return () => chart.remove();
  }, [candles, mode, points, showEma, showSma, volume]);

  return <div ref={containerRef} className="advanced-chart" />;
}
