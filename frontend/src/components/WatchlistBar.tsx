interface Props {
  watchlist: string[];
  activeTicker: string;
  onSelectTicker: (ticker: string) => void;
  onRemoveTicker: (ticker: string) => void;
}

export default function WatchlistBar({ watchlist, activeTicker, onSelectTicker, onRemoveTicker }: Props) {
  return (
    <section className="integration-row watchlist-row">
      {watchlist.map((symbol) => (
        <div className={`integration-pill watchlist-pill ${symbol === activeTicker ? "active" : ""}`} key={symbol}>
          <button type="button" className="watchlist-symbol" onClick={() => onSelectTicker(symbol)}>
            {symbol}
          </button>
          <button
            type="button"
            className="watchlist-remove"
            onClick={() => onRemoveTicker(symbol)}
            aria-label={`Remove ${symbol} from watchlist`}
            title={`Remove ${symbol}`}
            disabled={watchlist.length <= 1}
          >
            ×
          </button>
        </div>
      ))}
    </section>
  );
}
