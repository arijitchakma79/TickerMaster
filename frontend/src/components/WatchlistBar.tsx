interface Props {
  watchlist: string[];
  activeTicker: string;
  onSelectTicker: (ticker: string) => void;
  onRemoveTicker: (ticker: string) => void;
  favorites: string[];
  onToggleFavorite: (ticker: string) => void;
}

export default function WatchlistBar({
  watchlist,
  activeTicker,
  onSelectTicker,
  onRemoveTicker,
  favorites,
  onToggleFavorite
}: Props) {
  return (
    <section className="integration-row watchlist-row">
      {watchlist.map((symbol) => (
        <div className={`integration-pill watchlist-pill ${symbol === activeTicker ? "active" : ""}`} key={symbol}>
          <button type="button" className="watchlist-symbol" onClick={() => onSelectTicker(symbol)}>
            {symbol}
          </button>
          <button
            type="button"
            className={`watchlist-favorite ${favorites.includes(symbol) ? "active" : ""}`}
            onClick={() => onToggleFavorite(symbol)}
            aria-label={`${favorites.includes(symbol) ? "Remove" : "Add"} ${symbol} ${favorites.includes(symbol) ? "from" : "to"} favorites`}
            title={favorites.includes(symbol) ? "Unfavorite" : "Favorite"}
          >
            ★
          </button>
          <button
            type="button"
            className="watchlist-remove"
            onClick={() => onRemoveTicker(symbol)}
            aria-label={`Remove ${symbol} from watchlist`}
            title={`Remove ${symbol}`}
          >
            ×
          </button>
        </div>
      ))}
    </section>
  );
}
