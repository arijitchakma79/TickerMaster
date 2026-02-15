interface ImportMetaEnv {
  readonly VITE_API_URL?: string;
  readonly VITE_WS_URL?: string;
  readonly VITE_SUPABASE_URL?: string;
  readonly VITE_SUPABASE_ANON_KEY?: string;
  readonly VITE_FINNHUB_API_KEY?: string;
  readonly VITE_TWELVEDATA_API_KEY?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
