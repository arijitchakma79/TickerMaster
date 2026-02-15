import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";
import ResearchRail from "./components/ResearchRail";
import ResearchPanel from "./components/ResearchPanel";
import SimulationPanel from "./components/SimulationPanel";
import TrackerPrefsRail from "./components/TrackerPrefsRail";
import TrackerPanel from "./components/TrackerPanel";
import {
  getAuthSession,
  getFavoriteStocks,
  getUserProfile,
  getWatchlist,
  isAuthConfigured,
  setFavoriteStocks,
  setWatchlist as setTrackerWatchlist,
  signInWithPassword,
  signOut,
  signUpWithPassword,
  subscribeAuthSession,
  updateUserPreferences,
} from "./lib/api";
import { useSocket } from "./hooks/useSocket";
import brandLogo from "./images/TickerMaster.png";
import moonIcon from "./images/moon.png";
import sunIcon from "./images/sun.png";

type Tab = "research" | "simulation" | "tracker";
type Theme = "light" | "dark";
type AuthMode = "sign_in" | "sign_up";
type UserProfile = {
  display_name?: string;
  avatar_url?: string;
  email?: string;
  require_username_setup?: boolean;
  username_locked?: boolean;
};
const LANDING_VIDEO_SRC = "/videoplayback (1).mp4";
const METADATA_LOGO_SRC = "/logo.png";

function tabFromQuery(): Tab {
  const params = new URLSearchParams(window.location.search);
  const value = params.get("tab");
  if (value === "research" || value === "simulation" || value === "tracker")
    return value;
  return "simulation";
}

function tickerFromQuery() {
  return "";
}

function normalizeWatchlist(tickers: string[]) {
  const cleaned = tickers
    .map((symbol) => symbol.trim().toUpperCase())
    .filter(Boolean);
  return Array.from(new Set(cleaned));
}

function shouldRequireUsername(
  profile: UserProfile | null,
  authEmail?: string,
) {
  if (!profile) return true;
  const displayName = (profile.display_name ?? "").trim();
  const email = (profile.email ?? authEmail ?? "").trim();
  if (!displayName) return true;
  return Boolean(email && displayName.toLowerCase() === email.toLowerCase());
}

async function cropAvatarToDataUrl(
  source: string,
  zoom: number,
  offsetX: number,
  offsetY: number,
): Promise<string> {
  const image = new Image();
  image.src = source;
  await image.decode();

  const size = 320;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Unable to prepare avatar crop.");

  context.clearRect(0, 0, size, size);
  const baseScale = Math.max(size / image.width, size / image.height);
  const drawScale = baseScale * zoom;
  const drawWidth = image.width * drawScale;
  const drawHeight = image.height * drawScale;
  const centerX = (size - drawWidth) / 2 + offsetX;
  const centerY = (size - drawHeight) / 2 + offsetY;

  context.drawImage(image, centerX, centerY, drawWidth, drawHeight);
  return canvas.toDataURL("image/jpeg", 0.9);
}

function clampAvatarOffset(value: number) {
  return Math.max(-180, Math.min(180, value));
}

export default function App() {
  const [tab, setTab] = useState<Tab>(tabFromQuery());
  const [ticker, setTicker] = useState("");
  const [watchlist, setWatchlist] = useState<string[]>([]);
  const [favoriteStocks, setFavoriteStocksState] = useState<string[]>([]);
  const [theme, setTheme] = useState<Theme>(() => {
    const stored = window.localStorage.getItem("tickermaster-theme");
    if (stored === "light" || stored === "dark") return stored;
    return window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  });

  const {
    connected,
    events,
    lastSimulationTick,
    lastSimulationLifecycle,
    lastTrackerSnapshot,
  } = useSocket();
  const [authSession, setAuthSessionState] = useState(getAuthSession());
  const [authMode, setAuthMode] = useState<AuthMode>("sign_in");
  const [authEmail, setAuthEmail] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [authLoading, setAuthLoading] = useState(false);
  const [authError, setAuthError] = useState("");
  const [authModalOpen, setAuthModalOpen] = useState(false);
  const [workspaceLoading, setWorkspaceLoading] = useState(false);
  const [awaitingEmailConfirm, setAwaitingEmailConfirm] = useState(false);
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const [usernameSetupOpen, setUsernameSetupOpen] = useState(false);
  const [usernameInput, setUsernameInput] = useState("");
  const [profileError, setProfileError] = useState("");
  const [profileModalOpen, setProfileModalOpen] = useState(false);
  const [profileSaving, setProfileSaving] = useState(false);
  const [avatarCropSource, setAvatarCropSource] = useState<string | null>(null);
  const [avatarZoom, setAvatarZoom] = useState(1);
  const [avatarOffsetX, setAvatarOffsetX] = useState(0);
  const [avatarOffsetY, setAvatarOffsetY] = useState(0);
  const [avatarDragging, setAvatarDragging] = useState(false);
  const avatarDragRef = useRef<{
    pointerId: number;
    startX: number;
    startY: number;
    originX: number;
    originY: number;
  } | null>(null);

  useEffect(() => {
    const unsubscribe = subscribeAuthSession((session) => {
      setAuthSessionState(session);
    });
    return () => unsubscribe();
  }, []);

  useEffect(() => {
    let active = true;
    const hydrateWorkspace = async () => {
      if (authSession?.user?.id) {
        setWorkspaceLoading(true);
      }
      try {
        const [serverWatchlist, favoriteSymbols, profilePayload] =
          await Promise.all([
            getWatchlist().catch(() => []),
            authSession?.user?.id
              ? getFavoriteStocks().catch(() => [])
              : Promise.resolve([]),
            authSession?.user?.id
              ? getUserProfile().catch(() => ({
                  user_id: null,
                  profile: null,
                  require_username_setup: false,
                  username_locked: false,
                }))
              : Promise.resolve({
                  user_id: null,
                  profile: null,
                  require_username_setup: false,
                  username_locked: false,
                }),
          ]);

        if (!active) return;
        const synced = normalizeWatchlist(serverWatchlist);
        if (synced.length === 0) {
          setWatchlist([]);
          setTicker("");
        } else {
          setWatchlist(synced);
          // Keep research empty on startup/login until user picks a ticker.
          setTicker("");
        }
        setFavoriteStocksState(normalizeWatchlist(favoriteSymbols));
        if (profilePayload?.profile) {
          const nextProfile = {
            display_name: profilePayload.profile.display_name,
            avatar_url: profilePayload.profile.avatar_url,
            email: profilePayload.profile.email,
            require_username_setup: profilePayload.require_username_setup,
            username_locked: profilePayload.username_locked,
          };
          setUserProfile(nextProfile);
          const mustSetUsername =
            Boolean(profilePayload.require_username_setup) ||
            shouldRequireUsername(nextProfile, authSession?.user?.email);
          if (mustSetUsername) {
            setUsernameInput("");
            setUsernameSetupOpen(true);
          } else {
            setUsernameSetupOpen(false);
          }
        } else {
          setUserProfile(null);
          if (authSession?.user?.id && profilePayload?.require_username_setup) {
            setUsernameInput("");
            setUsernameSetupOpen(true);
          } else {
            setUsernameSetupOpen(false);
          }
        }
      } finally {
        if (active) setWorkspaceLoading(false);
      }
    };

    void hydrateWorkspace();
    return () => {
      active = false;
    };
  }, [authSession?.user?.id]);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    document.documentElement.style.colorScheme = theme;
    window.localStorage.setItem("tickermaster-theme", theme);
  }, [theme]);

  useEffect(() => {
    if (!authSession?.user?.id) {
      setUsernameSetupOpen(false);
      setProfileModalOpen(false);
      resetAvatarCropEditor();
    }
  }, [authSession?.user?.id]);

  useEffect(() => {
    return () => {
      if (avatarCropSource?.startsWith("blob:")) {
        URL.revokeObjectURL(avatarCropSource);
      }
    };
  }, [avatarCropSource]);

  useEffect(() => {
    let favicon = document.querySelector(
      "link[rel='icon']",
    ) as HTMLLinkElement | null;
    if (!favicon) {
      favicon = document.createElement("link");
      favicon.rel = "icon";
      document.head.appendChild(favicon);
    }
    favicon.type = "image/png";
    favicon.href = METADATA_LOGO_SRC;
  }, []);

  const title = useMemo(() => {
    if (tab === "research") return "Research Workbench";
    if (tab === "simulation") return "Simulation Arena";
    return "Ticker Tracker";
  }, [tab]);

  const subtitle = useMemo(() => {
    if (tab === "research") {
      return "Public/latest sentiments summarized into high-signal narratives.";
    }
    if (tab === "simulation") {
      return "Test your strategy using a sandbox of AI agents reacting to live prices, news catalysts, and volatility.";
    }
    return "Real-time ticker monitor with valuation metrics, alerting, and live anomaly detection pipeline.";
  }, [tab]);

  async function handleWatchlistChange(nextSymbols: string[]) {
    const normalized = normalizeWatchlist(nextSymbols);

    try {
      const serverUpdated = normalizeWatchlist(
        await setTrackerWatchlist(normalized),
      );
      const nextList = serverUpdated.length > 0 ? serverUpdated : normalized;
      setWatchlist(nextList);
      if (nextList.length === 0) {
        setTicker("");
      } else if (!nextList.includes(ticker)) {
        setTicker(nextList[0]);
      }
      return nextList;
    } catch {
      setWatchlist(normalized);
      if (normalized.length === 0) {
        setTicker("");
      } else if (!normalized.includes(ticker)) {
        setTicker(normalized[0]);
      }
      return normalized;
    }
  }

  async function handleToggleFavorite(symbol: string) {
    const normalized = symbol.trim().toUpperCase();
    if (!normalized) return;
    const next = favoriteStocks.includes(normalized)
      ? favoriteStocks.filter((item) => item !== normalized)
      : [...favoriteStocks, normalized];
    if (!authSession?.user?.id) {
      setFavoriteStocksState(normalizeWatchlist(next));
      return;
    }
    try {
      const saved = await setFavoriteStocks(normalizeWatchlist(next));
      setFavoriteStocksState(normalizeWatchlist(saved));
    } catch {
      // no-op
    }
  }

  async function handleAuthSubmit() {
    if (!authEmail.trim() || !authPassword) {
      setAuthError("Email and password are required.");
      return;
    }
    setAuthLoading(true);
    setAuthError("");
    try {
      if (authMode === "sign_in") {
        await signInWithPassword(authEmail.trim(), authPassword);
        setAwaitingEmailConfirm(false);
        setAuthModalOpen(false);
      } else {
        const session = await signUpWithPassword(
          authEmail.trim(),
          authPassword,
        );
        if (!session) {
          setAwaitingEmailConfirm(true);
          setAuthError("");
        } else {
          setAwaitingEmailConfirm(false);
          setAuthModalOpen(false);
        }
      }
      setAuthPassword("");
    } catch (error) {
      setAuthError(
        error instanceof Error ? error.message : "Authentication failed.",
      );
    } finally {
      setAuthLoading(false);
    }
  }

  useEffect(() => {
    if (!authModalOpen || !awaitingEmailConfirm || authMode !== "sign_up")
      return;
    if (!authEmail.trim() || !authPassword) return;
    if (authSession?.user?.id) return;

    let active = true;
    const interval = window.setInterval(() => {
      if (!active) return;
      void signInWithPassword(authEmail.trim(), authPassword)
        .then(() => {
          if (!active) return;
          setAwaitingEmailConfirm(false);
          setAuthModalOpen(false);
          setAuthError("");
          setAuthPassword("");
        })
        .catch(() => {
          // Keep polling until confirmation is completed.
        });
    }, 4000);

    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, [
    authModalOpen,
    awaitingEmailConfirm,
    authMode,
    authEmail,
    authPassword,
    authSession?.user?.id,
  ]);

  async function tryCompleteEmailConfirmation() {
    if (!authEmail.trim() || !authPassword) return;
    setAuthLoading(true);
    setAuthError("");
    try {
      await signInWithPassword(authEmail.trim(), authPassword);
      setAwaitingEmailConfirm(false);
      setAuthModalOpen(false);
      setAuthPassword("");
    } catch (error) {
      setAuthError(
        error instanceof Error
          ? error.message
          : "Confirmation not complete yet.",
      );
    } finally {
      setAuthLoading(false);
    }
  }

  function resetAvatarCropEditor() {
    setAvatarCropSource(null);
    setAvatarZoom(1);
    setAvatarOffsetX(0);
    setAvatarOffsetY(0);
    setAvatarDragging(false);
    avatarDragRef.current = null;
  }

  function handleAvatarFileSelected(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    const objectUrl = URL.createObjectURL(file);
    setAvatarCropSource(objectUrl);
    setAvatarZoom(1);
    setAvatarOffsetX(0);
    setAvatarOffsetY(0);
    setProfileError("");
    event.target.value = "";
  }

  function handleAvatarPointerDown(event: ReactPointerEvent<HTMLDivElement>) {
    if (!avatarCropSource) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    avatarDragRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      originX: avatarOffsetX,
      originY: avatarOffsetY,
    };
    setAvatarDragging(true);
  }

  function handleAvatarPointerMove(event: ReactPointerEvent<HTMLDivElement>) {
    const activeDrag = avatarDragRef.current;
    if (!activeDrag || activeDrag.pointerId !== event.pointerId) return;
    event.preventDefault();
    const deltaX = event.clientX - activeDrag.startX;
    const deltaY = event.clientY - activeDrag.startY;
    setAvatarOffsetX(clampAvatarOffset(activeDrag.originX + deltaX));
    setAvatarOffsetY(clampAvatarOffset(activeDrag.originY + deltaY));
  }

  function handleAvatarPointerUp(event: ReactPointerEvent<HTMLDivElement>) {
    const activeDrag = avatarDragRef.current;
    if (!activeDrag || activeDrag.pointerId !== event.pointerId) return;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    avatarDragRef.current = null;
    setAvatarDragging(false);
  }

  async function buildAvatarDataUrlIfNeeded() {
    if (!avatarCropSource) return undefined;
    try {
      return await cropAvatarToDataUrl(
        avatarCropSource,
        avatarZoom,
        avatarOffsetX,
        avatarOffsetY,
      );
    } catch {
      throw new Error("Could not crop avatar image.");
    }
  }

  async function handleCompleteUsernameSetup() {
    const candidate = usernameInput.trim();
    if (candidate.length < 3) {
      setProfileError("Username must be at least 3 characters.");
      return;
    }
    setProfileSaving(true);
    setProfileError("");
    try {
      const avatarDataUrl = await buildAvatarDataUrlIfNeeded();
      const response = await updateUserPreferences({
        display_name: candidate,
        avatar_data_url: avatarDataUrl,
      });
      if (!response.ok) {
        throw new Error("Could not save username right now. Please try again.");
      }
      const nextProfile = response.profile ?? {
        display_name: candidate,
        avatar_url: avatarDataUrl,
        email: userProfile?.email ?? authSession?.user?.email,
      };
      setUserProfile({
        display_name: nextProfile.display_name,
        avatar_url: nextProfile.avatar_url,
        email:
          nextProfile.email ?? userProfile?.email ?? authSession?.user?.email,
        require_username_setup: false,
        username_locked: true,
      });
      setUsernameSetupOpen(false);
      resetAvatarCropEditor();
    } catch (error) {
      setProfileError(
        error instanceof Error ? error.message : "Unable to save username.",
      );
    } finally {
      setProfileSaving(false);
    }
  }

  async function handleSaveProfile() {
    setProfileSaving(true);
    setProfileError("");
    try {
      const avatarDataUrl = await buildAvatarDataUrlIfNeeded();
      const response = await updateUserPreferences({
        avatar_data_url: avatarDataUrl,
      });
      if (!response.ok) {
        throw new Error(
          "Could not save profile image right now. Please try again.",
        );
      }
      const savedAvatar =
        response.profile?.avatar_url ??
        avatarDataUrl ??
        userProfile?.avatar_url;
      setUserProfile((prev) => ({
        ...(prev ?? {}),
        avatar_url: savedAvatar,
        email: prev?.email ?? authSession?.user?.email,
      }));
      setProfileModalOpen(false);
      resetAvatarCropEditor();
    } catch (error) {
      setProfileError(
        error instanceof Error
          ? error.message
          : "Unable to save profile image.",
      );
    } finally {
      setProfileSaving(false);
    }
  }

  const profileName = useMemo(() => {
    const fromProfile = userProfile?.display_name?.trim();
    if (fromProfile) return fromProfile;
    const email = userProfile?.email ?? authSession?.user?.email ?? "";
    if (email.includes("@")) return email.split("@")[0];
    return "Profile";
  }, [userProfile?.display_name, userProfile?.email, authSession?.user?.email]);
  const hasWorkbenchAccess = Boolean(authSession?.user?.id);
  const showLandingBackground = !hasWorkbenchAccess;
  const landingTapeItems = [
    { symbol: "SPX", value: "5,112.34", change: "+0.84%", up: true },
    { symbol: "NDX", value: "18,021.10", change: "+1.12%", up: true },
    { symbol: "DJI", value: "38,904.77", change: "-0.21%", up: false },
    { symbol: "BTC", value: "63,442", change: "+2.04%", up: true },
    { symbol: "ETH", value: "3,214", change: "+1.77%", up: true },
    { symbol: "VIX", value: "14.2", change: "-3.31%", up: false },
    { symbol: "AAPL", value: "192.18", change: "+0.49%", up: true },
    { symbol: "MSFT", value: "413.75", change: "+0.66%", up: true },
    { symbol: "NVDA", value: "726.41", change: "+1.38%", up: true },
    { symbol: "TSLA", value: "192.44", change: "-1.14%", up: false },
    { symbol: "AMZN", value: "176.21", change: "+0.41%", up: true },
    { symbol: "META", value: "468.10", change: "-0.33%", up: false },
  ];

  function openSignInModal() {
    setAuthError("");
    setProfileError("");
    setAwaitingEmailConfirm(false);
    setAuthEmail("");
    setAuthPassword("");
    setAuthMode("sign_in");
    setAuthModalOpen(true);
  }

  return (
    <div className="app-shell">
      {showLandingBackground ? (
        <>
          <div className="wallstreet-video-wrap" aria-hidden="true">
            <video
              className="wallstreet-video"
              autoPlay
              muted
              loop
              playsInline
              preload="auto"
            >
              <source src={LANDING_VIDEO_SRC} type="video/mp4" />
            </video>
          </div>
          <div className="wallstreet-video-overlay" aria-hidden="true" />
        </>
      ) : null}

      <div className="top-fixed-controls">
        <div className="top-fixed-controls-inner">
          <div className="brand-lockup" aria-label="TickerMaster">
            <img
              src={brandLogo}
              alt="TickerMaster"
              className="brand-logo-image"
            />
          </div>
          <div className="top-nav-right">
            <div className="theme-switch-wrap">
              <button
                type="button"
                className={`theme-switch ${theme}`}
                onClick={() =>
                  setTheme((prev) => (prev === "light" ? "dark" : "light"))
                }
                aria-label={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
                aria-pressed={theme === "dark"}
                title={theme === "light" ? "Light mode" : "Dark mode"}
              >
                <span className="theme-switch-track" />
                <span className="theme-switch-thumb">
                  <img src={theme === "light" ? sunIcon : moonIcon} alt="" />
                </span>
              </button>
            </div>
          </div>
        </div>
      </div>
      {isAuthConfigured() ? (
        authSession?.user?.id ? (
          <div className="account-corner">
            <div className="account-panel">
              <button
                type="button"
                className="profile-trigger"
                onClick={() => {
                  setProfileError("");
                  resetAvatarCropEditor();
                  setProfileModalOpen(true);
                }}
              >
                {userProfile?.avatar_url ? (
                  <img
                    src={userProfile.avatar_url}
                    alt="Profile"
                    className="profile-avatar"
                  />
                ) : (
                  <span className="profile-avatar profile-avatar-fallback">
                    {profileName.slice(0, 1).toUpperCase()}
                  </span>
                )}
                <span className="profile-name">{profileName}</span>
              </button>
              <button
                className="secondary auth-btn account-signout"
                onClick={() => void signOut()}
              >
                Sign Out
              </button>
            </div>
          </div>
        ) : (
          <div className="account-corner">
            <button className="secondary auth-btn" onClick={openSignInModal}>
              Sign In
            </button>
          </div>
        )
      ) : (
        <div className="account-corner">
          <span className="muted auth-config-hint auth-config-hint-left">
            Set Supabase env vars to enable sign in.
          </span>
        </div>
      )}

      <div className="ambient ambient-1" />
      <div className="ambient ambient-2" />

      {hasWorkbenchAccess ? (
        <>
          <header className="hero">
            <div>
              <h1>{title}</h1>
              <p className="subtitle">{subtitle}</p>
            </div>
          </header>

          <nav className="tab-row">
            <button
              className={tab === "research" ? "tab active" : "tab"}
              onClick={() => setTab("research")}
              aria-current={tab === "research" ? "page" : undefined}
            >
              <span className="tab-inner">
                <span className="tab-icon" aria-hidden="true">
                  🔎
                </span>
                <span>Research</span>
              </span>
            </button>
            <button
              className={tab === "simulation" ? "tab active core" : "tab core"}
              onClick={() => setTab("simulation")}
              aria-current={tab === "simulation" ? "page" : undefined}
            >
              <span className="tab-inner">
                <span className="tab-icon arena" aria-hidden="true">
                  🤖⚔️
                </span>
                <span>Simulation</span>
              </span>
            </button>
            <button
              className={tab === "tracker" ? "tab active" : "tab"}
              onClick={() => setTab("tracker")}
              aria-current={tab === "tracker" ? "page" : undefined}
            >
              <span className="tab-inner">
                <span className="tab-icon" aria-hidden="true">
                  📊
                </span>
                <span>Tracker</span>
              </span>
            </button>
          </nav>

          <main
            className={
              tab === "simulation"
                ? "layout-grid layout-grid-single"
                : "layout-grid"
            }
          >
            <div>
              {tab === "research" ? (
                <ResearchPanel
                  activeTicker={ticker}
                  onTickerChange={setTicker}
                  connected={connected}
                  events={events}
                />
              ) : null}
              {tab === "simulation" ? (
                <SimulationPanel
                  activeTicker={ticker}
                  onTickerChange={setTicker}
                  watchlist={watchlist}
                  connected={connected}
                  simulationEvent={lastSimulationTick}
                  simulationLifecycleEvent={lastSimulationLifecycle}
                />
              ) : null}
              {tab === "tracker" ? (
                <TrackerPanel
                  activeTicker={ticker}
                  onTickerChange={setTicker}
                  trackerEvent={lastTrackerSnapshot}
                  watchlist={watchlist}
                  onWatchlistChange={handleWatchlistChange}
                />
              ) : null}
            </div>

            {tab === "research" ? (
              <ResearchRail
                connected={connected}
                activeTicker={ticker}
                onTickerSelect={setTicker}
                trackerEvent={lastTrackerSnapshot}
              />
            ) : null}
            {tab === "tracker" ? (
              <TrackerPrefsRail connected={connected} userId={authSession?.user?.id ?? null} />
            ) : null}
          </main>
        </>
      ) : (
        <main className="tv-landing">
          <section className="tv-hero">
            <div className="tv-tape" aria-label="Market ticker">
              <div className="tv-tape-marquee">
                {[0, 1].map((loop) => (
                  <div
                    className="tv-tape-track"
                    aria-hidden={loop === 1}
                    key={loop}
                  >
                    {landingTapeItems.map((item) => (
                      <span
                        className="tv-tape-item"
                        key={`${loop}-${item.symbol}`}
                      >
                        {item.symbol} {item.value}{" "}
                        <b className={item.up ? "up" : "down"}>{item.change}</b>
                      </span>
                    ))}
                  </div>
                ))}
              </div>
            </div>
            <div className="tv-hero-inner">
              <p className="tv-kicker">TickerMaster Platform</p>
              <h2 className="tv-title">
                Professional Market Intelligence For Modern Traders
              </h2>
              <p className="tv-subtitle">
                Research, simulation, and personalized tracking in one
                integrated workflow.
              </p>
              <button className="tv-cta" onClick={openSignInModal}>
                Try Now
              </button>
            </div>
          </section>

          <section className="tv-section">
            <div className="tv-section-head">
              <p className="tv-kicker">Core Modules</p>
              <h3>Built for disciplined investment workflows</h3>
            </div>
            <div className="tv-feature-grid">
              <article className="tv-feature-card">
                <h4>AI Research Workbench</h4>
                <p>
                  Aggregate macro, sentiment, and market signals into a single
                  thesis view.
                </p>
              </article>
              <article className="tv-feature-card">
                <h4>Multi-Agent Simulation</h4>
                <p>
                  Stress-test strategies across volatility regimes before
                  committing capital.
                </p>
              </article>
              <article className="tv-feature-card">
                <h4>Persistent Tracker</h4>
                <p>
                  Maintain user-specific watchlists, favorites, alerts, and
                  monitoring agents.
                </p>
              </article>
            </div>
          </section>

          <section className="tv-section">
            <div className="tv-section-head">
              <p className="tv-kicker">Why TickerMaster</p>
              <h3>From idea generation to execution readiness</h3>
            </div>
            <div className="tv-stats-grid">
              <div className="tv-stat-card">
                <strong>Realtime</strong>
                <span>Live feed integrations across your market workflow.</span>
              </div>
              <div className="tv-stat-card">
                <strong>Per-User Data</strong>
                <span>
                  Profiles, favorites, and agents persist securely by account.
                </span>
              </div>
              <div className="tv-stat-card">
                <strong>End-to-End Flow</strong>
                <span>
                  Research, simulate, and track without switching platforms.
                </span>
              </div>
            </div>
          </section>
        </main>
      )}

      {workspaceLoading ? (
        <div className="app-loading-overlay" role="status" aria-live="polite">
          <div className="app-loading-card">Loading your workspace...</div>
        </div>
      ) : null}

      {authModalOpen ? (
        <div
          className="auth-modal-backdrop"
          onClick={() =>
            !authLoading && !awaitingEmailConfirm
              ? setAuthModalOpen(false)
              : null
          }
        >
          <div
            className="auth-modal"
            onClick={(event) => event.stopPropagation()}
          >
            <h3>{authMode === "sign_in" ? "Sign In" : "Create Account"}</h3>
            <p className="muted">
              {authMode === "sign_in"
                ? "Sign in to TickerMaster."
                : "Sign up for TickerMaster to save agents, favorites, and watchlists."}
            </p>
            <input
              type="email"
              value={authEmail}
              onChange={(event) => setAuthEmail(event.target.value)}
              placeholder="Email"
              autoFocus
            />
            <input
              type="password"
              value={authPassword}
              onChange={(event) => setAuthPassword(event.target.value)}
              placeholder="Password"
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault();
                  void handleAuthSubmit();
                }
              }}
            />
            <button
              onClick={() => void handleAuthSubmit()}
              disabled={authLoading}
            >
              {authLoading
                ? "Working..."
                : authMode === "sign_in"
                  ? "Sign In"
                  : "Sign Up"}
            </button>
            {authMode === "sign_in" ? (
              <button
                className="secondary"
                onClick={() => {
                  setAuthError("");
                  setAwaitingEmailConfirm(false);
                  setAuthMode("sign_up");
                }}
                disabled={authLoading}
              >
                Need an account? Sign up
              </button>
            ) : null}
            {authMode === "sign_up" && awaitingEmailConfirm ? (
              <div className="auth-confirm-box">
                <p className="muted">
                  Check your email and click the confirmation link. We will sign
                  you in automatically once confirmed.
                </p>
                <button
                  className="secondary"
                  onClick={() => void tryCompleteEmailConfirmation()}
                  disabled={authLoading}
                >
                  I've Confirmed My Email
                </button>
              </div>
            ) : null}
            {authMode === "sign_up" && !awaitingEmailConfirm ? (
              <button
                className="secondary"
                onClick={() => {
                  setAuthError("");
                  setAwaitingEmailConfirm(false);
                  setAuthMode("sign_in");
                }}
                disabled={authLoading}
              >
                Already have an account? Sign in
              </button>
            ) : null}
            {authError ? <p className="auth-error">{authError}</p> : null}
          </div>
        </div>
      ) : null}

      {usernameSetupOpen ? (
        <div className="auth-modal-backdrop">
          <div
            className="auth-modal"
            onClick={(event) => event.stopPropagation()}
          >
            <h3>Set Your Username</h3>
            <p className="muted">
              Choose your permanent username. It cannot be changed later.
            </p>
            <input
              type="text"
              value={usernameInput}
              onChange={(event) => setUsernameInput(event.target.value)}
              placeholder="Username (3-24 characters)"
              minLength={3}
              maxLength={24}
              autoFocus
            />
            <label className="auth-upload-field">
              Profile Photo
              <input
                type="file"
                accept="image/*"
                onChange={handleAvatarFileSelected}
              />
            </label>
            {avatarCropSource ? (
              <div className="avatar-crop-stack">
                <div
                  className={`avatar-crop-frame${avatarDragging ? " dragging" : ""}`}
                  aria-label="Avatar crop preview"
                  onPointerDown={handleAvatarPointerDown}
                  onPointerMove={handleAvatarPointerMove}
                  onPointerUp={handleAvatarPointerUp}
                  onPointerCancel={handleAvatarPointerUp}
                >
                  <img
                    src={avatarCropSource}
                    alt="Avatar crop preview"
                    className="avatar-crop-image"
                    style={{
                      transform: `translate(${avatarOffsetX}px, ${avatarOffsetY}px) scale(${avatarZoom})`,
                    }}
                  />
                </div>
                <label>
                  Zoom
                  <input
                    type="range"
                    min={1}
                    max={3}
                    step={0.01}
                    value={avatarZoom}
                    onChange={(event) =>
                      setAvatarZoom(Number(event.target.value))
                    }
                  />
                </label>
                <label>
                  Horizontal
                  <input
                    type="range"
                    min={-180}
                    max={180}
                    step={1}
                    value={avatarOffsetX}
                    onChange={(event) =>
                      setAvatarOffsetX(Number(event.target.value))
                    }
                  />
                </label>
                <label>
                  Vertical
                  <input
                    type="range"
                    min={-180}
                    max={180}
                    step={1}
                    value={avatarOffsetY}
                    onChange={(event) =>
                      setAvatarOffsetY(Number(event.target.value))
                    }
                  />
                </label>
              </div>
            ) : null}
            <button
              onClick={() => void handleCompleteUsernameSetup()}
              disabled={profileSaving}
            >
              {profileSaving ? "Saving..." : "Continue"}
            </button>
            {profileError ? <p className="auth-error">{profileError}</p> : null}
          </div>
        </div>
      ) : null}

      {profileModalOpen ? (
        <div
          className="auth-modal-backdrop"
          onClick={() => (!profileSaving ? setProfileModalOpen(false) : null)}
        >
          <div
            className="auth-modal"
            onClick={(event) => event.stopPropagation()}
          >
            <h3>Profile</h3>
            <p className="muted">
              Username is permanent and cannot be changed.
            </p>
            <input
              type="text"
              value={userProfile?.display_name ?? profileName}
              readOnly
            />
            <label className="auth-upload-field">
              Upload New Profile Photo
              <input
                type="file"
                accept="image/*"
                onChange={handleAvatarFileSelected}
              />
            </label>
            {avatarCropSource ? (
              <div className="avatar-crop-stack">
                <div
                  className={`avatar-crop-frame${avatarDragging ? " dragging" : ""}`}
                  aria-label="Avatar crop preview"
                  onPointerDown={handleAvatarPointerDown}
                  onPointerMove={handleAvatarPointerMove}
                  onPointerUp={handleAvatarPointerUp}
                  onPointerCancel={handleAvatarPointerUp}
                >
                  <img
                    src={avatarCropSource}
                    alt="Avatar crop preview"
                    className="avatar-crop-image"
                    style={{
                      transform: `translate(${avatarOffsetX}px, ${avatarOffsetY}px) scale(${avatarZoom})`,
                    }}
                  />
                </div>
                <label>
                  Zoom
                  <input
                    type="range"
                    min={1}
                    max={3}
                    step={0.01}
                    value={avatarZoom}
                    onChange={(event) =>
                      setAvatarZoom(Number(event.target.value))
                    }
                  />
                </label>
                <label>
                  Horizontal
                  <input
                    type="range"
                    min={-180}
                    max={180}
                    step={1}
                    value={avatarOffsetX}
                    onChange={(event) =>
                      setAvatarOffsetX(Number(event.target.value))
                    }
                  />
                </label>
                <label>
                  Vertical
                  <input
                    type="range"
                    min={-180}
                    max={180}
                    step={1}
                    value={avatarOffsetY}
                    onChange={(event) =>
                      setAvatarOffsetY(Number(event.target.value))
                    }
                  />
                </label>
              </div>
            ) : null}
            <button
              onClick={() => void handleSaveProfile()}
              disabled={profileSaving}
            >
              {profileSaving ? "Saving..." : "Save"}
            </button>
            <button
              className="secondary"
              onClick={() => setProfileModalOpen(false)}
              disabled={profileSaving}
            >
              Cancel
            </button>
            {profileError ? <p className="auth-error">{profileError}</p> : null}
          </div>
        </div>
      ) : null}
    </div>
  );
}
