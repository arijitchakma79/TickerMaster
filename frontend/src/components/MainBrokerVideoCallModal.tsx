import { useEffect, useRef, useState } from "react";
import { AgentEventsEnum, LiveAvatarSession, SessionEvent, SessionState } from "@heygen/liveavatar-web-sdk";
import { createMainBrokerAvatarSession, interactWithMainBroker } from "../lib/api";
import type { TrackerAgent } from "../lib/types";

type Props = {
  agents: TrackerAgent[];
  onClose: () => void;
};

function toErrorMessage(error: unknown): string {
  if (error instanceof Error && error.message.trim()) return error.message;
  return "Call request failed.";
}

export default function MainBrokerVideoCallModal({ agents, onClose }: Props) {
  const [callStatus, setCallStatus] = useState<"idle" | "starting" | "live" | "stopping">("idle");
  const [sessionState, setSessionState] = useState("INACTIVE");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [sending, setSending] = useState(false);
  const [listening, setListening] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const sessionRef = useRef<LiveAvatarSession | null>(null);
  const mountedRef = useRef(true);
  const sendingRef = useRef(false);
  const listeningRef = useRef(false);
  const voiceDebounceRef = useRef<number | null>(null);
  const latestVoicePromptRef = useRef("");
  const lastVoicePromptRef = useRef("");

  const activeCount = agents.filter((agent) => String(agent.status).toLowerCase() === "active").length;
  const totalCount = agents.length;

  function clearVoiceDebounce() {
    if (voiceDebounceRef.current !== null) {
      window.clearTimeout(voiceDebounceRef.current);
      voiceDebounceRef.current = null;
    }
  }

  async function stopSession() {
    const session = sessionRef.current;
    sessionRef.current = null;
    if (!session) return;
    try {
      await session.stop();
    } catch {
      // no-op
    }
  }

  async function requestBrokerReply(prompt: string) {
    const messageText = prompt.trim();
    if (!messageText || sendingRef.current) return;

    sendingRef.current = true;
    setSending(true);
    setError("");
    try {
      const interaction = await interactWithMainBroker(messageText, undefined, { agent_limit: 6 });
      console.log("[MainBroker Interaction]", {
        message: messageText,
        selected_agents: interaction.selected_agents,
        tool_outputs: interaction.tool_outputs,
      });
      const reply = String(interaction.reply?.response || "").trim();
      if (reply && sessionRef.current) {
        sessionRef.current.repeat(reply);
      }
    } catch (err) {
      setError(toErrorMessage(err));
    } finally {
      sendingRef.current = false;
      if (mountedRef.current) setSending(false);
    }
  }

  async function startSession() {
    if (sessionRef.current || callStatus === "starting" || callStatus === "live") return;
    setError("");
    setCallStatus("starting");
    latestVoicePromptRef.current = "";
    lastVoicePromptRef.current = "";
    try {
      const sessionMeta = await createMainBrokerAvatarSession();
      const session = new LiveAvatarSession(sessionMeta.session_token, {
        voiceChat: true,
        apiUrl: sessionMeta.api_url || undefined,
      });
      sessionRef.current = session;

      session.on(SessionEvent.SESSION_STATE_CHANGED, (nextState) => {
        if (!mountedRef.current) return;
        setSessionState(String(nextState));
        if (nextState === SessionState.CONNECTED) setCallStatus("live");
        if (nextState === SessionState.DISCONNECTED || nextState === SessionState.INACTIVE) {
          setCallStatus("idle");
          setListening(false);
        }
      });

      session.on(SessionEvent.SESSION_DISCONNECTED, () => {
        if (!mountedRef.current) return;
        setCallStatus("idle");
        setListening(false);
      });

      session.on(AgentEventsEnum.USER_TRANSCRIPTION, (event) => {
        if (!mountedRef.current || !listeningRef.current) return;
        const text = String(event.text || "").trim();
        if (!text) return;
        latestVoicePromptRef.current = text;
        clearVoiceDebounce();
        voiceDebounceRef.current = window.setTimeout(() => {
          const spokenPrompt = latestVoicePromptRef.current.trim();
          if (!spokenPrompt || spokenPrompt.length < 3) return;
          if (spokenPrompt.toLowerCase() === lastVoicePromptRef.current.toLowerCase()) return;
          if (sendingRef.current) return;
          lastVoicePromptRef.current = spokenPrompt;
          void requestBrokerReply(spokenPrompt);
        }, 900);
      });

      await session.start();
      if (videoRef.current) {
        session.attach(videoRef.current);
      }
      try {
        session.startListening();
        setListening(true);
      } catch {
        // Mic start can fail until user toggles manually.
      }
      setSessionState("CONNECTED");
      setCallStatus("live");
    } catch (err) {
      await stopSession();
      if (!mountedRef.current) return;
      setCallStatus("idle");
      setSessionState("DISCONNECTED");
      setError(toErrorMessage(err));
    }
  }

  async function stopCall() {
    if (callStatus === "stopping" || callStatus === "idle") return;
    setCallStatus("stopping");
    setListening(false);
    clearVoiceDebounce();
    await stopSession();
    if (!mountedRef.current) return;
    setCallStatus("idle");
    setSessionState("DISCONNECTED");
  }

  async function handleSendMessage() {
    const userMessage = message.trim();
    if (!userMessage || sending) return;
    setMessage("");
    await requestBrokerReply(userMessage);
  }

  function toggleListening() {
    const session = sessionRef.current;
    if (!session) {
      setError("Start the call first.");
      return;
    }
    try {
      if (listening) {
        session.stopListening();
        setListening(false);
        clearVoiceDebounce();
      } else {
        session.startListening();
        setListening(true);
      }
    } catch (err) {
      setError(toErrorMessage(err));
    }
  }

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      clearVoiceDebounce();
      void stopSession();
    };
  }, []);

  useEffect(() => {
    listeningRef.current = listening;
  }, [listening]);

  return (
    <div className="auth-modal-backdrop" onClick={onClose}>
      <div className="auth-modal tracker-call-modal" onClick={(event) => event.stopPropagation()}>
        <h3>Main Broker Call</h3>
        <p className="muted">
          Session: {sessionState} | Tracking {activeCount}/{totalCount} active agents
        </p>

        <div className="tracker-call-video-wrap">
          <video ref={videoRef} autoPlay playsInline />
        </div>

        <div className="tracker-call-actions">
          <button type="button" onClick={() => void startSession()} disabled={callStatus === "starting" || callStatus === "live"}>
            {callStatus === "starting" ? "Starting..." : callStatus === "live" ? "Live" : "Start Call"}
          </button>
          <button type="button" className="secondary" onClick={() => void stopCall()} disabled={callStatus === "idle" || callStatus === "stopping"}>
            {callStatus === "stopping" ? "Stopping..." : "End Call"}
          </button>
          <button type="button" className="secondary" onClick={toggleListening} disabled={callStatus !== "live"}>
            {listening ? "Stop Mic" : "Start Mic"}
          </button>
          <button type="button" className="secondary" onClick={onClose}>
            Close
          </button>
        </div>

        <div className="tracker-call-compose">
          <input
            value={message}
            onChange={(event) => setMessage(event.target.value)}
            placeholder="Ask the main broker..."
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                void handleSendMessage();
              }
            }}
          />
          <button type="button" onClick={() => void handleSendMessage()} disabled={sending || !message.trim()}>
            {sending ? "Thinking..." : "Send"}
          </button>
        </div>

        {error ? <p className="auth-error">{error}</p> : null}
      </div>
    </div>
  );
}
