import { useEffect, useRef, useState } from "react";
import { runVoiceTurn } from "../lib/api";
import type { VoiceHistoryMessage } from "../lib/types";

interface Props {
  activeTicker: string;
}

const RECORDER_MIME_TYPES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/mp4",
  "audio/ogg;codecs=opus",
];

function resolveRecorderMimeType() {
  if (typeof MediaRecorder === "undefined" || typeof MediaRecorder.isTypeSupported !== "function") {
    return undefined;
  }
  return RECORDER_MIME_TYPES.find((candidate) => MediaRecorder.isTypeSupported(candidate));
}

function errorMessage(error: unknown): string {
  if (
    error &&
    typeof error === "object" &&
    "response" in error &&
    (error as { response?: { data?: { detail?: unknown } } }).response?.data?.detail
  ) {
    const detail = (error as { response?: { data?: { detail?: unknown } } }).response?.data?.detail;
    if (typeof detail === "string" && detail.trim()) {
      return detail;
    }
  }
  if (error instanceof Error && error.message.trim()) {
    return error.message;
  }
  return "Voice request failed.";
}

export default function VoiceBrokerPanel({ activeTicker }: Props) {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [reply, setReply] = useState("");
  const [model, setModel] = useState("");
  const [error, setError] = useState("");
  const [history, setHistory] = useState<VoiceHistoryMessage[]>([]);
  const [recordingSupported, setRecordingSupported] = useState(true);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  function stopActiveStream() {
    const stream = streamRef.current;
    if (!stream) return;
    for (const track of stream.getTracks()) {
      track.stop();
    }
    streamRef.current = null;
  }

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      setRecordingSupported(false);
    }
  }, []);

  useEffect(() => {
    return () => {
      const recorder = recorderRef.current;
      if (recorder && recorder.state !== "inactive") {
        recorder.onstop = null;
        recorder.stop();
      }
      recorderRef.current = null;
      stopActiveStream();
      chunksRef.current = [];
    };
  }, []);

  async function finalizeRecording(mimeType: string) {
    const chunks = chunksRef.current;
    chunksRef.current = [];
    recorderRef.current = null;
    stopActiveStream();

    if (chunks.length === 0) {
      setError("No speech captured. Please try again.");
      return;
    }

    setIsProcessing(true);
    setError("");
    try {
      const blob = new Blob(chunks, { type: mimeType });
      const result = await runVoiceTurn(blob, history.slice(-12));
      setTranscript(result.transcript);
      setReply(result.response);
      setModel(result.model);
      setHistory((previous) => {
        const next: VoiceHistoryMessage[] = [
          ...previous,
          { role: "user", content: result.transcript },
          { role: "assistant", content: result.response },
        ];
        return next.slice(-20);
      });

      const player = audioRef.current;
      if (player) {
        player.src = `data:${result.audio_mime_type};base64,${result.audio_base64}`;
        await player.play().catch(() => null);
      }
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setIsProcessing(false);
    }
  }

  async function startRecording() {
    setError("");
    if (!recordingSupported) {
      setError("This browser does not support microphone recording.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = resolveRecorderMimeType();
      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);

      streamRef.current = stream;
      recorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };
      recorder.onerror = () => {
        setError("Microphone recording failed.");
      };
      recorder.onstop = () => {
        void finalizeRecording(recorder.mimeType || mimeType || "audio/webm");
      };
      recorder.start();
      setIsRecording(true);
    } catch (err) {
      stopActiveStream();
      setError(errorMessage(err));
    }
  }

  function stopRecording() {
    const recorder = recorderRef.current;
    if (!recorder) return;
    if (recorder.state !== "inactive") {
      recorder.stop();
    }
    setIsRecording(false);
  }

  function clearConversation() {
    setHistory([]);
    setTranscript("");
    setReply("");
    setModel("");
    setError("");
    if (audioRef.current) {
      audioRef.current.removeAttribute("src");
      audioRef.current.load();
    }
  }

  return (
    <section className="glass-card stack voice-broker-panel">
      <header className="panel-header">
        <h3>Broker Voice Agent</h3>
        <span className="muted">ElevenLabs STT/TTS + OpenAI tools</span>
      </header>

      <p className="muted">
        Press record, ask your question, then stop. Current focus ticker: <strong>{activeTicker}</strong>
      </p>

      <div className="voice-controls">
        {!isRecording ? (
          <button type="button" onClick={() => void startRecording()} disabled={isProcessing}>
            {isProcessing ? "Working..." : "Start Recording"}
          </button>
        ) : (
          <button type="button" onClick={stopRecording} disabled={isProcessing}>
            Stop Recording
          </button>
        )}
        <button type="button" className="secondary" onClick={clearConversation} disabled={isRecording || isProcessing || history.length === 0}>
          Clear Conversation
        </button>
      </div>

      {isProcessing ? <p className="voice-status muted">Transcribing, running tool calls, and generating voice reply...</p> : null}
      {error ? <p className="error">{error}</p> : null}

      {transcript || reply ? (
        <article className="chat-output">
          {transcript ? <p><strong>You:</strong> {transcript}</p> : null}
          {reply ? <p><strong>Agent:</strong> {reply}</p> : null}
          {model ? <span className="muted">model: {model}</span> : null}
        </article>
      ) : null}

      <audio ref={audioRef} controls className="voice-audio-player" />

      {history.length > 0 ? (
        <div className="voice-history">
          {history.slice(-6).map((entry, index) => (
            <p key={`voice-history-${index}`} className="voice-history-item">
              <strong>{entry.role === "user" ? "You" : "Agent"}:</strong> {entry.content}
            </p>
          ))}
        </div>
      ) : null}
    </section>
  );
}
