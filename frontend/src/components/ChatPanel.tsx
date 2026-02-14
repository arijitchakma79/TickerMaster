import { useState } from "react";
import { requestCommentary } from "../lib/api";

interface Props {
  activeTicker: string;
}

export default function ChatPanel({ activeTicker }: Props) {
  const [prompt, setPrompt] = useState(`What matters most for ${activeTicker} right now?`);
  const [response, setResponse] = useState("");
  const [model, setModel] = useState("");
  const [loading, setLoading] = useState(false);

  async function submit() {
    setLoading(true);
    try {
      const out = await requestCommentary(prompt, { ticker: activeTicker });
      setResponse(out.response);
      setModel(out.model);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="panel stack">
      <header className="panel-header">
        <h2>Desk Commentary</h2>
        <p>OpenAI-powered live narration for market context and educational signal interpretation.</p>
      </header>
      <div className="glass-card stack">
        <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} rows={4} />
        <button onClick={submit} disabled={loading}>
          {loading ? "Generating…" : "Generate Commentary"}
        </button>
        {response ? (
          <article className="chat-output">
            <p>{response}</p>
            <span className="muted">model: {model}</span>
          </article>
        ) : null}
      </div>
    </section>
  );
}
