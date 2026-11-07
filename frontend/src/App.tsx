import { useEffect, useMemo, useRef, useState } from "react";
import { type AskResponse, type Health, ApiError, api } from "./api";
import { DocumentPanel } from "./DocumentPanel";
import { RunDetails } from "./RunDetails";

const REFUSALS: Record<string, { title: string; body: string }> = {
  insufficient_evidence: {
    title: "The documents don't answer this",
    body: "No document covers this question, so no answer was written. Try rephrasing it, or ask about a topic the documents cover.",
  },
  check_failed: {
    title: "No verified answer",
    body: "A draft answer contained figures or links that are not in the sentences it cited, so it is not shown.",
  },
  unsupported: {
    title: "No verified answer",
    body: "A draft answer made a claim its cited sentences do not support, so it is not shown.",
  },
  contradicted: {
    title: "No verified answer",
    body: "A draft answer contradicted the sentences it cited, so it is not shown.",
  },
};

type State =
  | { phase: "idle" }
  | { phase: "asking"; question: string }
  | { phase: "answered"; question: string; result: AskResponse }
  | { phase: "failed"; question: string; message: string };

export function App() {
  const [question, setQuestion] = useState("");
  const [state, setState] = useState<State>({ phase: "idle" });
  const [health, setHealth] = useState<Health | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  // Bumped only by explicit citation clicks, so a new answer never scrolls the page.
  const [scrollRequest, setScrollRequest] = useState(0);
  const resultRef = useRef<HTMLElement>(null);

  useEffect(() => {
    api.health().then(setHealth, () => setHealth(null));
  }, []);

  async function ask(event: React.FormEvent) {
    event.preventDefault();
    const text = question.trim();
    if (text.length < 3 || state.phase === "asking") return;
    setState({ phase: "asking", question: text });
    setSelected(null);
    setScrollRequest(0);
    try {
      const result = await api.ask(text);
      setState({ phase: "answered", question: text, result });
      setSelected(result.claims[0]?.citations[0] ?? null);
    } catch (error) {
      const message = error instanceof ApiError ? error.message : "The request failed.";
      setState({ phase: "failed", question: text, message });
    }
    resultRef.current?.focus();
  }

  const result = state.phase === "answered" ? state.result : null;
  const selectedSource = result?.sources.find((s) => s.sentence_id === selected) ?? null;

  return (
    <div className="page">
      <header className="masthead">
        <h1 className="wordmark">Citarilo</h1>
        <p className="tagline">Answers from your documents. Every claim points to the sentence it comes from.</p>
      </header>

      {health && !health.index_ready && (
        <p className="notice" role="status">
          The document index is not built yet. Run <code>citarilo ingest</code> on the server, then restart it.
        </p>
      )}
      {health && !health.key_configured && (
        <p className="notice" role="status">
          The server has no OpenAI API key. Set <code>OPENAI_API_KEY</code> in its environment and restart it.
        </p>
      )}

      <main className="workspace">
        <section className="column question-column" aria-label="Question and answer">
          <form className="ask" onSubmit={ask}>
            <label htmlFor="question">Your question</label>
            <textarea
              id="question"
              value={question}
              maxLength={500}
              rows={3}
              placeholder="For example: How long do I have to return an item bought on sale?"
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                  e.preventDefault();
                  e.currentTarget.form?.requestSubmit();
                }
              }}
            />
            <div className="ask-row">
              <span className="hint">Enter to ask, Shift+Enter for a new line</span>
              <button type="submit" disabled={state.phase === "asking" || question.trim().length < 3}>
                {state.phase === "asking" ? "Checking…" : "Ask"}
              </button>
            </div>
          </form>

          <section ref={resultRef} tabIndex={-1} className="result" aria-live="polite" aria-busy={state.phase === "asking"}>
            {state.phase === "asking" && <Progress />}
            {state.phase === "failed" && (
              <div className="outcome outcome-error" role="alert">
                <h2>The question could not be answered</h2>
                <p>{state.message}</p>
              </div>
            )}
            {result && result.status === "refused" && (
              <div className="outcome outcome-refused">
                <h2>{REFUSALS[result.refusal_code ?? ""]?.title ?? "No verified answer"}</h2>
                <p>{REFUSALS[result.refusal_code ?? ""]?.body}</p>
                <RunDetails result={result} />
              </div>
            )}
            {result && result.status === "answered" && (
              <Answer
                result={result}
                selected={selected}
                onSelect={(id) => {
                  setSelected(id);
                  setScrollRequest((n) => n + 1);
                }}
              />
            )}
          </section>
        </section>

        <DocumentPanel
          runId={result?.run_id ?? ""}
          source={selectedSource}
          citedIds={result?.sources.map((s) => s.sentence_id) ?? []}
          scrollRequest={scrollRequest}
        />
      </main>
    </div>
  );
}

function Progress() {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    const timer = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(timer);
  }, []);
  return (
    <p className="progress">
      Searching the documents and checking every claim against its sources… <span>{seconds} s</span>
    </p>
  );
}

function Answer({
  result,
  selected,
  onSelect,
}: {
  result: AskResponse;
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  // Number sources in order of first citation, as a reader meets them.
  const numbers = useMemo(() => {
    const order = new Map<string, number>();
    for (const claim of result.claims)
      for (const id of claim.citations) if (!order.has(id)) order.set(id, order.size + 1);
    return order;
  }, [result]);
  const sources = [...result.sources].sort(
    (a, b) => (numbers.get(a.sentence_id) ?? 99) - (numbers.get(b.sentence_id) ?? 99),
  );

  return (
    <article className="answer">
      <h2 className="visually-hidden">Answer</h2>
      <ul className="claims" lang={result.answer_language ?? undefined}>
        {result.claims.map((claim, index) => (
          <li key={index}>
            {claim.text}{" "}
            {claim.citations.map((id) => (
              <button
                key={id}
                type="button"
                className="cite"
                aria-pressed={selected === id}
                aria-label={`Show source ${numbers.get(id)}`}
                onClick={() => onSelect(id)}
              >
                {numbers.get(id)}
              </button>
            ))}
          </li>
        ))}
      </ul>

      <h3 className="sources-heading">Sources</h3>
      <ol className="sources">
        {sources.map((source) => (
          <li key={source.sentence_id} value={numbers.get(source.sentence_id)}>
            <button
              type="button"
              className="source"
              aria-pressed={selected === source.sentence_id}
              onClick={() => onSelect(source.sentence_id)}
            >
              <q>{source.text}</q>
              <span className="where">
                {source.doc_title}, section {source.heading}
              </span>
            </button>
          </li>
        ))}
      </ol>
      <RunDetails result={result} />
    </article>
  );
}
