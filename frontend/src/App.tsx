import { useEffect, useRef, useState } from "react";
import { type AskResponse, type DocumentSetInfo, type Health, ApiError, api } from "./api";
import { Answer } from "./Answer";
import { Assembling } from "./Assembling";
import { plural } from "./format";
import { Help } from "./Help";
import { NoteForm } from "./Note";
import { Refusal } from "./Refusal";
import { ThemeToggle } from "./ThemeToggle";

const SET_KEY = "citarilo-document-set";

const EXAMPLES: Record<string, string[]> = {
  trailsupply: ["Do kdy můžu vrátit zboží z výprodeje?", "Jak reklamovat poškozené zboží?"],
  regulations: [
    "Jak často se reviduje elektroinstalace ve výrobní hale?",
    "Co dělat, když se pásový dopravník zastaví?",
  ],
};

function storedSet(): string | null {
  try {
    return localStorage.getItem(SET_KEY);
  } catch {
    return null;
  }
}

function rememberSet(id: string) {
  try {
    localStorage.setItem(SET_KEY, id);
  } catch {
    // Storage can be unavailable; the choice still applies on this page.
  }
}

type State =
  | { phase: "idle" }
  | { phase: "asking"; question: string }
  | { phase: "answered"; question: string; result: AskResponse }
  | { phase: "failed"; question: string; message: string };

export function App() {
  const [question, setQuestion] = useState("");
  const [state, setState] = useState<State>({ phase: "idle" });
  const [health, setHealth] = useState<Health | null>(null);
  const [sets, setSets] = useState<DocumentSetInfo[]>([]);
  const [setId, setSetId] = useState<string | null>(null);
  const resultRef = useRef<HTMLElement>(null);
  const requestId = useRef(0);

  useEffect(() => {
    api.health().then(setHealth, () => setHealth(null));
    api.sets().then(
      (list) => {
        setSets(list);
        const saved = storedSet();
        const chosen = list.find((item) => item.id === saved) ?? list[0];
        if (chosen) setSetId(chosen.id);
      },
      () => setSets([]),
    );
  }, []);

  function chooseSet(id: string) {
    if (id === setId) return;
    requestId.current += 1;
    setSetId(id);
    rememberSet(id);
    setQuestion("");
    setState({ phase: "idle" });
  }

  async function ask(event: React.FormEvent) {
    event.preventDefault();
    const text = question.trim();
    if (text.length < 3 || state.phase === "asking") return;
    const id = ++requestId.current;
    setState({ phase: "asking", question: text });
    try {
      const result = await api.ask(text, setId ?? undefined);
      if (id !== requestId.current) return;
      setState({ phase: "answered", question: text, result });
    } catch (error) {
      if (id !== requestId.current) return;
      const message = error instanceof ApiError ? error.message : "Požadavek se nepodařil.";
      setState({ phase: "failed", question: text, message });
    }
    resultRef.current?.focus();
  }

  const result = state.phase === "answered" ? state.result : null;
  const asking = state.phase === "asking";
  const selected = sets.find((item) => item.id === setId) ?? null;
  const examples = setId ? (EXAMPLES[setId] ?? []) : [];

  return (
    <div className="page">
      <header className="top">
        <h1 className="brand">
          <svg viewBox="0 0 33 33" aria-hidden="true">
            <rect className="m-blue" width="13" height="13" rx="3" />
            <rect className="m-violet" x="10" y="10" width="13" height="13" rx="3" />
            <rect className="m-green" x="20" y="20" width="13" height="13" rx="3" />
          </svg>
          Citarilo
        </h1>
        {setId && sets.length > 0 && (
          <label className="set-switch" htmlFor="document-set">
            Dokumenty:
            <select id="document-set" value={setId} onChange={(event) => chooseSet(event.target.value)}>
              {sets.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.title}
                  {item.ready ? "" : " (nedostupné)"}
                </option>
              ))}
            </select>
          </label>
        )}
        <Status health={health} selected={sets.length > 1 ? selected : null} />
        <ThemeToggle />
      </header>

      {health && !health.index_ready && (
        <p className="notice" role="status">
          Index dokumentů ještě není sestavený. Na serveru spusťte <code>citarilo ingest</code> a server restartujte.
        </p>
      )}
      {selected && !selected.ready && health?.index_ready && (
        <p className="notice" role="status">
          Vybraná sada dokumentů teď není dostupná. Ostatní sady lze dál používat.
        </p>
      )}
      {health && !health.key_configured && (
        <p className="notice" role="status">
          Server nemá klíč k OpenAI API. Nastavte <code>OPENAI_API_KEY</code> v jeho prostředí a restartujte ho.
        </p>
      )}

      <form className="ask" onSubmit={ask}>
        <div>
          <div className="ask-head">
            <label className="eyebrow l-blue" htmlFor="question">
              Vaše otázka
            </label>
            <span className="hint">Enter odešle · Shift+Enter nový řádek</span>
          </div>
          <textarea
            id="question"
            value={question}
            maxLength={500}
            rows={1}
            placeholder={examples[0] ?? "Na co se chcete zeptat?"}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                e.currentTarget.form?.requestSubmit();
              }
            }}
          />
        </div>
        <button
          type="submit"
          className={asking ? "send busy" : "send"}
          aria-label={asking ? "Ověřuji odpověď" : "Zeptat se"}
          title="Zeptat se"
          disabled={asking || question.trim().length < 3}
        >
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M12 19V5M5.5 11.5 12 5l6.5 6.5" />
          </svg>
        </button>
      </form>

      <main ref={resultRef} tabIndex={-1} className="result" aria-live="polite" aria-busy={asking}>
        {state.phase === "idle" && (
          <>
            <p className="empty">
              Zeptejte se na cokoli z firemních dokumentů. Každé tvrzení v odpovědi bude podložené doslovnými citacemi
              a uvidíte je přímo v dokumentu.
            </p>
            {examples.length > 0 && (
              <div className="examples">
                {examples.map((sample) => (
                  <button key={sample} type="button" onClick={() => setQuestion(sample)}>
                    {sample}
                  </button>
                ))}
              </div>
            )}
          </>
        )}
        {asking && <Assembling />}
        {state.phase === "failed" && (
          <section className="panel p-red outcome" role="alert">
            <h2>Na otázku se nepodařilo odpovědět</h2>
            <p>{state.message}</p>
          </section>
        )}
        {result && result.status === "refused" && <Refusal result={result} setId={setId} />}
        {result && result.status === "answered" && (
          <Answer key={`answer-${result.run_id}`} result={result} setId={setId} />
        )}
        {result && <NoteForm key={`note-${result.run_id}`} runId={result.run_id} />}
      </main>

      <Help />
    </div>
  );
}

function Status({ health, selected }: { health: Health | null; selected: DocumentSetInfo | null }) {
  if (!health) return null;
  const ready = selected ? selected.ready && health.status === "ok" : health.status === "ok";
  let label = ready ? "Připraveno" : "Není připraveno";
  if (selected) {
    const d = selected.document_count;
    label = `${d} ${plural(d, "dokument", "dokumenty", "dokumentů")}`;
  } else if (health.index_ready && health.document_count !== undefined && health.sentence_count !== undefined) {
    const d = health.document_count;
    const s = health.sentence_count;
    label = `${d} ${plural(d, "dokument", "dokumenty", "dokumentů")} · ${s} ${plural(s, "věta", "věty", "vět")}`;
  }
  return <p className={ready ? "status" : "status not-ready"}>{label}</p>;
}
