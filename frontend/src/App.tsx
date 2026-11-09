import { useEffect, useRef, useState } from "react";
import { type AskResponse, type Health, ApiError, api } from "./api";
import { Answer } from "./Answer";
import { Assembling } from "./Assembling";
import { plural } from "./format";
import { Help } from "./Help";
import { RunDetails } from "./RunDetails";
import { ThemeToggle } from "./ThemeToggle";

const REFUSALS: Record<string, { title: string; body: string }> = {
  insufficient_evidence: {
    title: "Dokumenty na tohle neodpovídají",
    body: "Žádný dokument tuto otázku nepokrývá, proto odpověď nevznikla. Zkuste otázku přeformulovat, nebo se zeptejte na téma, které dokumenty pokrývají.",
  },
  check_failed: {
    title: "Bez ověřené odpovědi",
    body: "Návrh odpovědi obsahoval čísla nebo odkazy, které v citacích nejsou, proto ho nezobrazujeme.",
  },
  unsupported: {
    title: "Bez ověřené odpovědi",
    body: "Návrh odpovědi obsahoval tvrzení, které jeho citace nepodporují, proto ho nezobrazujeme.",
  },
  contradicted: {
    title: "Bez ověřené odpovědi",
    body: "Návrh odpovědi odporoval svým citacím, proto ho nezobrazujeme.",
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
  const resultRef = useRef<HTMLElement>(null);

  useEffect(() => {
    api.health().then(setHealth, () => setHealth(null));
  }, []);

  async function ask(event: React.FormEvent) {
    event.preventDefault();
    const text = question.trim();
    if (text.length < 3 || state.phase === "asking") return;
    setState({ phase: "asking", question: text });
    try {
      const result = await api.ask(text);
      setState({ phase: "answered", question: text, result });
    } catch (error) {
      const message = error instanceof ApiError ? error.message : "Požadavek se nepodařil.";
      setState({ phase: "failed", question: text, message });
    }
    resultRef.current?.focus();
  }

  const result = state.phase === "answered" ? state.result : null;
  const asking = state.phase === "asking";

  return (
    <div className="page">
      <header className="top">
        <h1 className="brand">Citarilo</h1>
        <Status health={health} />
        <ThemeToggle />
      </header>

      {health && !health.index_ready && (
        <p className="notice" role="status">
          Index dokumentů ještě není sestavený. Na serveru spusťte <code>citarilo ingest</code> a server restartujte.
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
            placeholder="Na co se chcete zeptat? Třeba: Do kdy můžu vrátit zboží z výprodeje?"
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
          <p className="empty">
            Zeptejte se na cokoli z firemních dokumentů. Každé tvrzení v odpovědi bude podložené doslovnými citacemi
            a uvidíte je přímo v dokumentu.
          </p>
        )}
        {asking && <Assembling />}
        {state.phase === "failed" && (
          <section className="panel p-red outcome" role="alert">
            <h2>Na otázku se nepodařilo odpovědět</h2>
            <p>{state.message}</p>
          </section>
        )}
        {result && result.status === "refused" && (
          <div className="board">
            <div>
              <p className="eyebrow col-label l-soft">Odpověď</p>
              <section className="panel outcome">
                <h2>{REFUSALS[result.refusal_code ?? ""]?.title ?? "Bez ověřené odpovědi"}</h2>
                <p>{REFUSALS[result.refusal_code ?? ""]?.body}</p>
              </section>
            </div>
            <div>
              <p className="eyebrow col-label l-amber">Jak jsme to ověřili</p>
              <RunDetails result={result} />
            </div>
          </div>
        )}
        {result && result.status === "answered" && <Answer key={result.run_id} result={result} />}
      </main>

      <Help />
    </div>
  );
}

function Status({ health }: { health: Health | null }) {
  if (!health) return null;
  const ready = health.status === "ok";
  let label = ready ? "Připraveno" : "Není připraveno";
  if (health.index_ready && health.document_count !== undefined && health.sentence_count !== undefined) {
    const d = health.document_count;
    const s = health.sentence_count;
    label = `${d} ${plural(d, "dokument", "dokumenty", "dokumentů")} · ${s} ${plural(s, "věta", "věty", "vět")}`;
  }
  return <p className={ready ? "status" : "status not-ready"}>{label}</p>;
}
