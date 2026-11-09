import { useEffect, useRef, useState } from "react";
import { type Link, useThreads } from "./threads";

// When each stage typically starts, in ms. The API answers in one response, so these only
// pace the animation; real step times replace them once the answer arrives.
const STAGES = [
  { at: 0, step: "retrieve", text: "Hledám v dokumentech relevantní sekce" },
  { at: 700, step: "generate", text: "Píšu tvrzení a ke každému vybírám citace" },
  { at: 2600, step: "check", text: "Kód kontroluje čísla a citace" },
  { at: 3000, step: "verify", text: "Druhý model ověřuje, že citace tvrzení podporují" },
];

const STEP_NAMES: Record<string, string> = {
  retrieve: "Vyhledání relevantních sekcí",
  generate: "Napsání tvrzení podložených citacemi",
  check: "Kontrola čísel a citací kódem",
  verify: "Kontrola tvrzení druhým modelem",
};

const LINKS: Link[] = [
  { from: "sk-claim-0", to: "sk-cite-0", group: 0 },
  { from: "sk-claim-0", to: "sk-cite-1", group: 0 },
  { from: "sk-claim-1", to: "sk-cite-2", group: 1 },
];

const seconds = (ms: number) =>
  (ms / 1000).toLocaleString("cs-CZ", { minimumFractionDigits: 1, maximumFractionDigits: 1 });

/** The answer being put together: the same layout fills in stage by stage while a timer runs. */
export function Assembling() {
  const [elapsed, setElapsed] = useState(0);
  const board = useRef<HTMLDivElement>(null);
  const threads = useThreads(board, LINKS);

  useEffect(() => {
    const start = performance.now();
    const timer = setInterval(() => setElapsed(performance.now() - start), 100);
    return () => clearInterval(timer);
  }, []);

  const stage = STAGES.reduce((current, s, index) => (elapsed >= s.at ? index : current), 0);
  const reached = (index: number) => (stage >= index ? " go" : "");

  return (
    <div className="assembling">
      <p className="stage-line" role="status">
        <span className="pulse" aria-hidden="true" />
        {STAGES[stage].text}
        <span className="timer" aria-hidden="true">
          {seconds(elapsed)} s
        </span>
      </p>

      <div className="board" ref={board} aria-hidden="true">
        <div>
          <p className="eyebrow col-label l-violet">Odpověď</p>
          <section className={`panel p-violet sk-panel${reached(1)}`}>
            {[0, 1].map((i) => (
              <div key={i} className={`card claim sk-claim c${i}`}>
                <span className="bar eyebrow-bar" />
                <span className="bar w1" />
                <span className="bar w2" />
                <span className="port" data-port={`sk-claim-${i}`} />
              </div>
            ))}
          </section>

          <p className="eyebrow col-label l-amber gap">Jak jsme to ověřili</p>
          <section className="panel p-amber">
            <div className="checks">
              <ol>
                {STAGES.map((s, index) => {
                  // Only the total time is measured; steps show where the animation is, not timings.
                  const state = index < stage ? "past" : index === stage ? "active" : "pending";
                  return (
                    <li key={s.step} className={`sk-step ${state}`}>
                      {STEP_NAMES[s.step]}
                      <b>{state === "active" ? "probíhá" : state === "pending" ? "…" : ""}</b>
                    </li>
                  );
                })}
              </ol>
              <div className="stats">
                <div className="stat">
                  <b>{seconds(elapsed)} s</b>
                  <span>zatím</span>
                </div>
                <div className="stat">
                  <b className="mono">… $</b>
                  <span>cena se počítá</span>
                </div>
              </div>
            </div>
          </section>
        </div>

        <div>
          <p className="eyebrow col-label l-green">Citace z dokumentu</p>
          <section className={`panel p-green sk-panel${reached(0)}`}>
            <span className="bar src-bar" />
            {[0, 1, 2].map((i) => (
              <div key={i} className={`card cite sk-cite c${i}`}>
                <span className="bar w3" />
                <span className="bar w4" />
                <span className="port" data-port={`sk-cite-${i}`} />
              </div>
            ))}
          </section>

          <p className="eyebrow col-label l-soft gap">Zdrojový dokument</p>
          <div className={`paper sk-paper${reached(0)}`}>
            <div className="paper-body">
              <span className="bar title-bar" />
              {[1, 2, 3, 4, 5, 6].map((i) => (
                <span key={i} className={`bar line l${i}`} />
              ))}
              <span className="scan" />
            </div>
          </div>
        </div>

        <svg className={`threads sk-threads${reached(1)}`}>
          {threads.map((t, i) => (
            <path key={t.key} d={t.d} pathLength={1} className={`t${i}`} />
          ))}
        </svg>
      </div>
    </div>
  );
}
