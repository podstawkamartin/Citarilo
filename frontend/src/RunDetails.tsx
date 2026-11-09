import type { AskResponse } from "./api";

const STEP_NAMES: Record<string, string> = {
  retrieve: "Vyhledání relevantních sekcí",
  generate: "Napsání tvrzení podložených citacemi",
  check: "Kontrola čísel a citací kódem",
  verify: "Kontrola tvrzení druhým modelem",
  assemble: "Sestavení odpovědi",
  refuse: "Odmítnutí odpovědi",
};

// A step can finish without error yet reject the draft; the refusal code says which one did.
const FAILED_STEP: Record<string, string> = {
  check_failed: "check",
  unsupported: "verify",
  contradicted: "verify",
};

const cs = (value: number, digits: number) =>
  value.toLocaleString("cs-CZ", { minimumFractionDigits: digits, maximumFractionDigits: digits });

/** How the answer was checked: each step with its time, then total time and cost. */
export function RunDetails({ result }: { result: AskResponse }) {
  const { metrics } = result;
  const cost = Number(metrics.cost_usd);
  const rejectedBy = result.status === "refused" ? FAILED_STEP[result.refusal_code ?? ""] : undefined;
  return (
    <section className="panel p-amber" aria-label="Jak jsme to ověřili">
      <div className="checks">
        <ol>
          {metrics.node_timings.map((step, index) => (
            <li key={index} className={step.ok && step.node !== rejectedBy ? undefined : "failed"}>
              {STEP_NAMES[step.node] ?? step.node}
              {step.ok && step.node !== rejectedBy ? null : <span className="visually-hidden"> (neprošlo)</span>}
              <b>{cs(step.duration_ms / 1000, 2)} s</b>
            </li>
          ))}
        </ol>
        <div className="stats">
          <div className="stat">
            <b>{cs(metrics.latency_ms / 1000, 1)} s</b>
            <span>do odpovědi</span>
          </div>
          <div className="stat">
            <b className="mono">{cs(cost, cost < 0.01 ? 4 : 2)} $</b>
            <span>cena modelu</span>
          </div>
        </div>
        <p className="tokens">
          {metrics.input_tokens.toLocaleString("cs-CZ")} vstupních a {metrics.output_tokens.toLocaleString("cs-CZ")}{" "}
          výstupních tokenů · běh {result.run_id}
        </p>
      </div>
    </section>
  );
}
