import type { AskResponse } from "./api";

const STEP_NAMES: Record<string, string> = {
  retrieve: "Search documents",
  generate: "Draft answer",
  check: "Check figures and citations",
  verify: "Verify claims against sources",
  assemble: "Assemble answer",
  refuse: "Decline",
};

const seconds = (ms: number) => `${(ms / 1000).toFixed(1)} s`;

/** Time, cost and steps of one run, for people who want to audit it. */
export function RunDetails({ result }: { result: AskResponse }) {
  const { metrics } = result;
  const cost = Number(metrics.cost_usd);
  return (
    <details className="run-details">
      <summary>
        Answered in {seconds(metrics.latency_ms)} for ${cost < 0.01 ? cost.toFixed(4) : cost.toFixed(2)}
      </summary>
      <ol className="steps">
        {metrics.node_timings.map((step, index) => (
          <li key={index} className={step.ok ? undefined : "failed"}>
            <span>{STEP_NAMES[step.node] ?? step.node}</span>
            <span>{seconds(step.duration_ms)}</span>
          </li>
        ))}
      </ol>
      <p className="tokens">
        {metrics.input_tokens.toLocaleString("en")} input and {metrics.output_tokens.toLocaleString("en")} output
        tokens. Run {result.run_id}.
      </p>
    </details>
  );
}
