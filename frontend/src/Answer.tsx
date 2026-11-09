import { useMemo, useRef, useState } from "react";
import type { AskResponse, Source } from "./api";
import { plural } from "./format";
import { DocumentPanel } from "./DocumentPanel";
import { RunDetails } from "./RunDetails";
import { type Link, useThreads } from "./threads";

// Lets a claim name its citations for assistive technology, where the threads are not seen.
const citeId = (sentenceId: string) => `cite-${sentenceId.replace(/[^\w-]/g, "_")}`;

/** An answered question: claims, their citations, the source document and how it was checked. */
export function Answer({ result }: { result: AskResponse }) {
  const [selected, setSelected] = useState<string | null>(result.claims[0]?.citations[0] ?? null);
  const [focused, setFocused] = useState<number | null>(null);
  const board = useRef<HTMLDivElement>(null);
  const links = useMemo<Link[]>(
    () =>
      result.claims.flatMap((claim, index) =>
        claim.citations.map((id) => ({ from: `claim-${index}`, to: `cite-${id}`, group: index })),
      ),
    [result],
  );
  const threads = useThreads(board, links);

  // Citations in the order a reader meets them, grouped by document and section.
  const groups = useMemo(() => {
    const byId = new Map(result.sources.map((s) => [s.sentence_id, s]));
    const order: Source[] = [];
    for (const claim of result.claims)
      for (const id of claim.citations) {
        const source = byId.get(id);
        if (source && !order.includes(source)) order.push(source);
      }
    const grouped = new Map<string, Source[]>();
    for (const source of order) {
      const key = `${source.doc_id}\u0000${source.heading}`;
      grouped.set(key, [...(grouped.get(key) ?? []), source]);
    }
    return [...grouped.values()];
  }, [result]);

  const focusedIds = focused === null ? null : new Set(result.claims[focused]?.citations);
  const selectedSource = result.sources.find((s) => s.sentence_id === selected) ?? null;
  const citedIds = useMemo(() => result.sources.map((s) => s.sentence_id), [result]);

  return (
    <div className="board" ref={board}>
      <div>
        <p className="eyebrow col-label l-violet">Odpověď</p>
        <section className="panel p-violet" aria-label="Odpověď">
          {result.claims.map((claim, index) => {
            const n = claim.citations.length;
            return (
              <article
                key={index}
                className={focused !== null && focused !== index ? "card claim dim" : "card claim"}
                tabIndex={0}
                data-claim={index}
                onMouseEnter={() => setFocused(index)}
                onMouseLeave={() => setFocused(null)}
                onFocus={() => setFocused(index)}
                onBlur={() => setFocused(null)}
                aria-describedby={claim.citations.map(citeId).join(" ")}
              >
                <span className="eyebrow">
                  Podloženo {n} {plural(n, "citací", "citacemi", "citacemi")}
                </span>
                <p lang={result.answer_language ?? undefined}>{claim.text}</p>
                <span className="port" data-port={`claim-${index}`} aria-hidden="true" />
              </article>
            );
          })}
        </section>

        <p className="eyebrow col-label l-amber gap">Jak jsme to ověřili</p>
        <RunDetails result={result} />
      </div>

      <div>
        <p className="eyebrow col-label l-green">Citace z dokumentu</p>
        <section className="panel p-green" aria-label="Citace z dokumentu">
          {groups.map((group) => (
            <div className="group" key={`${group[0].doc_id}:${group[0].heading}`}>
              <p className="src">
                <b>{group[0].doc_title}</b> · {group[0].heading}
              </p>
              {group.map((source) => {
                const on = source.sentence_id === selected;
                const dim = focusedIds !== null && !focusedIds.has(source.sentence_id);
                return (
                  <button
                    key={source.sentence_id}
                    type="button"
                    className={dim ? "card cite dim" : "card cite"}
                    aria-pressed={on}
                    onClick={() => setSelected(source.sentence_id)}
                  >
                    {on && <span className="here">Zobrazeno níže</span>}
                    <q lang="cs" id={citeId(source.sentence_id)}>
                      {source.text}
                    </q>
                    <span className="port" data-port={`cite-${source.sentence_id}`} aria-hidden="true" />
                  </button>
                );
              })}
            </div>
          ))}
        </section>

        <p className="eyebrow col-label l-soft gap">Zdrojový dokument</p>
        <DocumentPanel runId={result.run_id} source={selectedSource} citedIds={citedIds} />
      </div>

      <svg className="threads" aria-hidden="true">
        {threads.map((t) => (
          <path key={t.key} d={t.d} className={focused !== null && focused !== t.group ? "dim" : undefined} />
        ))}
      </svg>
    </div>
  );
}
