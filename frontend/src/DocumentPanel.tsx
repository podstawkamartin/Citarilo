import { useEffect, useRef, useState } from "react";
import { type DocumentView, type Source, ApiError, api } from "./api";

// Scoped to one answer: after a corpus update, a new answer must show the new text.
const cache = new Map<string, Promise<DocumentView>>();
let cachedRun = "";

function loadDocument(runId: string, docId: string): Promise<DocumentView> {
  if (runId !== cachedRun) {
    cache.clear();
    cachedRun = runId;
  }
  let pending = cache.get(docId);
  if (!pending) {
    pending = api.document(docId);
    pending.catch(() => cache.delete(docId));
    cache.set(docId, pending);
  }
  return pending;
}

/** Shows the cited sentence highlighted in its full document, in reading context. */
export function DocumentPanel({
  runId,
  source,
  citedIds,
  scrollRequest,
}: {
  runId: string;
  source: Source | null;
  citedIds: string[];
  scrollRequest: number;
}) {
  const [doc, setDoc] = useState<{ id: string; view: DocumentView } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const marked = useRef<HTMLElement>(null);

  useEffect(() => {
    if (!source) return;
    let current = true;
    setError(null);
    loadDocument(runId, source.doc_id).then(
      (view) => current && setDoc({ id: `${runId}:${source.doc_id}`, view }),
      (e) => current && setError(e instanceof ApiError ? e.message : "The document could not be loaded."),
    );
    return () => {
      current = false;
    };
  }, [runId, source]);

  useEffect(() => {
    if (scrollRequest === 0) return;
    const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    marked.current?.scrollIntoView?.({ block: "nearest", behavior: reduced ? "auto" : "smooth" });
  }, [doc, scrollRequest]);

  if (!source) {
    return (
      <aside className="column document document-empty" aria-label="Source document">
        <p>Select a citation to read it in its document.</p>
      </aside>
    );
  }
  const view = doc?.id === `${runId}:${source.doc_id}` ? doc.view : null;

  return (
    <aside className="column document" aria-label="Source document">
      <header className="document-header">
        <h2>{source.doc_title}</h2>
      </header>
      {error && <p role="alert">{error}</p>}
      {!view && !error && <p className="loading">Opening the document…</p>}
      {view && (
        <div className="document-body">
          {view.sections.map((section) => (
            <section key={section.key}>
              <h3>{section.heading}</h3>
              <p>
                {section.sentences.map((sentence) => {
                  const isSelected = sentence.id === source.sentence_id;
                  const isCited = citedIds.includes(sentence.id);
                  return (
                    <span key={sentence.id}>
                      {isSelected ? (
                        <mark ref={marked} className="selected">
                          {sentence.text}
                        </mark>
                      ) : (
                        <span className={isCited ? "cited" : undefined}>{sentence.text}</span>
                      )}{" "}
                    </span>
                  );
                })}
              </p>
            </section>
          ))}
        </div>
      )}
    </aside>
  );
}
