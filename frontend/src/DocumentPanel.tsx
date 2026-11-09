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

/** The original document as a sheet of paper, with the selected citation highlighted in place. */
export function DocumentPanel({
  runId,
  source,
  citedIds,
}: {
  runId: string;
  source: Source | null;
  citedIds: string[];
}) {
  const [doc, setDoc] = useState<{ id: string; view: DocumentView } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const body = useRef<HTMLDivElement>(null);
  const marked = useRef<HTMLElement>(null);

  useEffect(() => {
    if (!source) return;
    let current = true;
    setError(null);
    loadDocument(runId, source.doc_id).then(
      (view) => current && setDoc({ id: `${runId}:${source.doc_id}`, view }),
      (e) => current && setError(e instanceof ApiError ? e.message : "Dokument se nepodařilo načíst."),
    );
    return () => {
      current = false;
    };
  }, [runId, source]);

  // Scroll only inside the sheet, so choosing a citation never moves the page.
  useEffect(() => {
    const container = body.current;
    const mark = marked.current;
    if (!container || !mark) return;
    const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    container.scrollTo?.({ top: Math.max(0, mark.offsetTop - 48), behavior: reduced ? "auto" : "smooth" });
  }, [doc, source]);

  if (!source) return null;
  const view = doc?.id === `${runId}:${source.doc_id}` ? doc.view : null;

  return (
    <article className="paper" aria-labelledby="doc-title" lang="cs">
      <header>
        <h2 id="doc-title">{source.doc_title}</h2>
        <p className="meta">{view?.file_name ?? "\u00a0"}</p>
      </header>
      {error && (
        <p className="loading" role="alert">
          {error}
        </p>
      )}
      {!view && !error && <p className="loading">Otevírám dokument…</p>}
      {view && (
        <div className="paper-body" ref={body} tabIndex={0} role="region" aria-label="Text dokumentu">
          {view.sections.map((section) => (
            <section key={section.key}>
              <h3>{section.heading}</h3>
              <p className="text">
                {section.sentences.map((sentence) => {
                  const isSelected = sentence.id === source.sentence_id;
                  const isCited = citedIds.includes(sentence.id);
                  return (
                    <span key={sentence.id}>
                      {isSelected ? (
                        <mark ref={marked} className="on">
                          {sentence.text}
                        </mark>
                      ) : isCited ? (
                        <mark>{sentence.text}</mark>
                      ) : (
                        sentence.text
                      )}{" "}
                    </span>
                  );
                })}
              </p>
            </section>
          ))}
        </div>
      )}
    </article>
  );
}
