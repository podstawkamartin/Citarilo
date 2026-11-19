import { useState } from "react";
import { type AskResponse, type RefusalSection, type Source } from "./api";
import { DocumentPanel } from "./DocumentPanel";
import { RunDetails } from "./RunDetails";

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

const STEPS: Record<string, string> = {
  no_evidence: "Návrh odmítlo vyhledání: v nalezených oddílech není dost podkladů.",
  deterministic_check: "Návrh odmítla kontrola v kódu.",
  verifier_unsupported: "Návrh odmítl druhý model, protože citace tvrzení nepodporují.",
  verifier_contradicted: "Návrh odmítl druhý model, protože citace jsou s tvrzením v rozporu.",
};

const CHECKS: Record<string, string> = {
  invalid_citation_count: "počet citací",
  unknown_citation: "neznámá citace",
  literal_not_in_citations: "číslo, datum nebo odkaz mimo citace",
  language_mismatch: "jazyk odpovědi",
};

function sectionSource(section: RefusalSection): Source {
  return {
    sentence_id: "",
    text: "",
    doc_id: section.doc_id,
    doc_title: section.doc_title,
    heading: section.heading,
  };
}

/** A refusal: the fixed explanation, the safe step detail, and the sections to open. */
export function Refusal({ result, setId }: { result: AskResponse; setId: string | null }) {
  const explanation = result.refusal ?? null;
  const [openKey, setOpenKey] = useState<string | null>(explanation?.sections[0]?.section_key ?? null);
  const open = explanation?.sections.find((section) => section.section_key === openKey) ?? null;
  const copy = REFUSALS[result.refusal_code ?? ""];
  const checkName = explanation?.failed_check ? CHECKS[explanation.failed_check] : undefined;

  return (
    <div>
      <div className="board">
        <div>
          <p className="eyebrow col-label l-soft">Odpověď</p>
          <section className="panel outcome">
            <h2>{copy?.title ?? "Bez ověřené odpovědi"}</h2>
            <p>{copy?.body}</p>
            {explanation && (
              <div className="refusal-detail">
                <p>{STEPS[explanation.step]}</p>
                {explanation.rejected_claims > 0 && <p>Zamítnutá tvrzení: {explanation.rejected_claims}</p>}
                {checkName && <p>Neprošla kontrola: {checkName}</p>}
                {explanation.sections.length > 0 && (
                  <>
                    <p className="sections-label">Podívejte se do těchto oddílů</p>
                    <ul className="sections">
                      {explanation.sections.map((section) => {
                        const on = section.section_key === openKey;
                        return (
                          <li key={section.section_key}>
                            <button
                              type="button"
                              aria-pressed={on}
                              onClick={() => setOpenKey(section.section_key)}
                            >
                              <b>{section.doc_title}</b>
                              <span>
                                {" "}
                                · {section.heading}
                              </span>
                            </button>
                          </li>
                        );
                      })}
                    </ul>
                  </>
                )}
              </div>
            )}
          </section>
        </div>
        <div>
          <p className="eyebrow col-label l-amber">Jak jsme to ověřili</p>
          <RunDetails result={result} />
        </div>
      </div>
      {open && (
        <DocumentPanel
          runId={result.run_id}
          source={sectionSource(open)}
          citedIds={[]}
          setId={setId}
          sectionKey={open.section_key}
        />
      )}
    </div>
  );
}
