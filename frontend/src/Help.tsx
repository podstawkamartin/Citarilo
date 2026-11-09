import { useEffect, useRef, useState } from "react";

const ROWS = [
  {
    colour: "c-blue",
    title: "Vaše otázka",
    text: "Sem napíšete dotaz a odešlete ho šipkou. Pod ním se ukáže odpověď.",
  },
  {
    colour: "c-violet",
    title: "Odpověď",
    text: "Napsal ji jazykový model. Každé tvrzení je čarou spojené s citacemi, které ho podkládají.",
  },
  {
    colour: "c-green",
    title: "Citace z dokumentu",
    text: "Doslovný text z dokumentu, beze změny. Kliknutím ho uvidíte v celém dokumentu.",
  },
  {
    colour: "c-soft",
    title: "Zdrojový dokument",
    text: "Původní text dokumentu, jazykový model ho neupravuje. Citace jsou v něm zvýrazněné.",
    paper: true,
  },
  {
    colour: "c-amber",
    title: "Jak jsme to ověřili",
    text: "Před zobrazením kód zkontroloval citace a čísla a druhý model potvrdil, že citace tvrzení opravdu podporují.",
  },
];

/** The "?" button in the corner: explains what each colour on the page means. */
export function Help() {
  const [open, setOpen] = useState(false);
  const button = useRef<HTMLButtonElement>(null);
  const panel = useRef<HTMLElement>(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setOpen(false);
        button.current?.focus();
      }
    };
    const onClick = (e: MouseEvent) => {
      const target = e.target as Node;
      if (!panel.current?.contains(target) && !button.current?.contains(target)) setOpen(false);
    };
    document.addEventListener("keydown", onKey);
    document.addEventListener("click", onClick);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("click", onClick);
    };
  }, [open]);

  return (
    <>
      <button
        ref={button}
        type="button"
        className="help"
        aria-expanded={open}
        aria-controls="help-panel"
        onClick={() => setOpen((o) => !o)}
      >
        <b aria-hidden="true">?</b>Nápověda
        <span className="dots" aria-hidden="true">
          <i className="c-blue" />
          <i className="c-violet" />
          <i className="c-green" />
          <i className="c-amber" />
          <i className="paper-dot" />
        </span>
      </button>
      {open && (
        <aside ref={panel} id="help-panel" className="legend" aria-labelledby="help-title">
          <h2 id="help-title">Co je co</h2>
          <p>Barva říká, odkud text pochází.</p>
          <ul>
            {ROWS.map((row) => (
              <li key={row.title} className={`row ${row.colour}`}>
                <span className={row.paper ? "sw paper-sw" : "sw"} aria-hidden="true" />
                <div>
                  <strong>{row.title}</strong>
                  <p>{row.text}</p>
                </div>
              </li>
            ))}
          </ul>
        </aside>
      )}
    </>
  );
}
