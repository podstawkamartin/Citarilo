import { useState } from "react";
import { type FeedbackLabel, ApiError, api } from "./api";

const LABELS: { value: FeedbackLabel | ""; text: string }[] = [
  { value: "", text: "Bez štítku" },
  { value: "wrongly_refused", text: "Špatně odmítnuto" },
  { value: "wrong_or_dangerous", text: "Chybná nebo nebezpečná odpověď" },
  { value: "missing_step", text: "Chybí důležitý krok" },
  { value: "other", text: "Jiné" },
];

/** A short local note stored against one run. The run record itself is not changed. */
export function NoteForm({ runId }: { runId: string }) {
  const [note, setNote] = useState("");
  const [label, setLabel] = useState<FeedbackLabel | "">("");
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const noteId = `note-${runId}`;
  const labelId = `note-label-${runId}`;

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (pending || (!note.trim() && !label)) return;
    setPending(true);
    setError(null);
    try {
      await api.feedback(runId, { note: note.trim(), ...(label ? { label } : {}) });
      setNote("");
      setLabel("");
      setSaved(true);
    } catch (caught) {
      setSaved(false);
      setError(caught instanceof ApiError ? caught.message : "Poznámku se nepodařilo uložit.");
    } finally {
      setPending(false);
    }
  }

  return (
    <form className="note" onSubmit={submit}>
      <label htmlFor={noteId}>Poznámka</label>
      <textarea
        id={noteId}
        value={note}
        maxLength={1000}
        rows={3}
        onChange={(event) => setNote(event.target.value)}
      />
      <div className="note-row">
        <label htmlFor={labelId}>Štítek</label>
        <select id={labelId} value={label} onChange={(event) => setLabel(event.target.value as FeedbackLabel | "")}>
          {LABELS.map((item) => (
            <option key={item.value || "none"} value={item.value}>
              {item.text}
            </option>
          ))}
        </select>
        <button type="submit" disabled={pending || (!note.trim() && !label)}>
          Uložit poznámku
        </button>
      </div>
      {saved && <p role="status">Poznámka je uložená.</p>}
      {error && <p role="alert">{error}</p>}
    </form>
  );
}
