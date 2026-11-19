/** Typed client for the Citarilo HTTP API (same origin; see src/citarilo/api.py). */

export type Source = {
  sentence_id: string;
  text: string;
  doc_id: string;
  doc_title: string;
  heading: string;
};

export type Claim = { text: string; citations: string[] };

export type NodeTiming = { node: string; duration_ms: number; ok: boolean };

export type RefusalSection = {
  doc_id: string;
  doc_title: string;
  heading: string;
  section_key: string;
};

export type CheckName =
  | "invalid_citation_count"
  | "unknown_citation"
  | "literal_not_in_citations"
  | "language_mismatch";

export type RefusalExplanation = {
  step: "no_evidence" | "deterministic_check" | "verifier_unsupported" | "verifier_contradicted";
  rejected_claims: number;
  failed_check: CheckName | null;
  sections: RefusalSection[];
};

export type DocumentSetInfo = {
  id: string;
  title: string;
  document_count: number;
  ready: boolean;
};

export type FeedbackLabel = "wrongly_refused" | "wrong_or_dangerous" | "missing_step" | "other";

export type AskResponse = {
  run_id: string;
  mode: string;
  status: "answered" | "refused";
  answer_language: "en" | "cs" | null;
  refusal_code: "insufficient_evidence" | "check_failed" | "unsupported" | "contradicted" | null;
  refusal?: RefusalExplanation | null;
  claims: Claim[];
  sources: Source[];
  metrics: {
    latency_ms: number;
    input_tokens: number;
    output_tokens: number;
    // Decimals are serialized as strings to keep exact values.
    cost_usd: string | number;
    node_timings: NodeTiming[];
  };
};

export type Health = {
  status: "ok" | "not_ready";
  index_ready: boolean;
  key_configured: boolean;
  // Absent on servers older than these fields.
  document_count?: number;
  sentence_count?: number;
};

export type DocumentView = {
  title: string;
  file_name: string;
  sections: { heading: string; key: string; sentences: { id: string; text: string }[] }[];
};

/** An API failure carrying the server's user-facing message. */
export class ApiError extends Error {
  constructor(
    readonly code: string,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, { ...init, headers: { Accept: "application/json", ...init?.headers } });
  } catch {
    throw new ApiError("network_error", "Server není dostupný. Zkontrolujte, že běží citarilo serve.");
  }
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const error = body?.error;
    throw new ApiError(
      typeof error?.code === "string" ? error.code : "http_error",
      typeof error?.message === "string" ? error.message : `Server odpověděl chybou ${response.status}.`,
    );
  }
  return body as T;
}

export const api = {
  ask: (question: string, setId?: string) =>
    request<AskResponse>("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(setId ? { question, set: setId } : { question }),
    }),
  health: () => request<Health>("/api/health"),
  sets: () => request<DocumentSetInfo[]>("/api/sets"),
  document: (docId: string, setId?: string) => {
    const query = setId ? `?set=${encodeURIComponent(setId)}` : "";
    return request<DocumentView>(`/api/documents/${encodeURIComponent(docId)}${query}`);
  },
  feedback: (runId: string, body: { note: string; label?: FeedbackLabel }) =>
    request<{ saved: true }>(`/api/runs/${encodeURIComponent(runId)}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
};
