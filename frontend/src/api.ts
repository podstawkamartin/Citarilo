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

export type AskResponse = {
  run_id: string;
  mode: string;
  status: "answered" | "refused";
  answer_language: "en" | "cs" | null;
  refusal_code: "insufficient_evidence" | "check_failed" | "unsupported" | "contradicted" | null;
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
};

export type DocumentView = {
  title: string;
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
    throw new ApiError("network_error", "The server is not reachable. Check that citarilo serve is running.");
  }
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const error = body?.error;
    throw new ApiError(
      typeof error?.code === "string" ? error.code : "http_error",
      typeof error?.message === "string" ? error.message : `The server responded with ${response.status}.`,
    );
  }
  return body as T;
}

export const api = {
  ask: (question: string) =>
    request<AskResponse>("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    }),
  health: () => request<Health>("/api/health"),
  document: (docId: string) => request<DocumentView>(`/api/documents/${encodeURIComponent(docId)}`),
};
