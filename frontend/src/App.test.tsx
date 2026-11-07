import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { App } from "./App";
import type { AskResponse } from "./api";

const answered: AskResponse = {
  run_id: "run1",
  mode: "live",
  status: "answered",
  answer_language: "cs",
  refusal_code: null,
  claims: [{ text: "Lhůta je 14 dní.", citations: ["returns.window.2"] }],
  sources: [
    {
      sentence_id: "returns.window.2",
      text: "Lhůta pro vrácení je 14 dní.",
      doc_id: "returns",
      doc_title: "Vrácení zboží",
      heading: "Lhůty",
    },
  ],
  metrics: {
    latency_ms: 4012,
    input_tokens: 2900,
    output_tokens: 250,
    cost_usd: "0.00042",
    node_timings: [{ node: "retrieve", duration_ms: 300, ok: true }],
  },
};

const documentView = {
  title: "Vrácení zboží",
  sections: [
    {
      heading: "Lhůty",
      key: "returns.window",
      sentences: [
        { id: "returns.window.1", text: "Běžné zboží lze vrátit do 30 dní." },
        { id: "returns.window.2", text: "Lhůta pro vrácení je 14 dní." },
      ],
    },
  ],
};

function mockFetch(ask: () => Response) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: string) => {
      if (input === "/api/health")
        return Response.json({ status: "ok", index_ready: true, key_configured: true });
      if (input === "/api/ask") return ask();
      if (input === "/api/documents/returns") return Response.json(documentView);
      return Response.json({ error: { code: "not_found", message: "Not found." } }, { status: 404 });
    }),
  );
}

beforeEach(() => vi.restoreAllMocks());
afterEach(() => vi.unstubAllGlobals());

test("shows cited claims and highlights the selected sentence in its document", async () => {
  mockFetch(() => Response.json(answered));
  render(<App />);
  await userEvent.type(screen.getByLabelText("Your question"), "Jaká je lhůta?{Enter}");

  const claim = await screen.findByText(/Lhůta je 14 dní\./);
  expect(claim.closest("ul")).toHaveAttribute("lang", "cs");
  expect(screen.getByRole("button", { name: "Show source 1" })).toHaveAttribute("aria-pressed", "true");
  expect(screen.getByText(/Vrácení zboží, section Lhůty/)).toBeInTheDocument();

  const panel = screen.getByRole("complementary", { name: "Source document" });
  const mark = await within(panel).findByText("Lhůta pro vrácení je 14 dní.");
  expect(mark.tagName).toBe("MARK");
  expect(within(panel).getByText("Běžné zboží lze vrátit do 30 dní.").tagName).not.toBe("MARK");
});

test("explains a refusal instead of showing an unverified answer", async () => {
  mockFetch(() =>
    Response.json({ ...answered, status: "refused", refusal_code: "insufficient_evidence", claims: [], sources: [] }),
  );
  render(<App />);
  await userEvent.type(screen.getByLabelText("Your question"), "Kolik stojí dárkový poukaz?{Enter}");
  expect(await screen.findByRole("heading", { name: "The documents don't answer this" })).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /Show source/ })).not.toBeInTheDocument();
});

test("shows the server's error message", async () => {
  mockFetch(() =>
    Response.json(
      { error: { code: "busy", message: "Another question is being answered." } },
      { status: 429 },
    ),
  );
  render(<App />);
  await userEvent.type(screen.getByLabelText("Your question"), "Jaká je lhůta?{Enter}");
  expect(await screen.findByRole("alert")).toHaveTextContent("Another question is being answered.");
});

test("renders document text as plain text, never as HTML", async () => {
  const hostile = { ...answered, claims: [{ text: "<img src=x onerror=alert(1)>", citations: ["returns.window.2"] }] };
  mockFetch(() => Response.json(hostile));
  render(<App />);
  await userEvent.type(screen.getByLabelText("Your question"), "test?{Enter}");
  expect(await screen.findByText("<img src=x onerror=alert(1)>")).toBeInTheDocument();
  expect(document.querySelector("img")).toBeNull();
});
