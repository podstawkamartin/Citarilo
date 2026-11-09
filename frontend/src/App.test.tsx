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
  file_name: "returns.md",
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

function mockFetch(ask: () => Response, counts: object = {}) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: string) => {
      if (input === "/api/health")
        return Response.json({ status: "ok", index_ready: true, key_configured: true, ...counts });
      if (input === "/api/ask") return ask();
      if (input === "/api/documents/returns") return Response.json(documentView);
      return Response.json({ error: { code: "not_found", message: "Not found." } }, { status: 404 });
    }),
  );
}

beforeEach(() => vi.restoreAllMocks());
afterEach(() => vi.unstubAllGlobals());

test("shows claims, their citations and the selected citation highlighted in its document", async () => {
  mockFetch(() => Response.json(answered));
  render(<App />);
  await userEvent.type(screen.getByLabelText("Vaše otázka"), "Jaká je lhůta?{Enter}");

  const claim = await screen.findByText("Lhůta je 14 dní.");
  expect(claim).toHaveAttribute("lang", "cs");
  expect(screen.getByText("Podloženo 1 citací")).toBeInTheDocument();
  const citations = screen.getByRole("region", { name: "Citace z dokumentu" });
  expect(within(citations).getByRole("button", { name: /Lhůta pro vrácení je 14 dní\./ })).toHaveAttribute(
    "aria-pressed",
    "true",
  );

  const paper = screen.getByRole("article", { name: "Vrácení zboží" });
  const mark = await within(paper).findByText("Lhůta pro vrácení je 14 dní.");
  expect(mark.tagName).toBe("MARK");
  expect(within(paper).getByText("returns.md")).toBeInTheDocument();
  expect(within(paper).getByText(/Běžné zboží lze vrátit do 30 dní\./).tagName).not.toBe("MARK");
  expect(screen.getByText("Vyhledání relevantních sekcí")).toBeInTheDocument();
  expect(screen.getByText("0,0004 $")).toBeInTheDocument();
});

test("names each claim's citations for assistive technology", async () => {
  mockFetch(() => Response.json(answered));
  render(<App />);
  await userEvent.type(screen.getByLabelText("Vaše otázka"), "Jaká je lhůta?{Enter}");
  const claim = (await screen.findByText("Lhůta je 14 dní.")).closest("article");
  expect(claim).toHaveAccessibleDescription("Lhůta pro vrácení je 14 dní.");
});

test("marks the check that rejected a draft answer", async () => {
  const timings = ["retrieve", "generate", "check", "verify", "refuse"].map((node) => ({
    node,
    duration_ms: 100,
    ok: true,
  }));
  mockFetch(() =>
    Response.json({
      ...answered,
      status: "refused",
      refusal_code: "unsupported",
      claims: [],
      sources: [],
      metrics: { ...answered.metrics, node_timings: timings },
    }),
  );
  render(<App />);
  await userEvent.type(screen.getByLabelText("Vaše otázka"), "Jaká je lhůta?{Enter}");
  const verify = await screen.findByText(/Kontrola tvrzení druhým modelem/);
  expect(verify.closest("li")).toHaveClass("failed");
  expect(verify.closest("li")).toHaveTextContent("(neprošlo)");
  expect(screen.getByText(/Kontrola čísel a citací kódem/).closest("li")).not.toHaveClass("failed");
});

test("shows document and sentence counts from the server", async () => {
  mockFetch(() => Response.json(answered), { document_count: 6, sentence_count: 146 });
  render(<App />);
  expect(await screen.findByText("6 dokumentů · 146 vět")).toBeInTheDocument();
});

test("explains the colours in the help panel and closes it with Escape", async () => {
  mockFetch(() => Response.json(answered));
  render(<App />);
  const help = screen.getByRole("button", { name: /Nápověda/ });
  await userEvent.click(help);
  expect(help).toHaveAttribute("aria-expanded", "true");
  expect(screen.getByText("Citace z dokumentu", { selector: "strong" })).toBeInTheDocument();
  await userEvent.keyboard("{Escape}");
  expect(help).toHaveAttribute("aria-expanded", "false");
  expect(help).toHaveFocus();
});

test("switches between light and dark theme", async () => {
  mockFetch(() => Response.json(answered));
  document.documentElement.dataset.theme = "light";
  render(<App />);
  await userEvent.click(screen.getByRole("button", { name: "Přepnout na tmavý režim" }));
  expect(document.documentElement.dataset.theme).toBe("dark");
  expect(screen.getByRole("button", { name: "Přepnout na světlý režim" })).toBeInTheDocument();
});

test("explains a refusal instead of showing an unverified answer", async () => {
  mockFetch(() =>
    Response.json({ ...answered, status: "refused", refusal_code: "insufficient_evidence", claims: [], sources: [] }),
  );
  render(<App />);
  await userEvent.type(screen.getByLabelText("Vaše otázka"), "Kolik stojí dárkový poukaz?{Enter}");
  expect(await screen.findByRole("heading", { name: "Dokumenty na tohle neodpovídají" })).toBeInTheDocument();
  expect(screen.queryByRole("region", { name: "Citace z dokumentu" })).not.toBeInTheDocument();
});

test("shows the server's error message", async () => {
  mockFetch(() =>
    Response.json(
      { error: { code: "busy", message: "Another question is being answered." } },
      { status: 429 },
    ),
  );
  render(<App />);
  await userEvent.type(screen.getByLabelText("Vaše otázka"), "Jaká je lhůta?{Enter}");
  expect(await screen.findByRole("alert")).toHaveTextContent("Another question is being answered.");
});

test("renders document text as plain text, never as HTML", async () => {
  const hostile = { ...answered, claims: [{ text: "<img src=x onerror=alert(1)>", citations: ["returns.window.2"] }] };
  mockFetch(() => Response.json(hostile));
  render(<App />);
  await userEvent.type(screen.getByLabelText("Vaše otázka"), "test?{Enter}");
  expect(await screen.findByText("<img src=x onerror=alert(1)>")).toBeInTheDocument();
  expect(document.querySelector("img")).toBeNull();
});
