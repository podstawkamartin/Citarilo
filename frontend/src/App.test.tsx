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

test("switches document sets, clears the answer and sends the selected set", async () => {
  localStorage.clear();
  const sets = [
    { id: "trailsupply", title: "Zákaznická podpora e-shopu", document_count: 6, ready: true },
    { id: "regulations", title: "Předpisy a provozní řád výroby", document_count: 5, ready: true },
  ];
  const asks: { question: string; set?: string }[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: string, init?: RequestInit) => {
      if (input === "/api/health") {
        return Response.json({
          status: "ok",
          index_ready: true,
          key_configured: true,
          document_count: 11,
          sentence_count: 200,
        });
      }
      if (input === "/api/sets") return Response.json(sets);
      if (input === "/api/ask") {
        asks.push(JSON.parse(String(init?.body)));
        return Response.json(answered);
      }
      if (input === "/api/documents/returns?set=trailsupply") return Response.json(documentView);
      return Response.json({ error: { code: "not_found", message: "Not found." } }, { status: 404 });
    }),
  );
  render(<App />);
  expect(await screen.findByRole("group", { name: "Sada dokumentů" })).toBeInTheDocument();
  expect(screen.getByRole("radio", { name: /E-shop/ })).toBeChecked();
  expect(screen.getByRole("button", { name: "Do kdy můžu vrátit zboží z výprodeje?" })).toBeInTheDocument();
  await userEvent.type(screen.getByLabelText("Vaše otázka"), "Jaká je lhůta?{Enter}");
  expect(await screen.findByText("Lhůta je 14 dní.")).toBeInTheDocument();
  await userEvent.click(screen.getByRole("radio", { name: /Výroba/ }));
  expect(localStorage.getItem("citarilo-document-set")).toBe("regulations");
  expect(screen.queryByText("Lhůta je 14 dní.")).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: /reviduje elektroinstalace/ })).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Do kdy můžu vrátit zboží z výprodeje?" })).not.toBeInTheDocument();
  await userEvent.type(screen.getByLabelText("Vaše otázka"), "Jak často?{Enter}");
  expect(asks.map((item) => item.set)).toEqual(["trailsupply", "regulations"]);
});

test("saves a note under an answer", async () => {
  const posts: { url: string; body: unknown }[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: string, init?: RequestInit) => {
      if (input === "/api/health")
        return Response.json({ status: "ok", index_ready: true, key_configured: true });
      if (input === "/api/ask") return Response.json(answered);
      if (input === "/api/documents/returns") return Response.json(documentView);
      if (input === "/api/runs/run1/feedback" && init?.method === "POST") {
        posts.push({ url: input, body: JSON.parse(String(init.body)) });
        return Response.json({ saved: true });
      }
      return Response.json({ error: { code: "not_found", message: "Not found." } }, { status: 404 });
    }),
  );
  render(<App />);
  await userEvent.type(screen.getByLabelText("Vaše otázka"), "Jaká je lhůta?{Enter}");
  await screen.findByText("Lhůta je 14 dní.");
  await userEvent.type(screen.getByLabelText("Poznámka"), "Chybí výjimka.");
  await userEvent.selectOptions(screen.getByLabelText("Štítek"), "missing_step");
  await userEvent.click(screen.getByRole("button", { name: "Uložit poznámku" }));
  expect(await screen.findByRole("status")).toHaveTextContent("Poznámka je uložená.");
  expect(posts).toEqual([
    { url: "/api/runs/run1/feedback", body: { note: "Chybí výjimka.", label: "missing_step" } },
  ]);
});

test("explains a refusal with retrieved sections and never the rejected claim", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: string) => {
      if (input === "/api/health")
        return Response.json({ status: "ok", index_ready: true, key_configured: true });
      if (input === "/api/ask") {
        return Response.json({
          ...answered,
          status: "refused",
          refusal_code: "check_failed",
          claims: [],
          sources: [],
          refusal: {
            step: "deterministic_check",
            rejected_claims: 1,
            failed_check: "literal_not_in_citations",
            sections: [
              {
                doc_id: "returns",
                doc_title: "Vrácení zboží",
                heading: "Lhůty",
                section_key: "returns.window",
              },
            ],
            reason: "SECRET-CLAIM-TEXT",
          },
        });
      }
      if (input === "/api/documents/returns") return Response.json(documentView);
      return Response.json({ error: { code: "not_found", message: "Not found." } }, { status: 404 });
    }),
  );
  render(<App />);
  await userEvent.type(screen.getByLabelText("Vaše otázka"), "Jaká je lhůta?{Enter}");
  expect(await screen.findByText("Podívejte se do těchto oddílů")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /Lhůty/ })).toHaveAttribute("aria-pressed", "true");
  expect(screen.getByText(/mimo citace/)).toBeInTheDocument();
  expect(screen.getByText("Zamítnutá tvrzení: 1")).toBeInTheDocument();
  expect(screen.queryByText("SECRET-CLAIM-TEXT")).not.toBeInTheDocument();
  expect(await screen.findByRole("article", { name: "Vrácení zboží" })).toBeInTheDocument();
  expect(screen.getByLabelText("Poznámka")).toBeInTheDocument();
});

test("renders document text as plain text, never as HTML", async () => {
  const hostile = { ...answered, claims: [{ text: "<img src=x onerror=alert(1)>", citations: ["returns.window.2"] }] };
  mockFetch(() => Response.json(hostile));
  render(<App />);
  await userEvent.type(screen.getByLabelText("Vaše otázka"), "test?{Enter}");
  expect(await screen.findByText("<img src=x onerror=alert(1)>")).toBeInTheDocument();
  expect(document.querySelector("img")).toBeNull();
});
