<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
    <img src="docs/assets/logo-light.svg" alt="Citarilo" width="277">
  </picture>
</p>

<p align="center"><b>English</b> · <a href="README.cs.md">Čeština</a></p>

**Answers from company documents where every claim is backed by a verbatim
sentence, or no answer at all.**

Citarilo answers customer questions from a company's own documents. Each claim
in an answer cites the exact sentences it rests on, and you can open each one
highlighted in the full document. Before an answer is shown, code checks the
citations and a second model call verifies that the cited sentences really
support the claim. When the documents do not contain the answer, or a claim
fails a check, Citarilo refuses instead of guessing.

https://github.com/user-attachments/assets/f4e7aa88-a632-49b7-a7df-56d9a095ffe9

It ships with two separate document sets of a company with an online shop
and a packing line, each with its own evaluation cases:

- **Customer support** (`data/trailsupply`): six sample documents of a fictional
  outdoor shop, TrailSupply (returns, warranty claims, shipping, payments,
  orders, rentals), written in Czech.
- **Equipment regulations** (`data/regulations`): four Czech government
  regulations that maintenance and health-and-safety staff work with:
  reserved electrical, pressure and lifting equipment (NV 190, 192 and
  193/2022 Sb.) and the safe operation of machinery (NV 378/2001 Sb.). They are
  downloaded from the public e-Sbírka API by
  [scripts/fetch_regulations.py](scripts/fetch_regulations.py) and converted
  into citable sentences; tables become one sentence per row and each list
  item carries its lead-in, so a single cited sentence is understandable on
  its own. Legal texts are official works and free to reuse (section 3 a) of
  Czech Act No. 121/2000 Coll.); the text is not legal advice. The set also
  contains an operating and maintenance procedure for a belt conveyor, a
  fictional test document with invented values that refers to the regulations
  where they set revision intervals.

A running server can answer from several sets at once (`DATA_DIRS`). Each set
keeps its own index, and an answer never mixes sets. `DATA_DIR` still selects
the single set used by `ingest`, `ask` and `eval`.

## What makes it different

- **Citations are sentences, not pages.** Every claim points to one to three
  specific sentences; the web interface shows them highlighted in place.
- **Checked twice before you see it.** Code rejects unknown sentence IDs and any
  number, date or URL that does not appear in the cited sentences; a separate
  model call then judges whether the citations support the claim.
- **Refusal is a feature.** Unanswerable questions and failed checks end in a
  clear refusal with its reason, never in an unverified answer.
- **Measured, not assumed.** Quality, latency and cost are evaluated on a fixed
  case set with acceptance gates set in advance; every report is in the
  repository.

## How it works

```text
question (EN/CS) → validation → budget reservation
  → retrieve  (embedding, 8 nearest sections)
  → generate  (gpt-6-luna: up to 4 claims with the IDs of cited sentences)
  → check     (code: IDs from retrieved sentences; numbers/dates/URLs in cited sentences)
  → verify    (second model call: do the cited sentences alone support the claim?)
  → assemble  or refuse (insufficient_evidence | check_failed | unsupported | contradicted)
  → run record (.runtime/runs) and cost accounting
```

- The flow is a LangGraph `StateGraph` with conditional edges; models, the
  index and the budget reach the nodes through the `Runtime` context, not
  global state.
- Model output uses a strict JSON Schema; usage is billed from the provider's
  response before the output is validated.
- Answers are in the language of the question; quoted sentences stay in the
  original Czech.
- Claims state only values from the documents and never derive new ones (sums,
  discounted prices); a derived number fails the check.
- Retrieval is vector search over document sections (`text-embedding-3-large`)
  in Qdrant embedded local mode (`.runtime/index/<set>-<hash>/`, one index per
  document set), with a corpus-bound
  manifest and a checksum of the stored points. The checksum detects corruption
  and inconsistent builds, not a malicious local writer.
- Design details: [docs/architecture.md](docs/architecture.md) (Czech),
  evaluation method: [docs/evaluation.md](docs/evaluation.md).

## Results (19 Nov 2026)

Live runs against the OpenAI API with prompt v10 and 8 retrieved sections, on a
clean commit; configuration, commit and corpus hash are stored in every report
([evals/results/2026-11-19](evals/results/2026-11-19)). Acceptance gates were set
before tuning. Each set has a development split used for tuning and a held-out
split run once at the end.

| Measure | Customer support, dev (29) | Customer support, held-out (8) | Regulations, dev (59) | Regulations, held-out (15) | Gate |
|---|---|---|---|---|---|
| Retrieval recall@5 | 0.96 (23/24) | 1.00 (6/6) | 0.95 (52/55) | 0.85 (11/13) | ≥ 0.9 |
| Answerable questions answered correctly | 95.8 % (23/24) | 100 % (6/6) | 83.6 % (46/55) | 84.6 % (11/13) | ≥ 80 % |
| of which general questions | 7/7 | 2/2 | 7/9 | 2/2 | – |
| Unanswerable questions answered | 0 of 5 | 0 of 2 | 0 of 4 | 0 of 2 | 0 |
| Successful prompt injections | 0 of 3 | 0 of 1 | 0 of 1 | – | 0 |
| Latency p50 / max | 4.0 s / 8.1 s | 3.5 s / 4.3 s | 4.1 s / 10.5 s | 3.4 s / 5.1 s | – |
| Cost per question | ≈ $0.0005 | ≈ $0.0005 | ≈ $0.0018 | ≈ $0.0017 | – |

**General questions.** Each set includes short, loosely worded questions as
customers and operators actually ask them, tagged `vague`: "Chci něco vrátit,
jak na to?", "Kolik stojí doprava?", "Pás stojí a nejede. Jak mám postupovat?",
"Svítí tam F01, co s tím?", "Hlásí to chybu E12, co s tím?". Some have no answer
in the documents ("Jak vyměním pás na dopravníku?") and must be refused.

**How it was graded.** Answer status, recall and the verifier probes are
deterministic. Semantic correctness (required facts, forbidden claims, support
for each citation) was graded offline by an independent LLM grader of a
different family than the app's model (Claude Sonnet). This is model grading,
not human annotation.

**What fails.** Every error is a refusal or an incomplete answer; no run
answered an unanswerable question, followed an injected instruction or kept an
unsupported claim. On the regulations development set, eight of the nine errors
are unnecessary refusals: three answers were rejected by the verifier and one by
a deterministic check although the grader found every claim supported; twice
the model found no evidence; twice a claim went beyond its citations (a lockout
step it did not cite, "repair" where the procedure says maintenance), which the
verifier rightly rejected. The ninth answer gives
the class III range for a steam boiler without saying that a 6 t/h boiler is
class IV. The two held-out errors follow the two retrieval misses: the
section on re-examining pressure-vessel operators and part of the large-boiler
revision rules did not reach the top five. On customer support, one answer gave
the return-label price without saying it is deducted from the refund.

**Tuning.** The general questions were tuned on the development split with
three runs each. With prompt v9, "Pás stojí a nejede" was refused 3 of 3 times
and "Hlásí to chybu E12" 3 of 3 times; F01 failed 2 of 3 times because the
answer invented an order of steps. Prompt v10 forbids claims that something is
absent from or not required by the documents, states rules for unlisted cases
as rules, gives safety steps one per claim and treats a problem reported in
general terms as within the situations the documents distinguish. The conveyor
procedure now states the F01 order explicitly. After that all three passed and
no unanswerable control was answered. The final run still refused three
operator questions that had passed while tuning, so single answers on those
questions vary between runs.

**Verifier probes.** Measured on 18 Nov with the same verifier prompt
([evals/results/2026-11-18](evals/results/2026-11-18)). Pairs of a supported
claim and a near-miss: on customer support (24 probes) 0 false acceptances and
0 false rejections; on regulations (30 probes, plus 4 conveyor probes run separately) 1 false
acceptance and 1 false rejection. The false acceptance labelled a class IV boiler as class I although
its own reason quoted class IV. Probes measure the verifier alone; in the full
pipeline a deterministic check also requires a Roman class numeral in a claim
("class I", "IV. třídy") to appear in its citations, so a claim like that
false acceptance is refused.

**Vector store.** An offline benchmark through the production code paths
([vector-store-bench.json](evals/results/2026-11-10/vector-store-bench.json))
shows startup validation of 24 ms and a median query of 1.5 ms for the 36
sample sections. Qdrant local mode compares the query with every point, so at
5,000 sections startup takes 3.4 s and a median query 34 ms; larger corpora call
for a Qdrant server.

The samples are small (98 answerable cases across both sets), so the numbers
show that the system works and fails safely, not a statistically precise error
rate. Latency is three sequential provider calls (embedding, generation,
verification). Earlier runs are in
[evals/results/2026-11-07](evals/results/2026-11-07),
[evals/results/2026-11-10](evals/results/2026-11-10) and
[evals/results/2026-11-18](evals/results/2026-11-18).

## Getting started

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), an OpenAI API
key; Node.js 24 for the web interface.

```sh
uv sync
cp .env.example .env              # set OPENAI_API_KEY
uv run citarilo ingest --dry-run  # cost estimate, no network
uv run citarilo ingest            # build the section index (≈ $0.001)
uv run citarilo ask "Do kdy můžu vrátit zboží z výprodeje?"
```

The equipment regulations are selected with `DATA_DIR` (stop the server first;
each set keeps its own index, so ingest runs once per set):

```sh
DATA_DIR=data/regulations uv run citarilo ingest   # ≈ $0.015
DATA_DIR=data/regulations uv run citarilo ask "Jak často se reviduje elektroinstalace ve výrobní hale?"
```

The web interface is built once; `citarilo serve` then serves it on the same
address as the API:

```sh
cd frontend && npm ci && npm run build && cd ..
uv run citarilo serve             # http://127.0.0.1:5177
DATA_DIRS=data/trailsupply,data/regulations uv run citarilo serve
```

Ingest each set once, with the server stopped, before serving both. The page
then offers a document-set switch; an answer always stays inside the set that
was selected.

The interface is in Czech. An answer is laid out as claims, the cited
sentences, the source document with the selected sentence highlighted, and the
checks the answer passed with their time and cost. A refusal explains its
reason. The API key stays on the server. For development with live reload:
`npm run dev` (port 5178, `/api` is proxied to 5177).

Evaluation (each paid run has its own `--max-usd` cap; the case set follows
`DATA_DIR` or is chosen with `--eval-set trailsupply|regulations`):

```sh
uv run citarilo eval --retrieval --max-usd 0.01
uv run citarilo eval --live --max-usd 0.05
uv run citarilo eval --probes --max-usd 0.05
DATA_DIR=data/regulations uv run citarilo eval --live --max-usd 0.12
```

Every question asked through `ask` or the web interface leaves a run record in
`.runtime/runs`. Refused runs show which step rejected the answer and why. In
the web interface a note can be attached to any answer or refusal;
`citarilo runs --feedback` lists those notes, and `--to-case` copies the note
into the draft case:

```sh
uv run citarilo runs --failed --since 2026-11-19
uv run citarilo runs --feedback
uv run citarilo runs --to-case RUN_ID --eval-set regulations
```

## Your own documents

`citarilo import` converts documents (PDF with a text layer, DOCX, Markdown)
into the format Citarilo cites from: sections by heading and one sentence per
line with a stable ID. The output is readable Markdown to review before
indexing.

```sh
uv run citarilo import ~/documents/*.pdf ~/documents/*.docx --out data/company
# review data/company/*.md, then set DATA_DIR=data/company in .env
# stop the running server before ingest, then start it again
uv run citarilo ingest
```

Stop the server, run `ingest`, then start it again. At startup, the server
checks the index against the documents and refuses an outdated index. It keeps
that validated snapshot until restart; document changes require another ingest
and restart.

Import refuses rather than silently losing or reordering text: scanned PDFs
without text, tables, multi-column layouts, tracked changes and other
unsupported structures are reported with guidance. It is meant for files from
an administrator, not for uploads from anonymous users.

The sample documents also come in office formats in
[examples/trailsupply-raw](examples/trailsupply-raw) (PDF, DOCX, Markdown);
importing them reproduces exactly the same sentences.

The prompts are not tied to a document set. They expect Czech source
documents; answers follow the language of the question (Czech or English).

## API

| Endpoint | Description |
|---|---|
| `POST /api/ask` | `{"question": "...", "set"?: "<id>"}` → status, refusal code, a safe refusal explanation, claims with citations, source sentences, answer language, metrics. `set` defaults to the first configured set |
| `GET /api/sets` | document sets: `id`, `title`, `document_count`, `ready` |
| `GET /api/health` | index and key readiness, corpus hash (one set only), number of documents and citable sentences (`document_count`, `sentence_count`; 0 without a corpus), remaining budget |
| `GET /api/documents/{id}` | the document by sections and sentences with IDs, its file name (`file_name`, without a path). Optional `set` query |
| `POST /api/runs/{run_id}/feedback` | `{"note"?: "...", "label"?: "..."}` appends a local note (at most 1000 characters). It does not change the run record |

Errors share one shape `{"error": {"code", "message"}}` (413, 422, 429, 502,
503). A question is limited to 3–500 characters and 4 KiB; one question is
processed at a time.

## Security and budget

- `.runtime/` is trusted local state, created so that only its owner can open
  it; Qdrant local storage must not be copied from untrusted sources. Keep `.env`
  private as well (`chmod 600 .env`).
- Documents and questions are untrusted data in separate blocks; the model has
  no tools and no secrets. An instruction planted in a document
  (`orders.order-notes.3`) is part of the tests.
- Every paid call reserves its worst-case cost in SQLite before it is sent
  (atomically, also across processes). Unbilled or interrupted calls stay
  counted. The total spending cap is at most $3; when it is reached, the server
  stops answering. Requests enforce the standard price tier.
- The administrator sets the API key in an environment variable or `.env`; it
  never appears in records, responses or logs. The API listens only on
  127.0.0.1 and answers only requests addressed to `127.0.0.1` or `localhost`,
  so a web page cannot reach it by rebinding its own domain. The web interface
  has a strict Content Security Policy; API documentation pages are not served.

## Development and tests

```sh
uv run ruff check && uv run ruff format --check
uv run pytest
cd frontend && npm test && npm run typecheck
```

Tests are deterministic, with the network blocked and fake models; they never
call the paid API or read a local key. The same checks run in GitHub Actions.

## Limitations

- Qdrant runs embedded, with one process at a time owning the local index.
  Stop the server before using CLI `ask`, `eval` or `ingest`. Qdrant server
  mode is not wired yet.
- One question at a time and local access only (127.0.0.1); there are no user
  accounts. Serving a team needs a reverse proxy with authentication and
  concurrent request handling.
- Text only: no OCR, tables or images. Documents are added on the server with
  `import` and `ingest`, not through the web interface.
- Single questions, no conversation; no streaming, hybrid search or reranking.
- Several document sets can be served at once with `DATA_DIRS`; an answer never
  mixes them. `ingest`, CLI `ask` and `eval` still use one `DATA_DIR`, and the
  server must be stopped first. Source documents are expected in Czech.
- Verification combines deterministic checks and a model judgement, so it is
  not a proof of truth; its error rate is measured. The customer-support data
  and the conveyor procedure are fictional; the regulations are real legal
  texts.
