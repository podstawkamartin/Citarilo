# Citarilo

Citarilo odpovídá na dotazy zákazníků z firemních dokumentů a každé tvrzení
dokládá konkrétní větou ze zdroje. Když dokumenty odpověď neobsahují, odmítne
místo vymýšlení. Dodává se se šesti testovacími dokumenty fiktivního e-shopu
TrailSupply (vrácení, reklamace, doprava, platby, objednávky, půjčovna).

## Jak to funguje

```text
otázka (EN/CS) → validace → rezervace rozpočtu
  → retrieve  (embedding, 5 nejbližších sekcí)
  → generate  (gpt-6-luna: až 4 tvrzení s ID citovaných vět)
  → check     (kód: ID z vyhledaných vět, čísla/data/URL v citovaných větách)
  → verify    (druhé volání modelu: podporují tvrzení samotné citované věty?)
  → assemble  nebo refuse (insufficient_evidence | check_failed | unsupported | contradicted)
  → záznam běhu (.runtime/runs) a vyúčtování
```

- Tok řídí LangGraph `StateGraph` s podmíněnými hranami; modely, index a
  rozpočet dostávají uzly přes `Runtime` kontext, ne přes globální stav.
- Strukturovaný výstup modelu má striktní JSON Schema; spotřeba se vyúčtuje
  z odpovědi poskytovatele dřív, než se výstup validuje.
- Odpověď je v jazyce otázky, zdrojové věty zůstávají v původní češtině.
- Tvrzení uvádějí jen hodnoty z dokumentů a nic nedopočítávají (součty,
  ceny po slevě); dopočítané číslo kontrola zamítne.
- Všechna tři volání poskytovatele sdílejí jedno HTTP spojení s keep-alive.
- Podrobný návrh: [docs/architecture.md](docs/architecture.md), metodika
  evaluace: [docs/evaluation.md](docs/evaluation.md).

## Výsledky (7. 11. 2026)

Živé běhy proti OpenAI API; konfigurace, commit a hash korpusu jsou uložené
v každém reportu. Vyhledávání a odpovědi (prompt v4) jsou v
[evals/results/2026-11-07](evals/results/2026-11-07), sondy ověřovače
v [evals/results/2026-11-06](evals/results/2026-11-06).
Brány přijetí byly stanovené před laděním.

| Měření | Vývojové (21) | Odložené (5) | Brána |
|---|---|---|---|
| Recall@5 vyhledávání | 1,00 (17/17) | – | ≥ 0,9 |
| Správně zodpovězené zodpověditelné | 82,4 % (14/17) | 100 % (4/4) | ≥ 80 % |
| Zodpovězené nezodpověditelné | 0 ze 4 | 0 z 1 | 0 |
| Úspěšné prompt injection | 0 ze 3 | 0 z 1 | 0 |
| Latence p50 / max | 4,0 s / 6,1 s | 4,4 s / 4,7 s | – |
| Cena běhu | $0.0079 | $0.0018 | – |

Ověřovač na 24 sondách (12 párů „podložené / téměř správné“): 0 falešných
přijetí, 0 falešných odmítnutí. Vývoj včetně indexace, ladění a měření
latence stál celkem $0.042.

**Jak se hodnotilo.** Stav odpovědi, recall a sondy jsou deterministické.
Sémantickou správnost (povinná fakta, zakázaná tvrzení, podpora každé citace)
hodnotil offline nezávislý LLM hodnotitel se silnějším modelem jiného typu
než aplikace (`gpt-6-astra`). Jde o hodnocení modelem, ne o lidskou anotaci; ruční
namátková kontrola hraničních případů zatím neproběhla. Odložené případy se
při ladění nepoužily; spouštějí se jednou pro každou finální verzi promptu.

**Chyby ve vývojové sadě (3):**
- Pozdní storno půjčovny: tvrzení o poplatku 30 % necitovalo větu se
  sedmidenní hranicí; ověřovač ho správně zamítl a odpověď byla odmítnuta.
- Zpětný štítek vs. reklamace: ověřovač zamítl podložené tvrzení, protože
  neopakovalo omezení na běžný balík (příliš přísné odmítnutí).
- Vrácení o víkendu: odpověď uvedla „osobně“ bez citace věty, která to říká.

Dvě ze tří chyb jsou zbytečná odmítnutí, ne nepravdivé odpovědi.

Vzorek je malý (16 zodpověditelných případů), takže čísla dokládají funkčnost
a bezpečné chování, ne statisticky přesnou míru chybovosti. Latenci tvoří tři
po sobě jdoucí volání poskytovatele (embedding, generování, ověření); většinu
času zabírá generování a ověření na straně modelu.

## Spuštění

Potřeba: Python 3.12, [uv](https://docs.astral.sh/uv/), OpenAI API klíč.

```sh
uv sync
cp .env.example .env              # doplnit OPENAI_API_KEY
uv run citarilo ingest --dry-run  # odhad ceny bez sítě
uv run citarilo ingest            # index sekcí (≈ $0.001)
uv run citarilo ask "Do kdy můžu vrátit zboží z výprodeje?"
uv run citarilo serve             # API na http://127.0.0.1:5174
```

Webové rozhraní (Node.js 24) se sestaví jednou a `citarilo serve` ho pak
servíruje na stejné adrese jako API:

```sh
cd frontend && npm ci && npm run build && cd ..
uv run citarilo serve             # http://127.0.0.1:5174
```

Uživatel položí otázku, dostane tvrzení s očíslovanými citacemi a kliknutím
na citaci uvidí zvýrazněnou větu v celém dokumentu. Odmítnutí rozhraní
vysvětlí místo neověřené odpovědi. Klíč zůstává jen na serveru. Vývoj
s automatickým obnovováním: `npm run dev` (port 5175, `/api` jde na 5174).

Evaluace (každý placený běh má vlastní strop `--max-usd`):

```sh
uv run citarilo eval --retrieval --max-usd 0.01
uv run citarilo eval --live --max-usd 0.05
uv run citarilo eval --probes --max-usd 0.05
```

## Vlastní dokumenty

`citarilo import` převede firemní dokumenty (PDF s textovou vrstvou, DOCX,
Markdown) do formátu, ze kterého Citarilo cituje: sekce podle nadpisů a jedna
věta na řádek se stabilním ID. Výstup je čitelný Markdown ke kontrole před
indexací.

```sh
uv run citarilo import ~/dokumenty/*.pdf ~/dokumenty/*.docx --out data/firma
# zkontrolovat data/firma/*.md, potom v .env nastavit DATA_DIR=data/firma
uv run citarilo ingest
```

Import raději odmítne, než aby tiše ztratil nebo přeházel text: naskenovaná
PDF bez textu, tabulky, vícesloupcovou sazbu, sledované změny a další
nepodporované struktury hlásí s návodem na úpravu. Je určený pro soubory od
správce, ne pro nahrávání od anonymních uživatelů.

Stejné testovací dokumenty jsou i ve firemních formátech v
[examples/trailsupply-raw](examples/trailsupply-raw) (PDF, DOCX, Markdown).
Po importu dávají přesně stejné věty a evaluace nad nimi běží beze změny
(vývojová sada 7. 11. 2026: recall@5 1,00; 20 z 21 očekávaných stavů).

## API

| Endpoint | Popis |
|---|---|
| `POST /api/ask` | `{"question": "..."}` → stav, kód odmítnutí, tvrzení s citacemi, zdrojové věty, jazyk odpovědi, metriky |
| `GET /api/health` | připravenost indexu a klíče, hash korpusu, zbývající rozpočet |
| `GET /api/documents/{id}` | dokument po sekcích a větách s ID |

Chyby mají jednotný tvar `{"error": {"code", "message"}}` (413, 422, 429,
502, 503). Dotaz je omezený na 3–500 znaků a 4 KiB, zpracovává se jeden
naráz.

## Bezpečnost a rozpočet

- Dokumenty i otázky jsou nedůvěryhodná data v oddělených blocích; model nemá
  nástroje ani tajemství. Vložený pokyn v dokumentu (`orders.order-notes.3`)
  je součástí testů.
- Každé placené volání si před odesláním rezervuje nejhorší možnou cenu
  v SQLite (atomicky, i mezi procesy). Nevyúčtovaná nebo přerušená volání
  zůstávají započtená. Pevný strop je $3; požadavky vynucují standardní
  cenovou úroveň.
- API klíč nastavuje správce v proměnné prostředí nebo v `.env`; nikdy se
  neobjeví v záznamech, odpovědích ani logu. API naslouchá pouze na 127.0.0.1.

## Vývoj a testy

```sh
uv run ruff check && uv run ruff format --check
uv run pytest
```

Testy jsou deterministické, se zablokovanou sítí a falešnými modely; nikdy
nevolají placené API ani nečtou lokální klíč. Stejné kontroly běží v GitHub
Actions.

## Omezení

Vědomě vynecháno: streamování, hybridní vyhledávání a reranking, vektorová
databáze, konverzace, nahrávání dokumentů přes webové rozhraní, OCR, Docker
a napojení na obchodní systémy. Ověření kombinuje deterministické filtry a kontrolu modelem, takže
není důkazem pravdivosti; jeho chybovost se měří. Data jsou fiktivní.
