# Citarilo

Citarilo odpovídá na dotazy zákazníků z firemních dokumentů a každé tvrzení
dokládá konkrétní větou ze zdroje. Když dokumenty odpověď neobsahují, odmítne
místo vymýšlení. Ukázková data tvoří šest českých dokumentů fiktivního e-shopu
TrailSupply (vrácení, reklamace, doprava, platby, objednávky, půjčovna).

Stav: fáze 1, lokální backend s živou evaluací. Frontend s přehráváním
uložených běhů je fáze 2.

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
v každém reportu. Odpovědi (prompt v4) jsou v
[evals/results/2026-11-07](evals/results/2026-11-07), vyhledávání a sondy
ověřovače v [evals/results/2026-11-06](evals/results/2026-11-06).
Brány přijetí byly stanovené před laděním.

| Měření | Vývojové (15) | Odložené (5) | Brána |
|---|---|---|---|
| Recall@5 vyhledávání | 1,00 (12/12) | – | ≥ 0,9 |
| Správně zodpovězené zodpověditelné | 91,7 % (11/12) | 100 % (4/4) | ≥ 80 % |
| Zodpovězené nezodpověditelné | 0 ze 3 | 0 z 1 | 0 |
| Úspěšné prompt injection | 0 ze 2 | 0 z 1 | 0 |
| Latence p50 / max | 4,0 s / 6,6 s | 4,2 s / 5,3 s | – |
| Cena běhu | $0.0056 | $0.0018 | – |

Ověřovač na 24 sondách (12 párů „podložené / téměř správné“): 0 falešných
přijetí, 0 falešných odmítnutí. Celá fáze 1 včetně indexace, ladění a
měření latence stála $0.031.

**Jak se hodnotilo.** Stav odpovědi, recall a sondy jsou deterministické.
Sémantickou správnost (povinná fakta, zakázaná tvrzení, podpora každé citace)
hodnotil agent v T3 s jiným, silnějším modelem než aplikace (`gpt-6-astra`),
bez nákladů z API. Jde o hodnocení agentem, ne o lidskou anotaci; ruční
namátková kontrola hraničních případů zatím neproběhla. Odložené případy se
při ladění nepoužily; spouštějí se jednou pro každou finální verzi promptu.

**Jediná chyba ve vývojové sadě:** u pozdního storna půjčovny odpověď uvedla
správný poplatek 30 %, ale necitovala větu se sedmidenní hranicí, ze které
plyne; ověřovač to nezachytil.

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

Evaluace (každý placený běh má vlastní strop `--max-usd`):

```sh
uv run citarilo eval --retrieval --max-usd 0.01
uv run citarilo eval --live --max-usd 0.05
uv run citarilo eval --probes --max-usd 0.05
```

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
- Klíč je jen v `.env`, nikdy v záznamech, odpovědích ani logu. API naslouchá
  pouze na 127.0.0.1.

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
databáze, konverzace, nahrávání dokumentů, Docker a napojení na obchodní
systémy. Ověření kombinuje deterministické filtry a kontrolu modelem, takže
není důkazem pravdivosti; jeho chybovost se měří. Data jsou fiktivní.
