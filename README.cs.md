<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
    <img src="docs/assets/logo-light.svg" alt="Citarilo" width="277">
  </picture>
</p>

<p align="center"><a href="README.md">English</a> · <b>Čeština</b></p>

**Odpovědi z firemních dokumentů, kde každé tvrzení dokládá doslovná věta.
Jinak žádná odpověď.**

Citarilo odpovídá na dotazy zákazníků z vlastních dokumentů firmy. Každé
tvrzení v odpovědi cituje přesné věty, o které se opírá, a každou z nich lze
otevřít zvýrazněnou v celém dokumentu. Než se odpověď zobrazí, kód zkontroluje
citace a druhé volání modelu ověří, že citované věty tvrzení opravdu
podporují. Když dokumenty odpověď neobsahují nebo tvrzení kontrolou neprojde,
Citarilo odmítne místo hádání.

Dodává se se šesti testovacími dokumenty fiktivního outdoorového obchodu
TrailSupply (vrácení, reklamace, doprava, platby, objednávky, půjčovna)
v češtině.

## Čím se liší

- **Citace jsou věty, ne stránky.** Každé tvrzení odkazuje na jednu až tři
  konkrétní věty; webové rozhraní je ukáže zvýrazněné přímo v dokumentu.
- **Dvakrát zkontrolováno, než to uvidíte.** Kód zamítne neznámá ID vět
  a každé číslo, datum nebo URL, které v citovaných větách není; samostatné
  volání modelu pak posoudí, zda citace tvrzení podporují.
- **Odmítnutí je funkce.** Nezodpověditelná otázka nebo neúspěšná kontrola
  končí srozumitelným odmítnutím s důvodem, nikdy neověřenou odpovědí.
- **Změřeno, ne předpokládáno.** Kvalita, latence a cena se vyhodnocují na
  pevné sadě případů s předem stanovenými branami; všechny reporty jsou
  v repozitáři.

## Jak to funguje

```text
otázka (EN/CS) → validace → rezervace rozpočtu
  → retrieve  (embedding, 5 nejbližších sekcí)
  → generate  (gpt-6-luna: až 4 tvrzení s ID citovaných vět)
  → check     (kód: ID z vyhledaných vět; čísla/data/URL v citovaných větách)
  → verify    (druhé volání modelu: podporují tvrzení samotné citované věty?)
  → assemble  nebo refuse (insufficient_evidence | check_failed | unsupported | contradicted)
  → záznam běhu (.runtime/runs) a vyúčtování
```

- Tok řídí LangGraph `StateGraph` s podmíněnými hranami; modely, index a
  rozpočet dostávají uzly přes `Runtime` kontext, ne přes globální stav.
- Strukturovaný výstup modelu má striktní JSON Schema; spotřeba se vyúčtuje
  z odpovědi poskytovatele dřív, než se výstup validuje.
- Odpověď je v jazyce otázky, citované věty zůstávají v původní češtině.
- Tvrzení uvádějí jen hodnoty z dokumentů a nic nedopočítávají (součty,
  ceny po slevě); dopočítané číslo kontrola zamítne.
- Vyhledávání je vektorové nad sekcemi dokumentů (`text-embedding-3-large`)
  v paměťovém úložišti uloženém do souboru s manifestem a kontrolním součtem.
  Při této velikosti není potřeba samostatná databáze.
- Podrobný návrh: [docs/architecture.md](docs/architecture.md), metodika
  evaluace: [docs/evaluation.md](docs/evaluation.md) (anglicky).

## Výsledky (7. 11. 2026)

Živé běhy proti OpenAI API; konfigurace, commit a hash korpusu jsou uložené
v každém reportu. Vyhledávání a odpovědi (prompt v4) jsou v
[evals/results/2026-11-07](evals/results/2026-11-07). Brány přijetí byly
stanovené před laděním.

| Měření | Vývojové (21 případů) | Odložené (5 případů) | Brána |
|---|---|---|---|
| Recall@5 vyhledávání | 1,00 (17/17) | – | ≥ 0,9 |
| Správně zodpovězené zodpověditelné | 82,4 % (14/17) | 100 % (4/4) | ≥ 80 % |
| Zodpovězené nezodpověditelné | 0 ze 4 | 0 z 1 | 0 |
| Úspěšné prompt injection | 0 ze 3 | 0 z 1 | 0 |
| Latence p50 / max | 4,0 s / 6,1 s | 4,4 s / 4,7 s | – |
| Cena za otázku | ≈ $0.0004 | ≈ $0.0004 | – |

**Jak se hodnotilo.** Stav odpovědi, recall a sondy ověřovače jsou
deterministické. Sémantickou správnost (povinná fakta, zakázaná tvrzení,
podpora každé citace) hodnotil offline nezávislý LLM hodnotitel se silnějším
modelem jiného typu než aplikace (`gpt-6-astra`). Jde o hodnocení modelem, ne
o lidskou anotaci; ruční kontrola hraničních případů zatím neproběhla.
Odložené případy se při ladění nepoužily; spouštějí se jednou pro každou
finální verzi promptu.

**Chyby ve vývojové sadě (3):**
- Pozdní storno půjčovny: tvrzení o poplatku 30 % necitovalo větu se
  sedmidenní hranicí; ověřovač ho správně zamítl a odpověď byla odmítnuta.
- Zpětný štítek vs. reklamace: ověřovač zamítl podložené tvrzení, protože
  neopakovalo omezení na běžný balík (příliš přísné odmítnutí).
- Vrácení o víkendu: odpověď uvedla „osobně“ bez citace věty, která to říká,
  a ověřovač ji propustil.

Dvě ze tří chyb jsou zbytečná odmítnutí, jedna je odpověď s necitovaným
detailem.

**Sondy ověřovače.** Na 24 sondách (12 párů „podložené / téměř správné“
tvrzení) měl ověřovač 0 falešných přijetí a 0 falešných odmítnutí
([evals/results/2026-11-06](evals/results/2026-11-06)). Měřeno s promptem v3
nad předchozí verzí testovacích dokumentů; pro v4 zatím znovu nespuštěno.

Vzorek je malý (celkem 21 zodpověditelných případů), takže čísla dokládají
funkčnost a bezpečné chování, ne statisticky přesnou míru chybovosti. Latenci
tvoří tři po sobě jdoucí volání poskytovatele (embedding, generování,
ověření); většinu času zabírá generování a ověření na straně modelu.

## Spuštění

Potřeba: Python 3.12, [uv](https://docs.astral.sh/uv/), OpenAI API klíč; pro
webové rozhraní Node.js 24.

```sh
uv sync
cp .env.example .env              # doplnit OPENAI_API_KEY
uv run citarilo ingest --dry-run  # odhad ceny bez sítě
uv run citarilo ingest            # index sekcí (≈ $0.001)
uv run citarilo ask "Do kdy můžu vrátit zboží z výprodeje?"
```

Webové rozhraní se sestaví jednou a `citarilo serve` ho pak servíruje na
stejné adrese jako API:

```sh
cd frontend && npm ci && npm run build && cd ..
uv run citarilo serve             # http://127.0.0.1:5174
```

Rozhraní je česky. Odpověď ukáže tvrzení, citované věty, zdrojový dokument se
zvýrazněnou vybranou větou a kontroly, kterými prošla, včetně času a ceny.
Odmítnutí vysvětlí důvod. Klíč zůstává jen na serveru. Vývoj s automatickým
obnovováním: `npm run dev` (port 5175, `/api` jde na 5174).

Evaluace (každý placený běh má vlastní strop `--max-usd`):

```sh
uv run citarilo eval --retrieval --max-usd 0.01
uv run citarilo eval --live --max-usd 0.05
uv run citarilo eval --probes --max-usd 0.05
```

## Vlastní dokumenty

`citarilo import` převede dokumenty (PDF s textovou vrstvou, DOCX, Markdown)
do formátu, ze kterého Citarilo cituje: sekce podle nadpisů a jedna věta na
řádek se stabilním ID. Výstup je čitelný Markdown ke kontrole před indexací.

```sh
uv run citarilo import ~/dokumenty/*.pdf ~/dokumenty/*.docx --out data/firma
# zkontrolovat data/firma/*.md, potom v .env nastavit DATA_DIR=data/firma
uv run citarilo ingest
```

Po novém `ingest` restartujte server. Index je svázaný s hashem dokumentů,
takže server nikdy neodpovídá ze zastaralého indexu; místo toho ohlásí, že je
potřeba index sestavit znovu.

Import raději odmítne, než aby tiše ztratil nebo přeházel text: naskenovaná
PDF bez textu, tabulky, vícesloupcovou sazbu, sledované změny a další
nepodporované struktury hlásí s návodem na úpravu. Je určený pro soubory od
správce, ne pro nahrávání od anonymních uživatelů.

Testovací dokumenty jsou i ve firemních formátech v
[examples/trailsupply-raw](examples/trailsupply-raw) (PDF, DOCX, Markdown);
po importu dávají přesně stejné věty.

**Pozor:** prompty pro odpovídání a ověřování jsou zatím napsané pro
testovací dokumenty TrailSupply v češtině. Dokumenty jiné firmy nebo v jiném
jazyce vyžadují úpravu promptů v `src/citarilo/prompts.py` a nové evaluační
případy.

## API

| Endpoint | Popis |
|---|---|
| `POST /api/ask` | `{"question": "..."}` → stav, kód odmítnutí, tvrzení s citacemi, zdrojové věty, jazyk odpovědi, metriky |
| `GET /api/health` | připravenost indexu a klíče, hash korpusu, počty dokumentů a citovatelných vět (`document_count`, `sentence_count`; bez korpusu 0), zbývající rozpočet |
| `GET /api/documents/{id}` | dokument po sekcích a větách s ID, název souboru (`file_name`, bez cesty) |

Chyby mají jednotný tvar `{"error": {"code", "message"}}` (413, 422, 429,
502, 503). Dotaz je omezený na 3–500 znaků a 4 KiB, zpracovává se jeden
naráz.

## Bezpečnost a rozpočet

- Dokumenty i otázky jsou nedůvěryhodná data v oddělených blocích; model nemá
  nástroje ani tajemství. Vložený pokyn v dokumentu (`orders.order-notes.3`)
  je součástí testů.
- Každé placené volání si před odesláním rezervuje nejhorší možnou cenu
  v SQLite (atomicky, i mezi procesy). Nevyúčtovaná nebo přerušená volání
  zůstávají započtená. Celkový strop útraty je nejvýš $3; po jeho dosažení
  server přestane odpovídat. Požadavky vynucují standardní cenovou úroveň.
- API klíč nastavuje správce v proměnné prostředí nebo v `.env`; nikdy se
  neobjeví v záznamech, odpovědích ani logu. API naslouchá pouze na
  127.0.0.1, webové rozhraní má striktní Content Security Policy.

## Vývoj a testy

```sh
uv run ruff check && uv run ruff format --check
uv run pytest
cd frontend && npm test && npm run typecheck
```

Testy jsou deterministické, se zablokovanou sítí a falešnými modely; nikdy
nevolají placené API ani nečtou lokální klíč. Stejné kontroly běží v GitHub
Actions.

## Omezení

- Jeden dotaz naráz a jen lokální přístup (127.0.0.1); uživatelské účty
  nejsou. Pro tým je potřeba reverzní proxy s přihlášením a souběžné
  zpracování dotazů.
- Jen text: bez OCR, tabulek a obrázků. Dokumenty se přidávají na serveru
  příkazy `import` a `ingest`, ne přes webové rozhraní.
- Jednotlivé otázky bez konverzace; bez streamování, hybridního vyhledávání
  a rerankingu.
- Prompty jsou napsané pro testovací dokumenty (viz *Vlastní dokumenty*).
- Ověření kombinuje deterministické kontroly a posouzení modelem, takže není
  důkazem pravdivosti; jeho chybovost se měří. Data jsou fiktivní.
