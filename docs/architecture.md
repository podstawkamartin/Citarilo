# Architektura Citarila

> Modely, ceny a verze knihoven ověřeny 6. 11. 2026. Odhady nákladů jsou
> orientační; skutečnou spotřebu měří záznam každého běhu.

## Jak zpracování dotazu funguje

```mermaid
flowchart TD
    Q(["POST /api/ask<br/>otázka EN nebo CS"]) --> V["Validace a limity"]
    V -->|neplatný vstup| E422[["422"]]
    V --> B["Rezervace rozpočtu<br/>SQLite, strop $3"]
    B -->|rozpočet vyčerpán| E503[["503"]]
    B --> R
    subgraph LG ["LangGraph StateGraph"]
        R["retrieve<br/>embedding → 5 sekcí"] --> G["generate<br/>gpt-6-luna: tvrzení + ID vět"]
        G -->|nedostatek podkladů| X["refuse<br/>pevná odpověď"]
        G --> C["check<br/>deterministické kontroly"]
        C -->|kontrola selhala| X
        C --> VR["verify<br/>gpt-6-luna: podpora tvrzení"]
        VR -->|nepodloženo| X
        VR --> A["assemble<br/>jen ověřená tvrzení"]
    end
    A --> REC["Záznam běhu a vyúčtování"]
    X --> REC
    REC --> OUT(["200 JSON<br/>answered / refused"])
    REC -.->|chyba poskytovatele| E502[["502"]]
```

- Neověřený text nikdy neopustí server. Odpověď skládá kód pouze
  z tvrzení, která prošla kontrolami, proto se nestreamuje.
- Odmítnutí znamená chybějící nebo nepodložené důkazy (HTTP 200 s kódem
  důvodu). Technická chyba nebo vyčerpaný rozpočet je chyba (502/503).
- Validace, limity, rozpočet a uložení záznamu jsou mimo graf. Graf řídí
  pouze rozhodování o důkazech.
- Záznam běhu a vyúčtování se uloží i při chybě poskytovatele.

## Příkazy a data

```mermaid
flowchart LR
    D["data/trailsupply/*.md"] --> ING["citarilo ingest"] --> I[("Qdrant<br/>.runtime/qdrant")]
    API["citarilo serve<br/>FastAPI"] --> P["LangGraph pipeline"]
    CASES["evals/*.jsonl"] --> EV["citarilo eval --live"] --> P
    I --> P
    P --> RUNS[("runs/*.json")]
    ING -.-> L[("spend.sqlite")]
    P -.-> L
```

Index, záznamy běhů i útrata leží v Gitem ignorované `.runtime/`.

## Struktura a komponenty

```text
pyproject.toml  uv.lock  .env.example  README.md
src/citarilo/              balíček (src layout), příkaz `citarilo` přes [project.scripts]
data/trailsupply/*.md      české fiktivní dokumenty
evals/cases.jsonl          vývojové a odložené případy
evals/verifier_cases.jsonl sondy pro měření ověřovače
tests/
.github/workflows/ci.yml   ruff + pytest bez sítě a klíčů
```

| Modul | Odpovědnost |
|---|---|
| `config.py` | Nastavení, klíče (`SecretStr`), modely, ceník, limity |
| `corpus.py` | Markdown → sekce a věty se stabilními ID, hash korpusu |
| `importer.py` | Lokální PDF/DOCX/Markdown → normalizovaný korpus bez sítě |
| `index.py` | Embeddingy se spotřebou, index s manifestem, vyhledávání |
| `pipeline.py` | LangGraph graf, prompty, schémata, kontroly |
| `budget.py` | Rezervace a vyúčtování v SQLite, strop |
| `records.py` | Schéma záznamu běhu a odpovědi API |
| `api.py` | FastAPI endpointy, chybová obálka |
| `cli.py` | `ingest`, `ask`, `serve`, `eval` |
| `frontend/` | React + TypeScript (Vite): otázka, ověřená odpověď, citace a zvýraznění věty v dokumentu; build servíruje `serve` |
| `evals.py` | Spuštění živé evaluace a vyhodnocení |

## LangChain a LangGraph

| Kde | API | Proč |
|---|---|---|
| Řízení toku | `StateGraph`, podmíněné hrany | Ověřovací brány jako explicitní a testovatelný graf |
| Závislosti uzlů | `context_schema`, `Runtime` | Modely a záznamník běhu bez globálních proměnných |
| Prompty | `ChatPromptTemplate` | Oddělené role a proměnné; dokumenty se vkládají jen jako hodnoty |
| Generování | `ChatOpenAI` (Responses API), `with_structured_output(method="json_schema", strict=True, include_raw=True)` | Striktní schéma výstupu a spotřeba z `usage_metadata` |
| Vyhledávání | `Document`, `SectionIndex.search`, `QdrantClient.query_points` | Přímý klient Qdrantu v lokálním režimu; kosinové skóre, ID a text vět v metadatech |
| Testy | `DeterministicFakeEmbedding`, `RunnableLambda` | Deterministické testy bez sítě a placených volání |

- Embeddingy: OpenAI API spotřebu vrací, ale rozhraní LangChain
  `Embeddings` ji zahazuje. Tenký adaptér nad OpenAI SDK implementuje
  totéž rozhraní a spotřebu zachová.
- Časy uzlů měří malý obal s `perf_counter`, který zachytí i selhání.
- Živý kontext vlastní a zavírá sdíleného HTTP klienta pro embeddingy i oba modely s expirací nečinných spojení 60 s.
- Neparsovatelný výstup modelu je chyba 502, ne odmítnutí.

## Modely a nastavení

| Role | Model | Cena za 1M tokenů |
|---|---|---|
| Generování a ověřování | `gpt-6-luna` | $0,10 vstup / $0,50 výstup |
| Embeddingy | `text-embedding-3-large` | $0,13 |

- Reasoning effort: `none` pro generování, `low` pro ověřování. Výchozí
  volba; změnu rozhoduje evaluace.
- `store=False`, `max_retries=0`, časový limit a limit výstupních tokenů
  u každého volání. Verze knihoven zamknuté v `uv.lock`.
- Model, nastavení a verze ceníku se ukládají do každého záznamu běhu.
- Pokud evaluace ukáže slabou češtinu, záloha je `gpt-6.1-terra`
  ($1 / $5): změna jednoho řádku konfigurace.
- Orientačně: dotaz ≈ $0,001 (≈ 0,02 Kč), evaluace 20 případů ≈ 0,5 Kč.

## Rozpočet

- Každý placený vstup (`ingest`, `ask`, `eval`) před voláním rezervuje
  v SQLite nejvyšší možnou cenu podle limitů vstupu a výstupu. Po volání
  se vyúčtuje skutečná spotřeba.
- Timeout, pád procesu nebo chybějící údaj o spotřebě: rezervace zůstává
  započtená jako útrata.
- Strop $3 pro OpenAI; evaluace má navíc vlastní `--max-usd`.
- Žádné automatické opakování ani přepnutí na jiný model.

## Korpus a vyhledávání

- Dodávané testovací dokumenty: šest českých Markdown dokumentů fiktivního
  e-shopu TrailSupply, každý zřetelně označený jako fiktivní.
- Sekce H2 nese explicitní anglické ID (`## Lhůta pro vrácení {#return-window}`)
  a je jednotkou vyhledávání. Jedna věta na řádek je jednotka citace:
  `returns.return-window.2`.
- ID platí vždy spolu s hashem korpusu, který obsahuje každý záznam běhu.
- Pouze vektorové vyhledávání, 5 nejbližších sekcí. Index se staví jen
  explicitním `citarilo ingest`; manifest obsahuje model, dimenze, verzi
  dělení textu, hash korpusu, počet sekcí a kontrolní součet bodů. Nesoulad → 503.
- Qdrant běží ve vestavěném lokálním režimu (`.runtime/qdrant`), bez samostatného
  serveru. Kolekce `sections` používá přesné kosinové vyhledávání. Manifest v2
  obsahuje kontrolní součet textu, metadat i vektorů přečtených zpět z úložiště.
  Odhalí poškození a nekonzistentní sestavení, nechrání před útočníkem
  s možností lokálního zápisu. Načtení nevolá model.
- Aplikační výhradní zámek `.runtime/index.lock` se získává před indexací
  a před otevřením indexu, včetně intervalu zavření a opětovného otevření
  Qdrantu. Indexace po úspěšném embeddingu odstraní celé staré úložiště
  a vytvoří nové, takže opraví i poškozený index bez načtení jeho obsahu.
- Index se ověřuje při startu. Pouze připravený server drží tento stav
  i zámek do ukončení; nepřipravený server při sondě index otevře a zavře.
  Před `ingest`, CLI `ask` nebo `eval` server zastavte. Po indexaci ho znovu
  spusťte. Chybějící či neplatný index zámek nedrží; serverový režim Qdrantu
  ani automatické přenačítání nejsou zapojené.
- Otázky anglicky i česky (i bez diakritiky). Odpověď je v jazyce otázky
  (`answer_language`, pro atribut `lang` ve frontendu), zdroje v původní
  češtině. Rozhraní, kód a důvody ověřovače jsou anglicky.

## Import vlastních dokumentů

`citarilo import SRC... --out DIR [--force]` převádí lokální PDF s textovou
vrstvou, DOCX a UTF-8 Markdown. Nečte nastavení ani klíče, nevolá síť,
pluginy, OCR ani modely.
Import je určen pro soubory dodané správcem, nikoli pro nedůvěryhodné uploady.
Skenované PDF bez textu odmítne; OCR musí správce zajistit předem.
Text i rozvržení PDF čte pdfplumber (MIT), odstavce a styly DOCX čte
python-docx (MIT), XML archivů chrání defusedxml (PSF). Markdown se čte
přímo jako UTF-8 text.

Výstup je standardní korpus: H1, volitelná citace v úvodu, H2 sekce a jedna
věta na řádek. ID dokumentu vychází z názvu souboru; ID sekce z nadpisu bez
diakritiky, kolize dostávají přípony `-2`, `-3`. Pořadí vět určuje jejich ID.
Deterministická česká pravidla chrání data, časy, pořadová čísla, zkratky,
částky, URL a e-maily. Import spojuje zalomené řádky; odstraní pouze měkké
spojovníky, tvrdé zachová a jejich spojení vypíše ke kontrole. U PDF
odstraňuje pouze opakované texty a číslování v okrajích a odhaduje nadpisy podle
velikosti písma. Text před první sekcí zachová v `Introduction` s upozorněním.
Nejednoznačné zkratky na konci vět a složité PDF sazby vyžadují kontrolu
normalizovaného výstupu. PDF tabulky a více sloupců odmítá; v DOCX odmítá
tabulky, sledované změny, ovládací prvky obsahu a textová pole. Správce musí
změny přijmout či odmítnout a nepodporované struktury převést na prostý text
nebo Markdown před importem.

Limit vstupu je 20 MiB na soubor, PDF nejvýše 100 stran, DOCX nejvýše
2 000 ZIP položek a 40 MiB rozbaleného obsahu. Makra, vložené objekty,
externí DOCX vztahy a XML entity jsou odmítnuty. Platí také limity korpusu
(256 KiB na výsledný dokument, 4 096 znaků na řádek). Nad 600 znaků na větu
se vypíše upozornění. Celá dávka se validuje před publikací souborů; existující
cíl vyžaduje `--force`. Původní zdroje nelze přepsat ani s tímto přepínačem.

Správce nastaví `DATA_DIR` na výsledný adresář a explicitně spustí
`citarilo ingest` při zastaveném serveru a potom server znovu zapne.
Změněná ID mění hash korpusu; starý index se při startu odmítne
standardní kontrolou manifestu. Testovací surové dokumenty v
`examples/trailsupply-raw/` sestaví `uv run python scripts/build_sample_documents.py`
ze stejných vět jako `data/trailsupply`. Písmo Liberation Sans je přiloženo
s licencí SIL OFL 1.1 a v PDF vloženo. ReportLab (BSD) je závislost pouze
pro vývoj.

## Ověřování citací

1. **generate** vrací jazyk odpovědi a nejvýše 4 atomická tvrzení, každé
   s 1–3 ID vět z vyhledaných sekcí.
   Odpovědi uvádějí pouze dokumentované hodnoty bez odvozených výpočtů;
   pojistkou zůstává kontrola literálů.
2. **check** (kód): ID musí patřit k vyhledaným větám; čísla, data a URL
   z tvrzení musí být po normalizaci (české a anglické formáty) v citovaných
   větách; anglická odpověď nesmí obsahovat českou prózu. Jde o filtr,
   ne důkaz.
3. **verify** (samostatné volání modelu): podporu musí nést samotné
   citované věty. Celé sekce slouží jen k odhalení podmínek, výjimek
   a rozporů. Každé tvrzení má právě jeden verdikt; chybějící nebo
   duplicitní verdikt je selhání.
4. Odpověď vznikne, jen když jsou podložená všechna tvrzení. Jinak
   odmítnutí s kódem `insufficient_evidence`, `check_failed`,
   `unsupported` nebo `contradicted`.

Ověření kombinuje deterministické filtry s kontrolou modelem, takže není
důkazem pravdivosti. Jeho chybovost měříme (viz Evaluace).

## HTTP API

- `POST /api/ask` → `run_id`, `mode`, `status`, kód důvodu odmítnutí,
  tvrzení s citacemi, zdrojové věty, metriky (latence, tokeny, cena,
  časy uzlů).
- `GET /api/health`, `GET /api/documents/{id}`.
- Chyby: `{"error": {"code", "message"}}`.
- Frontend vykresluje pouze prostý text; citace přicházejí strukturovaně.

## Bezpečnost

- Lokální Qdrant deserializuje body pomocí `pickle` před kontrolou součtu;
  `.runtime/` je důvěryhodný lokální stav zapisovaný pouze Citarilem a úložiště
  Qdrantu se nesmí kopírovat z nedůvěryhodných zdrojů.

- Dokumenty jsou nedůvěryhodná data v oddělených blocích. Model nemá
  nástroje ani tajemství.
- Vstup: otázka 3–500 znaků, tělo požadavku nejvýše 4 KiB, žádná
  neznámá pole, 1 souběžný dotaz.
- API klíč nastavuje správce na serveru (proměnná prostředí nebo `.env`);
  nikdy se neobjeví v záznamech, logech, odpovědích ani frontendu. Externí
  tracing je vypnutý.
- API ve výchozím nastavení naslouchá pouze na 127.0.0.1.

## Evaluace a testy

- **Případy:** 15 vývojových a 5 odložených. Odložené se při ladění
  nepoužívají a spouštějí se jednou pro každou finální verzi promptu. Každý případ má ručně určený
  očekávaný stav, povinná fakta, zakázaná tvrzení a podpůrné věty.
  Pokrytí: angličtina, čeština, bez diakritiky, více sekcí,
  nezodpověditelné otázky, prompt injection v otázce i v dokumentu.
- **Hodnocení:** odpověď je správná, když má očekávaný stav, všechna
  povinná fakta, žádné zakázané tvrzení a každá citace tvrzení podporuje.
  Podporu citací u každé živé odpovědi hodnotí offline nezávislý LLM
  hodnotitel (silnější model jiného typu než aplikace) s namátkovou ruční
  kontrolou. README to uvádí jako hodnocení modelem, ne lidskou anotaci.
- **Brány přijetí** (stanovené před laděním): recall@5 ≥ 0,9; správně
  zodpovězeno ≥ 80 % zodpověditelných; 0 zodpovězených nezodpověditelných;
  0 úspěšných injection; 0 falešných přijetí v sondách ověřovače. Změnu
  brány musí zdůvodnit README.
- **Přínos ověřovače:** z uložených kandidátních tvrzení měříme, kolik
  nepodložených tvrzení `verify` zachytil a kolik správných odmítl,
  bez dalších placených volání.
- **Výkon:** latence p50 a maximum (p95 při 20 vzorcích nemá vypovídací
  hodnotu), tokeny, cena.
- **Testy** jsou deterministické, se zablokovanou sítí a falešnými modely;
  běží lokálně i v CI. Živá evaluace pouze `citarilo eval --live --max-usd X`.

## Záznam běhu

Každý běh ukládá: `schema_version`, `run_id`, čas, `mode` (`live`/`fake`),
původ (API nebo případ evaluace), Git commit, hash korpusu, verzi promptů,
konfiguraci (modely, reasoning, `top_k`, limity), otázku, vyhledané sekce
se skóre, kandidátní tvrzení, výsledky kontrol, verdikty, výsledek,
spotřebu a cenu každého volání, časy uzlů a případnou chybu.

## Mimo rozsah

Streamování, hybridní vyhledávání a reranking, opravné
přegenerování, konverzace, nahrávání dokumentů, checkpointy a přerušení
v LangGraph, Docker, LangSmith, placený model v roli hodnotitele a napojení na
obchodní systémy. Každá položka se přidá až s doloženou potřebou.

LangGraph dělá tok explicitním a další kroky se přidávají jako uzly.
Škálování výkonu (více procesů, fronta, sdílené úložiště) neřeší.
