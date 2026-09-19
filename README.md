# AI Biodiversity Scientist (Darukaa.Earth Hackathon)

Retrieval-grounded conversational system that diagnoses land/soil/climate inputs and returns cited, multi-metric biodiversity recommendations.

## Architecture
```
User text / JSON -> extract() -> merge() [session memory] -> questions() [clarify if soc/rainfall/land use missing]
  -> diagnose() [issue codes: low_soc, acidic, water_limited, heat, monoculture, fragmentation, pollution, deforestation, declining]
  -> recommend(): TF-IDF vector index over knowledge_base.json + trigger matching -> ranked interventions
  -> LINKS: cross-variable reasoning (soil <-> water <-> habitat) -> render(): what / why / metrics / effect / horizon / confidence / source
```
The retrieval trace (entry, matched issues, similarity, score) is shown in the UI.

## Knowledge schema (`knowledge_base.json`)
`id, title, triggers[], action, why, metrics[], effect, horizon, confidence, source`

Extend it by adding entries (e.g. chunked from papers/reports). Swap TF-IDF for sentence-transformer embeddings + FAISS/Chroma in `engine.py` if desired.

## Local setup
```
pip install -r requirements.txt
streamlit run app.py
pytest
```

## CI/CD
GitHub Actions (`.github/workflows/ci.yml`) runs tests on every push. Streamlit Community Cloud auto-redeploys on push to `main` (CD).

## Note
Cited figures come from the referenced studies; verify exact numbers against the originals before citing in a formal report.

## Instant demo (no install)
`index.html` is a standalone, browser-only version of the same engine (same knowledge base, same retrieval and reasoning). Open it directly, or host it free on GitHub Pages: repo Settings > Pages > Deploy from branch `main` / root. The URL will be `https://<user>.github.io/<repo>/`.
