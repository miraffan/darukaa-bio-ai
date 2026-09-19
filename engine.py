"""Retrieval + multi-metric reasoning engine (no API keys needed)."""
import json
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Knowledge layer: structured KB + TF-IDF vector index ----------
KB = json.loads((Path(__file__).parent / "knowledge_base.json").read_text(encoding="utf-8"))
_docs = [
    f'{k["title"]} {k["action"]} {k["why"]} {" ".join(k["metrics"])} {" ".join(k["triggers"])}'
    for k in KB
]
_vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
_mat = _vec.fit_transform(_docs)

# Cross-variable linkages: (required issue codes, explanation, source)
LINKS = [
    ({"low_soc", "water_limited"},
     "Low soil organic carbon weakens aggregate stability and water-holding, so scarce rain is lost to runoff and evaporation. Soil carbon and water capture must be raised together.",
     "Lal (2004), Science 304:1623-1627; FAO (2017)"),
    ({"monoculture", "fragmentation"},
     "Monoculture at field scale plus fragmented habitat at landscape scale removes both food and refuge for pollinators and natural enemies, so field margins and corridors matter more than either alone.",
     "Tscharntke et al. (2005), Ecology Letters 8:857-874"),
    ({"acidic", "low_soc"},
     "Acidity suppresses decomposition and legume nodulation, so legume cover crops may underperform until pH is corrected. Sequence liming before cover crops.",
     "Fierer & Jackson (2006), PNAS 103:626-631"),
    ({"heat", "water_limited"},
     "Heat raises evapotranspiration and speeds soil carbon decomposition, compounding drought stress on species survival. Shade and mulch address both.",
     "IPCC AR6 WGII (2022) Ch. 5"),
    ({"pollution", "monoculture"},
     "Monocultures tend to depend on more pesticide, which suppresses the natural enemies that would otherwise reduce that dependence.",
     "Geiger et al. (2010), Basic Appl. Ecol. 11:97-105"),
    ({"deforestation", "water_limited"},
     "Losing tree cover reduces infiltration and local moisture recycling, making dry land drier. Restoring cover feeds back into water availability.",
     "IPCC SRCCL (2019)"),
]

REQUIRED = {
    "soc": "soil organic carbon (%)",
    "rainfall": "rainfall pattern (low / medium / high, or mm per year)",
    "land_use": "land-use type / crop (e.g. monoculture wheat, pasture, forest)",
}
OPTIONAL = {"ph": "soil pH", "region": "climate region (arid, semi-arid, humid, temperate...)"}

CROPS = ["wheat", "rice", "maize", "corn", "cotton", "soy", "sugarcane", "millet", "sorghum", "barley"]
LAND_WORDS = {"pasture": "pasture", "grazing": "pasture", "forest": "forest", "plantation": "plantation",
              "wetland": "wetland", "orchard": "orchard", "cropland": "cropland", "farm": "cropland"}


# ---------- Input handling ----------
def extract(text: str) -> dict:
    t = text.lower()
    o = {}
    m = (re.search(r"(?:organic carbon|soc)\D{0,20}?(\d+(?:\.\d+)?)\s*%", t)
         or re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:soc|organic carbon)", t))
    if m:
        o["soc"] = float(m.group(1))
    m = re.search(r"\bph\b\D{0,8}(\d+(?:\.\d+)?)", t)
    if m:
        o["ph"] = float(m.group(1))
    m = re.search(r"(\d{2,4})\s*mm", t)
    if m:
        o["rainfall_mm"] = float(m.group(1))
    elif re.search(r"(low|scanty|erratic|poor)\W+(rain|rainfall)|rainfall\W+(is\s+)?(low|poor)|drought|\bdry\b", t):
        o["rainfall"] = "low"
    elif re.search(r"(high|heavy)\W+(rain|rainfall)|rainfall\W+(is\s+)?high", t):
        o["rainfall"] = "high"
    elif re.search(r"(medium|moderate)\W+(rain|rainfall)|rainfall\W+(is\s+)?(medium|moderate)", t):
        o["rainfall"] = "medium"
    m = re.search(r"(\d{2}(?:\.\d+)?)\s*(?:°|deg(?:rees)?)\s*c", t)
    if m:
        o["temp_c"] = float(m.group(1))
    m = re.search(r"richness\D{0,15}(\d+)", t)
    if m:
        o["species_richness"] = int(m.group(1))
    crop = next((c for c in CROPS if c in t), None)
    if crop:
        o["crop"] = crop
    if re.search(r"monoculture|monocrop|single crop", t):
        o["land_use"] = "monoculture"
    else:
        lu = next((v for k, v in LAND_WORDS.items() if k in t), None)
        if lu:
            o["land_use"] = lu
        elif crop:
            o["land_use"] = "cropland"
    for r in ["semi-arid", "semiarid", "arid", "humid", "tropical", "temperate", "mediterranean", "boreal"]:
        if r in t:
            o["region"] = "semi-arid" if r == "semiarid" else r
            break
    if re.search(r"pesticide|fertili[sz]er|pollut|contaminat", t):
        o["pollution"] = True
    if re.search(r"deforest|logging|clear(?:ed|ing)", t):
        o["deforestation"] = True
    if re.search(r"fragment|isolated|patches", t):
        o["fragmentation"] = True
    if re.search(r"declin|decreas|dropp|loss of (?:species|biodiversity)", t):
        o["declining"] = True
    return o


def normalize(p: dict) -> dict:
    """Derive categories from numbers; infer zone from lat if given (spatial bonus)."""
    p = dict(p)
    if "rainfall_mm" in p and "rainfall" not in p:
        mm = p["rainfall_mm"]
        p["rainfall"] = "low" if mm < 500 else "medium" if mm < 1200 else "high"
    if "rainfall_mm" in p and p["rainfall"] != ("low" if p["rainfall_mm"] < 500 else "medium" if p["rainfall_mm"] < 1200 else "high"):
        mm = p["rainfall_mm"]
        p["rainfall"] = "low" if mm < 500 else "medium" if mm < 1200 else "high"
    lat = p.get("lat")
    if lat is not None and "region" not in p:
        a = abs(float(lat))
        p["climate_zone_from_lat"] = "tropical" if a < 23.5 else "subtropical" if a < 35 else "temperate" if a < 55 else "boreal"
    return p


def merge(profile: dict, new: dict) -> dict:
    """Conversation memory: newer information overrides older."""
    out = dict(profile)
    out.update({k: v for k, v in new.items() if v is not None})
    return normalize(out)


def questions(p: dict, include_optional: bool = True) -> list:
    qs = [f"Could you share your {v}?" for k, v in REQUIRED.items() if k not in p]
    if not qs and include_optional:
        qs = [f"(Optional, improves precision) What is your {v}?" for k, v in OPTIONAL.items() if k not in p]
    return qs


# ---------- Diagnosis (multi-variable) ----------
def diagnose(p: dict) -> dict:
    d = {}
    soc = p.get("soc")
    if soc is not None and soc < 1.0:
        d["low_soc"] = f"Very low soil organic carbon ({soc}%): weak structure, low microbial food supply"
    ph = p.get("ph")
    if ph is not None and ph < 5.5:
        d["acidic"] = f"Acidic soil (pH {ph}): suppresses microbial diversity and nutrient availability"
    if ph is not None and ph > 8.0:
        d["alkaline"] = f"Alkaline soil (pH {ph}): reduced micronutrient availability"
    zone = p.get("region") or p.get("climate_zone_from_lat")
    if p.get("rainfall") == "low" or zone in ("arid", "semi-arid"):
        d["water_limited"] = "Water-limited conditions (low rainfall / arid climate): survival of species depends on moisture retention"
    if p.get("temp_c") is not None and p["temp_c"] >= 30:
        d["heat"] = f"High mean temperature ({p['temp_c']} C): raises evaporation and carbon loss"
    if p.get("land_use") == "monoculture":
        d["monoculture"] = "Monoculture land use: low habitat and structural diversity"
    if p.get("fragmentation"):
        d["fragmentation"] = "Habitat fragmentation: isolated patches limit species movement"
    if p.get("pollution"):
        d["pollution"] = "Chemical inputs / pollution pressure"
    if p.get("deforestation"):
        d["deforestation"] = "Deforestation / loss of tree cover"
    if p.get("declining"):
        d["declining"] = "Reported biodiversity decline"
    return d


# ---------- Retrieval + ranking (hybrid: vector similarity + trigger match) ----------
def recommend(p: dict, k: int = 4):
    issues = diagnose(p)
    codes = set(issues)
    query = " ".join(codes) + " " + " ".join(issues.values()) + " " + " ".join(str(v) for v in p.values())
    sims = cosine_similarity(_vec.transform([query]), _mat)[0]
    rows = []
    for i, e in enumerate(KB):
        hit = codes & set(e["triggers"])
        if hit:
            rows.append({"entry": e, "hit": sorted(hit), "sim": float(sims[i]), "score": 0.4 * len(hit) + float(sims[i])})
    rows.sort(key=lambda r: r["score"], reverse=True)
    links = [(txt, src) for req, txt, src in LINKS if req <= codes]
    return issues, rows[:k], links


def render(issues: dict, recs: list, links: list) -> str:
    if not issues:
        return "I could not detect a clear stressor from the inputs. Please share soil, rainfall and land-use details."
    md = ["### Diagnosis"] + [f"- {v}" for v in issues.values()]
    if links:
        md.append("\n### Multi-metric linkages")
        md += [f"- {t} *({s})*" for t, s in links]
    md.append("\n### Recommendations")
    for n, r in enumerate(recs, 1):
        e = r["entry"]
        md.append(
            f"**{n}. {e['title']}**  \n"
            f"- **What to do:** {e['action']}  \n"
            f"- **Why it works:** {e['why']}  \n"
            f"- **Metrics improved:** {', '.join(e['metrics'])}  \n"
            f"- **Expected effect:** {e['effect']}  \n"
            f"- **Time horizon:** {e['horizon']}  \n"
            f"- **Confidence:** {e['confidence']}  \n"
            f"- **Addresses:** {', '.join(r['hit'])}  \n"
            f"- **Source:** {e['source']}\n"
        )
    return "\n".join(md)
