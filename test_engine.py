import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from engine import extract, merge, questions, recommend

def test_extract_and_recommend():
    p = merge({}, extract("Soil organic carbon 0.3%, low rainfall, monoculture wheat, semi-arid"))
    assert p["soc"] == 0.3 and p["land_use"] == "monoculture"
    issues, recs, links = recommend(p)
    assert {"low_soc", "water_limited", "monoculture"} <= set(issues)
    assert recs and links
    assert any("Agroforestry" in r["entry"]["title"] for r in recs)

def test_clarifying_questions():
    p = merge({}, extract("Biodiversity is declining on my land"))
    assert len(questions(p, include_optional=False)) == 3
