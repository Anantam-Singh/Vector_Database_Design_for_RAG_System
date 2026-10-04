"""Extra metadata tags for finer filtering: topic, source_type and corpus.

* topic (15 everyday subjects, e.g. "Animals & Nature", "Food & Nutrition") — zero-shot: each topic is described by a
  few short sentences; a passage gets the topic whose description its dense vector is closest to. It reuses the
  vectors already in Qdrant, so tagging 100k passages takes seconds and needs NO re-embedding or re-index.
* source_type (government, education, reference, health, community, news, organization, commercial) — rules on the
  website domain. Deterministic, explainable.
* corpus — "web" for MS MARCO passages, "internal" for documents added through the API (e.g. company policies).

The same functions tag new documents at upsert time, so filters work for live data too.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

TOPICS: dict[str, list[str]] = {
    "Health & Medicine": ["symptoms, causes and treatment of a disease or medical condition",
                          "medication dosage, side effects and prescription drugs",
                          "the human body, organs, blood, bones and anatomy",
                          "doctors, hospitals, pregnancy, surgery and medical tests"],
    "Food & Nutrition": ["recipes, cooking times and how to cook or bake food",
                         "calories, vitamins, protein and nutrition facts of food",
                         "ingredients, drinks, coffee, wine and restaurants"],
    "Animals & Nature": ["animals, pets, dogs, cats, birds, fish and wildlife",
                         "plants, trees, flowers, seeds and gardening",
                         "ecosystems, forests, oceans, the environment and natural habitats"],
    "Science": ["physics, chemistry, chemical elements and reactions",
                "biology, cells, genes, DNA and evolution",
                "astronomy, planets, space, rocks and the earth's geology"],
    "Technology": ["computers, software, programming and the internet",
                   "smartphones, electronics, apps, gadgets and networks"],
    "Money & Finance": ["taxes, loans, mortgages, banks and credit cards",
                        "how much something costs, prices and fees",
                        "stock market, investing, insurance and retirement savings"],
    "Law & Government": ["laws, legal rights, courts, lawyers and lawsuits",
                         "government, politics, elections and public policy",
                         "police, military, permits and regulations"],
    "Business & Careers": ["companies, business management, marketing and sales",
                           "careers, job duties and how to become a professional",
                           "salary, wages and pay for a job; employment and the workplace"],
    "Places & Travel": ["countries, cities and states and where they are located",
                        "travel, tourism, distances, flights and time zones",
                        "weather, climate and temperatures in a place"],
    "History & Society": ["historical events, wars, empires and ancient civilizations",
                          "religion, culture, traditions and holidays",
                          "population, society, social issues and demographics"],
    "Language & Education": ["definition and meaning of a word",
                             "grammar, synonyms, spelling and pronunciation",
                             "schools, college, degrees, courses and education",
                             "meaning and origin of a baby name"],
    "Home & Construction": ["home improvement, renovation and construction costs",
                            "plumbing, roofing, flooring, painting and house repairs",
                            "home appliances, furniture, cleaning and tools"],
    "Transport & Vehicles": ["cars, engines, tires, vehicle repair and driving",
                             "airplanes, airports, trains, ships and public transport"],
    "Arts & Entertainment": ["movies, TV shows, actors and celebrities",
                             "music, songs, bands and singers",
                             "books, authors, painting, art and video games"],
    "Sports & Fitness": ["sports, teams, athletes, matches and tournaments",
                         "exercise, fitness, workouts, running and weight loss"],
}
TOPIC_NAMES = list(TOPICS)

SOURCE_TYPES = ["government", "education", "reference", "health", "community", "news", "organization", "commercial"]
_REFERENCE = ("wikipedia.", "britannica.", "dictionary", "merriam-webster.", "definitions.net", "vocabulary.com",
              "audioenglish.", "wiktionary.", "thesaurus.", "encyclopedia", "babyname", "thinkbabynames.",
              "ourbabynamer.", "behindthename.", "collinsdictionary.", "oxforddictionaries.", "lexico.")
_HEALTH = ("webmd.", "mayoclinic.", "medicinenet.", "healthline.", "everydayhealth.", "medicalnewstoday.",
           "emedicinehealth.", "sharecare.", "rxlist.", "medscape.", "drugs.com", "rightdiagnosis.", "livestrong.",
           "clevelandclinic.", "healthgrades.", "verywell", "patient.info", "medlineplus.", "health", "medical",
           "clinic", "hopkinsmedicine.")
_COMMUNITY = ("answers.com", "answers.yahoo.", "quora.", "reddit.", "ehow.", "wikihow.", "wisegeek.", "stackexchange.",
              "stackoverflow.", "chacha.", "ask.com", "answerbag.", "blurtit.", "youtube.", "reference.com")
_NEWS = ("huffingtonpost.", "bbc.", "cnn.", "nytimes.", "usatoday.", "theguardian.", "forbes.", "washingtonpost.",
         "nbcnews.", "abcnews.", "foxnews.", "latimes.", "telegraph.", "independent.", "reuters.", "npr.org",
         "time.com", "usnews.", "chron.com", "sfgate.", "livescience.", "dailymail.", "businessinsider.", "cbsnews.")
_GOV = (".gov", ".mil", ".gc.ca", "europa.eu", "nhs.uk")          # ".gov" also matches .gov.uk, .gov.au, ...
_EDU = (".edu", ".ac.", ".k12.", "study.com", "khanacademy.", "sparknotes.", "quizlet.", "coursehero.", "education.com")


def source_type(domain: str) -> str:
    """Website domain → one of SOURCE_TYPES. First matching rule wins (most official first)."""
    d = "." + (domain or "").lower()     # leading dot so "gov.uk" matches ".gov" like "www.gov.uk" does
    if any(s in d for s in _GOV):
        return "government"
    if any(s in d for s in _EDU):
        return "education"
    if any(s in d for s in _REFERENCE):
        return "reference"
    if any(s in d for s in _HEALTH):
        return "health"
    if any(s in d for s in _COMMUNITY):
        return "community"
    if any(s in d for s in _NEWS):
        return "news"
    if d.endswith(".org") or ".org." in d:
        return "organization"
    return "commercial"


@lru_cache(maxsize=1)
def _topic_matrix() -> tuple[np.ndarray, np.ndarray]:
    """One row per seed sentence (passage-style embedding, normalised) and the topic index of each row."""
    from .encoders import get_dense
    seeds, owner = [], []
    for i, name in enumerate(TOPIC_NAMES):
        for s in TOPICS[name]:
            seeds.append(s)
            owner.append(i)
    return get_dense().encode_passages(seeds), np.array(owner)


def topic_scores(vectors) -> np.ndarray:
    """(n, 384) normalised vectors → (n, n_topics) similarity: best-matching seed sentence per topic."""
    seeds, owner = _topic_matrix()
    sims = np.asarray(vectors, dtype=np.float32) @ seeds.T
    out = np.full((sims.shape[0], len(TOPIC_NAMES)), -1.0, dtype=np.float32)
    for t in range(len(TOPIC_NAMES)):
        out[:, t] = sims[:, owner == t].max(axis=1)
    return out


def assign_topics(vectors) -> list[tuple[str, float]]:
    """Topic + confidence margin (best minus second-best similarity) for each vector."""
    sc = topic_scores(vectors)
    order = np.argsort(-sc, axis=1)
    rows = np.arange(len(sc))
    margin = sc[rows, order[:, 0]] - sc[rows, order[:, 1]]
    return [(TOPIC_NAMES[i], round(float(m), 4)) for i, m in zip(order[:, 0], margin)]


def query_topics(query_vec, top: int = 2) -> list[tuple[str, float]]:
    """Most likely topics of a QUESTION (its vector includes the BGE query instruction) with their similarity."""
    sc = topic_scores(np.asarray(query_vec, dtype=np.float32)[None, :])[0]
    order = np.argsort(-sc)[:top]
    return [(TOPIC_NAMES[i], round(float(sc[i]), 4)) for i in order]


def tags_for(vector, domain: str, corpus: str) -> dict:
    topic, conf = assign_topics(np.asarray(vector, dtype=np.float32)[None, :])[0]
    return {"topic": topic, "topic_conf": conf, "source_type": source_type(domain), "corpus": corpus}
