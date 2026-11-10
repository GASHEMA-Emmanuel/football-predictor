"""European club cups and international football (all free, no API keys).

Fixtures and results: https://fixturedownload.com (Champions League, Europa League, Conference
League, UEFA Nations League), times in UTC.
International results since 1872: https://github.com/martj42/international_results

Club cups are predicted with ONE joint model fitted on every domestic league we have plus the
European cup results; the cup matches are what link the leagues, so a Dutch team can be rated
against a Spanish one. National teams get their own model on international results.
"""
from __future__ import annotations

import difflib
import io
import time

import pandas as pd
import requests

import data

FIXTUREDOWNLOAD = "https://fixturedownload.com/feed/json"
INTL_RESULTS = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

CUPS = {
    "UCL": ("UEFA – Champions League", "champions-league"),
    "UEL": ("UEFA – Europa League", "europa-league"),
    "UECL": ("UEFA – Conference League", "conference-league"),
}
INTERNATIONAL = {
    "UNL": ("UEFA – Nations League", "nations-league"),
}
COMPETITIONS = {code: name for code, (name, _) in {**CUPS, **INTERNATIONAL}.items()}
DEFAULT_PICK_SET = data.TOP10 + list(COMPETITIONS)  # what the cross-competition pages use

# fixturedownload name -> football-data name, where fuzzy matching would get it wrong
CLUB_ALIASES = {
    "PSV": "PSV Eindhoven", "Paris": "Paris SG", "Paris Saint-Germain": "Paris SG",
    "Man City": "Man City", "Manchester City": "Man City", "Man Utd": "Man United",
    "Manchester United": "Man United", "Spurs": "Tottenham", "Nottm Forest": "Nott'm Forest",
    "Nottingham Forest": "Nott'm Forest", "Atleti": "Ath Madrid", "Atlético de Madrid": "Ath Madrid",
    "Atletico Madrid": "Ath Madrid", "Athletic Club": "Ath Bilbao", "Real Betis": "Betis",
    "Real Sociedad": "Sociedad", "Celta Vigo": "Celta", "Rayo Vallecano": "Vallecano",
    "Inter": "Inter", "Internazionale": "Inter", "AC Milan": "Milan", "Bayern München": "Bayern Munich",
    "Bayern Munich": "Bayern Munich", "B. Dortmund": "Dortmund", "Borussia Dortmund": "Dortmund",
    "Leverkusen": "Leverkusen", "Bayer Leverkusen": "Leverkusen", "Frankfurt": "Ein Frankfurt",
    "Eintracht Frankfurt": "Ein Frankfurt", "M'gladbach": "M'gladbach", "Mönchengladbach": "M'gladbach",
    "Köln": "FC Koln", "Marseille": "Marseille", "Lyon": "Lyon", "Sporting CP": "Sp Lisbon",
    "Sporting": "Sp Lisbon", "Braga": "Sp Braga", "SC Braga": "Sp Braga", "Vitória SC": "Guimaraes",
    "Club Brugge": "Club Brugge", "Union SG": "St. Gilloise", "Union Saint-Gilloise": "St. Gilloise",
    "Royale Union SG": "St. Gilloise", "Gent": "Gent", "AZ": "AZ Alkmaar", "AZ Alkmaar": "AZ Alkmaar",
    "Twente": "Twente", "Fenerbahçe": "Fenerbahce", "Beşiktaş": "Besiktas", "Başakşehir": "Basaksehir",
    "Celtic": "Celtic", "Rangers": "Rangers", "Hearts": "Hearts", "Aberdeen": "Aberdeen",
    "Olympiacos": "Olympiakos", "Olympiakos": "Olympiakos", "PAOK": "PAOK", "AEK Athens": "AEK",
    "Panathinaikos": "Panathinaikos", "Crystal Palace": "Crystal Palace",
    "Leipzig": "RB Leipzig", "RB Leipzig": "RB Leipzig", "N.E.C.": "Nijmegen", "NEC": "Nijmegen",
    "Sint-Truidense": "St Truiden", "Sint-Truiden": "St Truiden", "Standard Liège": "Standard",
    "Royal Antwerp": "Antwerp", "Başakşehir FK": "Buyuksehyr", "Istanbul Basaksehir": "Buyuksehyr",
    "Göztepe": "Goztep", "Trabzonspor": "Trabzonspor", "Feyenoord": "Feyenoord",
    "Vitória de Guimarães": "Guimaraes", "Hibernian": "Hibernian", "Hibs": "Hibernian",
}
# the same foreign club spelled two ways in the cup feeds (kept as one team)
CLUB_SPELLINGS = {"Omonoia": "Omonia", "Crvena Zvezda": "Crvena Zvezda", "GNK Dinamo": "Dinamo Zagreb"}

NATION_ALIASES = {  # fixturedownload -> international_results spelling
    "Czechia": "Czech Republic", "Türkiye": "Turkey", "Republic of Ireland": "Republic of Ireland",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina", "North Macedonia": "North Macedonia",
    "Korea Republic": "South Korea", "IR Iran": "Iran", "USA": "United States",
}

# competitions whose results say less about real strength
FRIENDLY_WEIGHT = 0.6

_CACHE: dict[str, tuple[float, object]] = {}


def _cached(key: str, seconds: int, fn):
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < seconds:
        return hit[1]
    value = fn()
    _CACHE[key] = (time.time(), value)
    return value


def load_fixturedownload(slug: str, start_year: int) -> pd.DataFrame:
    """All matches of one competition season, played or not. Empty if the feed doesn't exist."""
    cols = ["Date", "Time", "Kickoff", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Round"]

    def fetch():
        r = requests.get(f"{FIXTUREDOWNLOAD}/{slug}-{start_year}", headers=data.HEADERS, timeout=30)
        if r.status_code == 404:
            return []
        r.raise_for_status()
        return r.json()

    rows = _cached(f"{slug}-{start_year}", 1800, fetch)
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    kick = pd.to_datetime(df["DateUtc"], utc=True, errors="coerce").dt.tz_convert(data.UK)
    out = pd.DataFrame({
        "Kickoff": kick,
        "Date": kick.dt.tz_localize(None).dt.normalize(),
        "Time": kick.dt.strftime("%H:%M"),
        "HomeTeam": df["HomeTeam"].str.strip(), "AwayTeam": df["AwayTeam"].str.strip(),
        "FTHG": pd.to_numeric(df["HomeTeamScore"], errors="coerce"),
        "FTAG": pd.to_numeric(df["AwayTeamScore"], errors="coerce"),
        "Round": df.get("RoundNumber"),
    })
    # placeholders such as "To be announced" / "Winner Match 3" are not real teams yet
    real = ~out["HomeTeam"].str.contains(r"announced|winner|loser|^tbd|^tba", case=False, na=True) \
        & ~out["AwayTeam"].str.contains(r"announced|winner|loser|^tbd|^tba", case=False, na=True)
    return out[real & out["Kickoff"].notna()].reset_index(drop=True)


def _season_years(seasons: int, today=None) -> list[int]:
    start = data.season_start_year(today)
    return list(range(start - seasons + 1, start + 1))


# ---------------- club cups ----------------
def cup_results(seasons: int = 2) -> pd.DataFrame:
    """Played cup matches (raw fixturedownload names), all three cups, last `seasons` seasons."""
    frames = []
    for code, (_, slug) in CUPS.items():
        for y in _season_years(seasons):
            df = load_fixturedownload(slug, y)
            df = df.dropna(subset=["FTHG", "FTAG"]).assign(Comp=code)
            frames.append(df)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not df.empty:
        df[["FTHG", "FTAG"]] = df[["FTHG", "FTAG"]].astype(int)
    return df


def map_club_names(cup_names: set[str], domestic_names: set[str]) -> dict[str, str]:
    """Cup team name -> domestic (football-data) name. Clubs from leagues we don't cover keep
    their cup name; they are rated from their cup results only."""
    mapping: dict[str, str] = {}
    by_norm = {data._norm(n): n for n in domestic_names}
    for raw in sorted(cup_names):
        if raw in CLUB_SPELLINGS:
            mapping[raw] = CLUB_SPELLINGS[raw]
        elif raw in CLUB_ALIASES and CLUB_ALIASES[raw] in domestic_names:
            mapping[raw] = CLUB_ALIASES[raw]
        elif raw in domestic_names:
            mapping[raw] = raw
        else:
            hit = difflib.get_close_matches(data._norm(raw), list(by_norm), n=1, cutoff=0.86)
            if hit:
                mapping[raw] = by_norm[hit[0]]
    return mapping


def europe_matches(domestic: list[pd.DataFrame], seasons: int = 2) -> tuple[pd.DataFrame, dict]:
    """Domestic results + cup results with cup names mapped onto domestic names."""
    dom = pd.concat(domestic, ignore_index=True)[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]]
    cups = cup_results(seasons)
    mapping = map_club_names(set(cups["HomeTeam"]) | set(cups["AwayTeam"]),
                             set(dom["HomeTeam"]) | set(dom["AwayTeam"]))
    cups = cups.assign(HomeTeam=cups["HomeTeam"].replace(mapping),
                       AwayTeam=cups["AwayTeam"].replace(mapping))
    both = pd.concat([dom, cups[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]]], ignore_index=True)
    return both.sort_values("Date").reset_index(drop=True), mapping


def cup_fixtures(code: str, mapping: dict, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Upcoming matches of one cup, names mapped like the model's."""
    now = now or pd.Timestamp.now(tz=data.UK)
    _, slug = CUPS[code]
    df = load_fixturedownload(slug, data.season_start_year(now.date()))
    df = df[df["FTHG"].isna() & (df["Kickoff"] > now)].copy()
    df["HomeTeam"] = df["HomeTeam"].replace(mapping)
    df["AwayTeam"] = df["AwayTeam"].replace(mapping)
    return df.drop(columns=["FTHG", "FTAG"]).assign(Div=code).sort_values("Kickoff").reset_index(drop=True)


def cup_played(code: str, mapping: dict) -> pd.DataFrame:
    """This season's played matches of one cup (for the results table)."""
    _, slug = CUPS[code]
    df = load_fixturedownload(slug, data.season_start_year()).dropna(subset=["FTHG", "FTAG"]).copy()
    df["HomeTeam"] = df["HomeTeam"].replace(mapping)
    df["AwayTeam"] = df["AwayTeam"].replace(mapping)
    return df.sort_values("Kickoff", ascending=False)


# ---------------- international ----------------
def international_results(years: int = 8) -> pd.DataFrame:
    """Full-international results of the last `years` years with Neutral and Weight columns."""
    def fetch():
        r = requests.get(INTL_RESULTS, headers=data.HEADERS, timeout=60)
        r.raise_for_status()
        return r.text

    df = pd.read_csv(io.StringIO(_cached("intl", 6 * 3600, fetch)))
    df["Date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["Date", "home_score", "away_score"])
    df = df[df["Date"] >= pd.Timestamp.now() - pd.DateOffset(years=years)]
    return pd.DataFrame({
        "Date": df["Date"], "HomeTeam": df["home_team"], "AwayTeam": df["away_team"],
        "FTHG": df["home_score"].astype(int), "FTAG": df["away_score"].astype(int),
        "Tournament": df["tournament"],
        "Neutral": df["neutral"].astype(str).str.upper().eq("TRUE"),
        "Weight": df["tournament"].eq("Friendly").map({True: FRIENDLY_WEIGHT, False: 1.0}),
    }).sort_values("Date").reset_index(drop=True)


def nation_name(raw: str, known: set[str]) -> str:
    if raw in NATION_ALIASES:
        return NATION_ALIASES[raw]
    if raw in known:
        return raw
    hit = difflib.get_close_matches(raw, list(known), n=1, cutoff=0.85)
    return hit[0] if hit else raw


def international_fixtures(code: str, known: set[str], now: pd.Timestamp | None = None) -> pd.DataFrame:
    now = now or pd.Timestamp.now(tz=data.UK)
    _, slug = INTERNATIONAL[code]
    df = load_fixturedownload(slug, data.season_start_year(now.date()))
    df = df[df["FTHG"].isna() & (df["Kickoff"] > now)].copy()
    df["HomeTeam"] = df["HomeTeam"].map(lambda t: nation_name(t, known))
    df["AwayTeam"] = df["AwayTeam"].map(lambda t: nation_name(t, known))
    return df.drop(columns=["FTHG", "FTAG"]).assign(Div=code).sort_values("Kickoff").reset_index(drop=True)
