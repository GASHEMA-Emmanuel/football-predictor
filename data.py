"""Data access (all free, no API keys).

Results:  https://www.football-data.co.uk/mmz4281/{season}/{div}.csv
Fixtures: https://www.football-data.co.uk/fixtures.csv  (current round, all leagues, with odds)
Schedule: https://github.com/openfootball/football.json  (full-season fixture lists, no odds)

fixtures.csv is only refreshed a couple of times a week, so it keeps listing a round after it
has been played. We therefore drop anything that has already kicked off, and fall back to the
openfootball season schedule for the leagues it covers.
"""
from __future__ import annotations

import datetime as dt
import difflib
import io
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from zoneinfo import ZoneInfo

import pandas as pd
import requests

BASE = "https://www.football-data.co.uk"
HEADERS = {"User-Agent": "football-predictor/1.0"}

LEAGUES = {
    "E0": "England – Premier League",
    "E1": "England – Championship",
    "E2": "England – League One",
    "E3": "England – League Two",
    "SC0": "Scotland – Premiership",
    "SP1": "Spain – La Liga",
    "SP2": "Spain – Segunda",
    "I1": "Italy – Serie A",
    "I2": "Italy – Serie B",
    "D1": "Germany – Bundesliga",
    "D2": "Germany – 2. Bundesliga",
    "F1": "France – Ligue 1",
    "F2": "France – Ligue 2",
    "N1": "Netherlands – Eredivisie",
    "B1": "Belgium – Pro League",
    "P1": "Portugal – Primeira Liga",
    "T1": "Turkey – Süper Lig",
    "G1": "Greece – Super League",
}

# Ten strongest European leagues available in the data (roughly by UEFA coefficient)
TOP10 = ["E0", "SP1", "I1", "D1", "F1", "P1", "N1", "B1", "T1", "SC0"]

# openfootball/football.json file names for leagues that publish a full schedule
OPENFOOTBALL = {
    "E0": "en.1", "E1": "en.2", "SP1": "es.1", "I1": "it.1",
    "D1": "de.1", "F1": "fr.1", "N1": "nl.1", "P1": "pt.1",
}
OPENFOOTBALL_URL = "https://raw.githubusercontent.com/openfootball/football.json/master"

UK = ZoneInfo("Europe/London")  # football-data.co.uk times are UK time
LOCAL_TZ = {  # openfootball times are local kick-off times
    "E": "Europe/London", "SC": "Europe/London", "P": "Europe/Lisbon",
    "T": "Europe/Istanbul", "G": "Europe/Athens",
}

# Market-average odds columns (present in recent seasons and in fixtures.csv)
ODDS_COLS = ["AvgH", "AvgD", "AvgA", "Avg>2.5", "Avg<2.5"]
BASE_COLS = ["Div", "Date", "Time", "HomeTeam", "AwayTeam"]
RESULT_COLS = ["FTHG", "FTAG", "FTR", "HxG", "AxG"]


def league_tz(div: str) -> ZoneInfo:
    for prefix, tz in LOCAL_TZ.items():
        if div.startswith(prefix):
            return ZoneInfo(tz)
    return ZoneInfo("Europe/Paris")


def season_start_year(today: dt.date | None = None) -> int:
    today = today or dt.date.today()
    return today.year if today.month >= 7 else today.year - 1


def season_code(start_year: int) -> str:
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


_SHORT_CACHE: dict[str, tuple[float, pd.DataFrame | None]] = {}


def _get_csv(url: str, cache_seconds: int = 0) -> pd.DataFrame | None:
    if cache_seconds and url in _SHORT_CACHE and time.time() - _SHORT_CACHE[url][0] < cache_seconds:
        df = _SHORT_CACHE[url][1]
        return None if df is None else df.copy()
    df = _download_csv(url)
    if cache_seconds:
        _SHORT_CACHE[url] = (time.time(), df)
    return None if df is None else df.copy()


def _download_csv(url: str) -> pd.DataFrame | None:
    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    try:
        text = r.content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = r.content.decode("latin-1")
    df = pd.read_csv(io.StringIO(text), on_bad_lines="skip", low_memory=False)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df.dropna(axis=1, how="all")


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")
    return df.dropna(subset=["Date"])


def _keep(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[[c for c in cols if c in df.columns]].copy()


def load_results(div: str, n_seasons: int = 3, today: dt.date | None = None) -> pd.DataFrame:
    """Finished matches for `div` over the last `n_seasons` seasons (incl. current)."""
    start = season_start_year(today)
    years = list(range(start - n_seasons + 1, start + 1))
    with ThreadPoolExecutor(max_workers=len(years)) as pool:  # seasons download in parallel
        downloads = pool.map(lambda y: _get_csv(f"{BASE}/mmz4281/{season_code(y)}/{div}.csv"), years)
    frames = []
    for y, df in zip(years, downloads):
        if df is None or df.empty:
            continue
        df = _keep(df, BASE_COLS + RESULT_COLS + ODDS_COLS)
        df["Season"] = season_code(y)
        frames.append(df)
    if not frames:
        raise ValueError(f"No results found for {div}")
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df[["FTHG", "FTAG"]] = df[["FTHG", "FTAG"]].astype(int)
    df["HomeTeam"] = df["HomeTeam"].str.strip()
    df["AwayTeam"] = df["AwayTeam"].str.strip()
    return _parse_dates(df).sort_values("Date").reset_index(drop=True)


def kickoffs(df: pd.DataFrame, tz: ZoneInfo) -> pd.Series:
    """Timezone-aware kick-off times. A missing time counts as end of day (keeps the match)."""
    time = df["Time"].fillna("23:59") if "Time" in df else "23:59"
    stamp = pd.to_datetime(df["Date"].dt.strftime("%Y-%m-%d") + " " + time, errors="coerce")
    stamp = stamp.fillna(df["Date"] + pd.Timedelta(hours=23, minutes=59))
    return stamp.dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")


def load_fixtures(div: str | None = None, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Fixtures from football-data.co.uk that have NOT kicked off yet, with market odds."""
    df = _get_csv(f"{BASE}/fixtures.csv", cache_seconds=600)  # same file for every league
    if df is None or df.empty:
        return pd.DataFrame(columns=BASE_COLS + ODDS_COLS + ["Kickoff"])
    df = _parse_dates(_keep(df, BASE_COLS + ODDS_COLS))
    df["HomeTeam"] = df["HomeTeam"].str.strip()
    df["AwayTeam"] = df["AwayTeam"].str.strip()
    if div:
        df = df[df["Div"] == div].copy()
    df["Kickoff"] = kickoffs(df, UK)
    now = now or pd.Timestamp.now(tz=UK)
    df = df[df["Kickoff"] > now]
    return df.sort_values("Kickoff").reset_index(drop=True)


_SCHEDULE_CACHE: dict[str, tuple[float, list]] = {}


def load_schedule(div: str, today: dt.date | None = None) -> pd.DataFrame:
    """Full current-season schedule from openfootball (raw team names, scores if played)."""
    cols = ["Div", "Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    code = OPENFOOTBALL.get(div)
    if not code:
        return pd.DataFrame(columns=cols)
    start = season_start_year(today)
    url = f"{OPENFOOTBALL_URL}/{start}-{(start + 1) % 100:02d}/{code}.json"
    hit = _SCHEDULE_CACHE.get(url)
    if hit and time.time() - hit[0] < 6 * 3600:  # schedules rarely change; GitHub rate-limits
        matches = hit[1]
    else:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 404:
            return pd.DataFrame(columns=cols)
        r.raise_for_status()
        matches = r.json().get("matches", [])
        _SCHEDULE_CACHE[url] = (time.time(), matches)
    rows = []
    for m in matches:
        score = m.get("score") or {}
        ft = score.get("ft") if isinstance(score, dict) else score  # both formats exist
        if not ft or len(ft) < 2:
            ft = [m.get("score1"), m.get("score2")]
        rows.append({
            "Div": div, "Date": m.get("date"), "Time": m.get("time"),
            "HomeTeam": m.get("team1"), "AwayTeam": m.get("team2"),
            "FTHG": ft[0], "FTAG": ft[1],
        })
    df = pd.DataFrame(rows, columns=cols)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)


_NOISE = {
    "fc", "afc", "cf", "sc", "ac", "as", "ss", "us", "rc", "rcd", "cd", "ud", "sd", "sv", "vfb",
    "vfl", "tsg", "bv", "ogc", "fk", "sk", "club", "de", "da", "calcio", "sad", "rsc", "krc",
    "kv", "kaa", "royal", "cp", "and", "hove", "the", "1",
}


def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    words = [w for w in re.split(r"[^a-z0-9]+", s) if w and w not in _NOISE and not w.isdigit()]
    return " ".join(words)


def map_team_names(schedule: pd.DataFrame, results: pd.DataFrame) -> dict[str, str]:
    """Map openfootball team names -> football-data team names.

    1) Played matches vote: same date (±1 day) and same score => the names likely correspond.
    2) Anything left is matched by fuzzy string similarity.
    """
    known = sorted(set(results["HomeTeam"]) | set(results["AwayTeam"]))
    votes: dict[tuple[str, str], int] = {}
    played = schedule.dropna(subset=["FTHG", "FTAG"])
    recent = results[results["Date"] >= schedule["Date"].min() - pd.Timedelta(days=1)]
    for m in played.itertuples():
        cand = recent[
            ((recent["Date"] - m.Date).abs() <= pd.Timedelta(days=1))
            & (recent["FTHG"] == m.FTHG) & (recent["FTAG"] == m.FTAG)
        ]
        for c in cand.itertuples():
            for pair in ((m.HomeTeam, c.HomeTeam), (m.AwayTeam, c.AwayTeam)):
                votes[pair] = votes.get(pair, 0) + 1

    mapping: dict[str, str] = {}
    used: set[str] = set()
    for (raw, fd), n in sorted(votes.items(), key=lambda kv: -kv[1]):  # strongest first, 1-to-1
        if n >= 2 and raw not in mapping and fd not in used:
            mapping[raw] = fd
            used.add(fd)

    names = set(schedule["HomeTeam"]) | set(schedule["AwayTeam"])
    by_norm = {_norm(k): k for k in known if k not in used}
    for raw in sorted(names - set(mapping)):
        key = _norm(raw)
        hit = difflib.get_close_matches(key, list(by_norm), n=1, cutoff=0.6)
        if not hit:  # containment, e.g. "manchester united" vs "man united" fails, "psv" works
            hit = [k for k in by_norm if k and (k in key or key in k)][:1]
        if hit:
            mapping[raw] = by_norm.pop(hit[0])
    return mapping


def load_upcoming(
    div: str, results: pd.DataFrame, now: pd.Timestamp | None = None
) -> tuple[pd.DataFrame, list[str], str | None]:
    """Not-yet-played fixtures for `div`: football-data (with odds) first, schedule as fallback.

    Returns (fixtures, schedule team names not matched to the results data, schedule error).
    A failed schedule download is not fatal: the odds feed's fixtures are still returned.
    """
    now = now or pd.Timestamp.now(tz=UK)
    fx = load_fixtures(div, now)
    fx["Source"] = "football-data"

    unmatched: list[str] = []
    error = None
    try:
        sched = load_schedule(div, now.date())
    except (requests.RequestException, ValueError) as e:
        sched, error = pd.DataFrame(), f"season schedule unavailable ({e})"
    if not sched.empty:
        mapping = map_team_names(sched, results)
        sched = sched[sched["FTHG"].isna()].drop(columns=["FTHG", "FTAG"])
        sched["Kickoff"] = kickoffs(sched, league_tz(div))
        sched = sched[sched["Kickoff"] > now]
        unmatched = sorted((set(sched["HomeTeam"]) | set(sched["AwayTeam"])) - set(mapping))
        sched["HomeTeam"] = sched["HomeTeam"].map(mapping)
        sched["AwayTeam"] = sched["AwayTeam"].map(mapping)
        sched = sched.dropna(subset=["HomeTeam", "AwayTeam"])
        sched["Source"] = "openfootball"
        have = set(zip(fx["HomeTeam"], fx["AwayTeam"]))  # odds feed wins when both have it
        sched = sched[[(h, a) not in have for h, a in zip(sched["HomeTeam"], sched["AwayTeam"])]]
        if not sched.empty:
            fx = pd.concat([fx, sched], ignore_index=True) if not fx.empty else sched

    if fx.empty:
        return fx, unmatched, error
    fx["Kickoff"] = pd.to_datetime(fx["Kickoff"], utc=True).dt.tz_convert(UK)
    return fx.sort_values("Kickoff").reset_index(drop=True), unmatched, error
