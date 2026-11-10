"""Premier League team news from the official Fantasy Premier League API (free, no key).

For every player it gives availability (injured / doubtful / suspended), the chance of
playing next round and the player's expected goal involvements (xG + xA) this season.
We turn that into "share of the team's attacking output that is likely missing".

Other leagues have no free, reliable injury feed, so team news is Premier League only.
"""
from __future__ import annotations

import time

import pandas as pd
import requests

import data

FPL_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
# FPL team name -> football-data name (where they differ)
FPL_TO_FD = {
    "Man Utd": "Man United", "Nott'm Forest": "Nott'm Forest", "Spurs": "Tottenham",
    "Wolves": "Wolves", "Newcastle": "Newcastle", "Sheffield Utd": "Sheffield United",
    "Leeds": "Leeds", "Ipswich": "Ipswich", "Hull": "Hull", "Coventry": "Coventry",
    "Coventry City": "Coventry", "Hull City": "Hull", "Ipswich Town": "Ipswich",
    "Leicester City": "Leicester", "Norwich City": "Norwich", "Luton Town": "Luton",
}
STATUS = {"i": "Injured", "d": "Doubtful", "s": "Suspended", "u": "Unavailable", "n": "Not in squad"}
# how strongly missing attacking output lowers a team's expected goals. Not backtestable
# (no free injury history), so it is deliberately cautious.
IMPACT = 0.5

_cache: tuple[float, dict] | None = None


def _bootstrap() -> dict:
    global _cache
    if _cache and time.time() - _cache[0] < 1800:
        return _cache[1]
    r = requests.get(FPL_URL, headers=data.HEADERS, timeout=30)
    r.raise_for_status()
    _cache = (time.time(), r.json())
    return _cache[1]


def team_news() -> dict[str, dict]:
    """football-data team name -> {"missing_attack": share 0..1, "players": DataFrame}."""
    j = _bootstrap()
    teams = {t["id"]: FPL_TO_FD.get(t["name"], t["name"]) for t in j["teams"]}
    players = pd.DataFrame(j["elements"])
    players["team_name"] = players["team"].map(teams)
    players["xgi"] = pd.to_numeric(players["expected_goal_involvements"], errors="coerce").fillna(0)
    players["chance"] = pd.to_numeric(players["chance_of_playing_next_round"], errors="coerce")
    players.loc[players["status"] == "a", "chance"] = 100
    players["chance"] = players["chance"].fillna(100)
    out = {}
    for team, g in players.groupby("team_name"):
        total_xgi = g["xgi"].sum() or 1.0
        total_min = g["minutes"].sum() or 1.0
        absent = g[(g["status"] != "a") & (g["minutes"] > 0)].copy()
        absent["miss"] = 1 - absent["chance"] / 100
        absent["xgi_share"] = absent["xgi"] / total_xgi
        absent["minutes_share"] = absent["minutes"] / total_min
        absent = absent[absent["miss"] > 0].sort_values(["miss", "minutes"], ascending=False)
        out[team] = {
            "missing_attack": float((absent["xgi_share"] * absent["miss"]).sum()),
            "missing_minutes": float((absent["minutes_share"] * absent["miss"]).sum()),
            "players": pd.DataFrame({
                "Player": absent["web_name"], "Status": absent["status"].map(STATUS),
                "Chance to play": absent["chance"].map(lambda c: f"{c:.0f}%"),
                "Share of team xG+xA": absent["xgi_share"].map(lambda s: f"{s:.0%}"),
                "Share of team minutes": absent["minutes_share"].map(lambda s: f"{s:.0%}"),
                "News": absent["news"],
            }).reset_index(drop=True),
        }
    return out


def xg_multiplier(news: dict | None) -> float:
    """Factor applied to a team's expected goals for its missing attackers."""
    if not news:
        return 1.0
    return max(0.6, 1 - IMPACT * news["missing_attack"])
