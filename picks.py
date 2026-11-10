"""Pick selection shared by the Most likely, Daily pick and chat features.

Everything works on a "candidates" table: one row per (upcoming match, market) with the model's
probability and the odds you could take. Odds are the bookmaker market average when football-data
has published them, otherwise the model's fair odds (1 / probability), flagged in `OddsSource`.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

import competitions
import data
import model

CANDIDATE_COLS = [
    "Kickoff", "Day", "Div", "League", "Home", "Away", "Match", "Market", "Pick",
    "Prob", "FairOdds", "MarketOdds", "Odds", "OddsSource", "Edge", "xG", "LikelyScore", "HasTime",
]


NAMES = {**data.LEAGUES, **competitions.COMPETITIONS}


def build_candidates(fixtures: list[tuple], min_matches: int = 0) -> pd.DataFrame:
    """fixtures: [(competition code, fitted model, upcoming fixtures[, result_fn])] -> one row
    per (match, market). `result_fn(fixture) -> (home, draw, away) | None` overrides the
    result probabilities (the deep model for domestic leagues); goal markets keep the
    Dixon-Coles model. Matches with a team the model has seen fewer than `min_matches` times
    are skipped: their probabilities are too uncertain to rank against well-known teams."""
    rows = []
    for item in fixtures:
        div, mdl, fx = item[:3]
        result_fn = item[3] if len(item) > 3 else None
        for f in fx.to_dict("records"):  # not itertuples: it renames "Avg>2.5"
            home, away = f["HomeTeam"], f["AwayTeam"]
            if not (mdl.knows(home) and mdl.knows(away)):
                continue
            if min(mdl.n_matches.get(home, 0), mdl.n_matches.get(away, 0)) < min_matches:
                continue
            p = mdl.predict(home, away)
            res = result_fn(f) if result_fn else None
            if res is not None:
                p = {**p, "home": float(res[0]), "draw": float(res[1]), "away": float(res[2])}
            for market, (keys, col) in model.MARKETS.items():
                prob = float(sum(p[k] for k in keys))
                mo = f.get(col) if col else None
                mo = float(mo) if mo is not None and pd.notna(mo) and float(mo) > 1 else np.nan
                rows.append({
                    "Kickoff": f["Kickoff"], "Day": f["Kickoff"].date(), "Div": div,
                    "League": NAMES.get(div, div), "Home": home, "Away": away,
                    "Match": f"{home} vs {away}", "Market": market,
                    "Pick": market.replace("Home", home).replace("Away", away),
                    "Prob": prob, "FairOdds": 1 / prob if prob > 0 else np.inf,
                    "MarketOdds": mo,
                    "Odds": mo if np.isfinite(mo) else (1 / prob if prob > 0 else np.inf),
                    "OddsSource": "market" if np.isfinite(mo) else "model",
                    "Edge": prob * mo - 1 if np.isfinite(mo) else np.nan,
                    "xG": f"{p['xG_home']:.2f} – {p['xG_away']:.2f}",
                    "LikelyScore": p["top_scores"][0][0],
                    "HasTime": pd.notna(f.get("Time")),
                })
    return pd.DataFrame(rows, columns=CANDIDATE_COLS)


def best_per_match(cands: pd.DataFrame, markets: list[str]) -> pd.DataFrame:
    """The single most probable outcome per match among `markets`."""
    c = cands[cands["Market"].isin(markets)]
    if c.empty:
        return c
    return c.loc[c.groupby("Match")["Prob"].idxmax()].sort_values("Prob", ascending=False)


def near_odds(cands: pd.DataFrame, target: float, tol: float | None = None) -> pd.DataFrame:
    """Selections priced close to `target` odds, best first.

    With market odds, "best" = highest model edge (model thinks it's underpriced).
    With model-only odds every pick at the same price has the same probability, so we just
    rank by closeness to the target.
    """
    tol = tol if tol is not None else max(0.05, 0.06 * target)
    c = cands[(cands["Odds"] - target).abs() <= tol].copy()
    if c.empty:
        return c
    c["Distance"] = (c["Odds"] - target).abs()
    c["_edge"] = c["Edge"].fillna(-np.inf)
    return c.sort_values(["_edge", "Distance", "Prob"], ascending=[False, True, False]).drop(columns="_edge")


def best_accumulator(
    cands: pd.DataFrame,
    min_odds: float = 8.0,
    max_odds: float = 15.0,
    max_legs: int = 6,
    min_leg_prob: float = 0.0,
    exclude_matches: set[str] | None = None,
) -> pd.DataFrame | None:
    """Most probable ticket (one leg per match) whose combined odds land in [min_odds, max_odds].

    Dynamic programming over (number of legs, combined log-odds bucket); keeps the highest
    log-probability per state. Ties (common with model-only odds, where probability is exactly
    1/odds) go to fewer legs.
    """
    exclude_matches = exclude_matches or set()
    c = cands[(cands["Prob"] >= min_leg_prob) & (cands["Odds"] > 1.01)
              & ~cands["Match"].isin(exclude_matches)]
    if c.empty:
        return None
    step = 0.005
    hi = math.log(max_odds)
    # state: (legs, bucket) -> (score, log_odds, [row indices])
    states: dict[tuple[int, int], tuple[float, float, list]] = {(0, 0): (0.0, 0.0, [])}
    for _, grp in c.groupby("Match"):
        options = list(zip(grp.index, np.log(grp["Prob"]), np.log(grp["Odds"])))
        new = dict(states)
        for (legs, _), (score, lo, idx) in states.items():
            if legs >= max_legs:
                continue
            for i, lp, lodd in options:
                tot = lo + lodd
                if tot > hi + 1e-9:
                    continue
                key = (legs + 1, int(round(tot / step)))
                cand = (score + lp, tot, idx + [i])
                if key not in new or cand[0] > new[key][0]:
                    new[key] = cand
        states = new

    lo_lim = math.log(min_odds)
    best = None
    for (legs, _), (score, lo, idx) in states.items():
        if legs and lo_lim - 1e-9 <= lo <= hi + 1e-9:
            key = (score, -legs)
            if best is None or key > best[0]:
                best = (key, idx)
    if best is None:
        return None
    return c.loc[best[1]].sort_values("Kickoff")


def accumulators(cands: pd.DataFrame, n: int = 3, **kw) -> list[pd.DataFrame]:
    """Up to `n` tickets that share no matches."""
    out, used = [], set()
    for _ in range(n):
        t = best_accumulator(cands, exclude_matches=used, **kw)
        if t is None:
            break
        out.append(t)
        used |= set(t["Match"])
    return out


def ticket_summary(t: pd.DataFrame) -> dict:
    odds = float(np.prod(t["Odds"]))
    prob = float(np.prod(t["Prob"]))
    all_market = bool((t["OddsSource"] == "market").all())
    return {
        "legs": len(t), "combined_odds": odds, "model_probability": prob,
        "expected_return_per_unit": prob * odds - 1, "odds_source": "market" if all_market
        else "model" if (t["OddsSource"] == "model").all() else "mixed",
    }


# ---------------- rule-based chat ----------------
LEAGUE_WORDS = {
    "premier": "E0", "epl": "E0", "england": "E0", "championship": "E1",
    "la liga": "SP1", "laliga": "SP1", "spain": "SP1", "serie a": "I1", "italy": "I1",
    "bundesliga": "D1", "germany": "D1", "ligue 1": "F1", "france": "F1",
    "eredivisie": "N1", "netherlands": "N1", "dutch": "N1", "primeira": "P1", "portugal": "P1",
    "pro league": "B1", "belgium": "B1", "super lig": "T1", "süper lig": "T1", "turkey": "T1",
    "scotland": "SC0", "scottish": "SC0",
    "champions league": "UCL", "ucl": "UCL", "europa": "UEL", "uel": "UEL",
    "conference": "UECL", "uecl": "UECL", "nations league": "UNL", "international": "UNL",
    "national team": "UNL",
}
MARKET_WORDS = [
    (r"both teams|btts|gg\b", ["Both teams score"]),
    (r"over\s*1\.?5", ["Over 1.5 goals"]),
    (r"over\s*2\.?5|over goals|\bover\b", ["Over 2.5 goals"]),
    (r"under\s*3\.?5", ["Under 3.5 goals"]),
    (r"under\s*2\.?5|\bunder\b", ["Under 2.5 goals"]),
    (r"double chance|or draw|1x|x2", ["Home or draw", "Away or draw"]),
    (r"\bdraws?\b", ["Draw"]),
    (r"away win|away team", ["Away win"]),
    (r"home win|home team", ["Home win"]),
    (r"\bwins?\b|winner|1x2|result", ["Home win", "Away win"]),
]
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

HELP = (
    "I can answer things like:\n"
    "- *best prediction at odds 1.7* or *best over 2.5 around @2.0*\n"
    "- *build an acca with odds 8 to 15 for Saturday*\n"
    "- *safest picks in the Premier League this weekend*\n"
    "- *what about Arsenal?* (next match prediction)"
)


def _fmt_odds(row) -> str:
    tag = "" if row["OddsSource"] == "market" else " (model)"
    return f"{row['Odds']:.2f}{tag}"


def _when(row) -> str:
    return row["Kickoff"].strftime("%a %d %b %H:%M" if row["HasTime"] else "%a %d %b")


def parse_question(q: str, cands: pd.DataFrame, today) -> dict:
    """Pull odds, ranges, markets, leagues, days and teams out of a free-text question."""
    s = q.lower().replace(",", ".")
    out: dict = {}
    rng = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|to|and|–|~)\s*(\d+(?:\.\d+)?)", s)
    if rng and float(rng.group(2)) > float(rng.group(1)) > 1:
        out["range"] = (float(rng.group(1)), float(rng.group(2)))
    else:
        m = re.search(r"@\s*(\d+(?:\.\d+)?)", s) or re.search(
            r"odds?\s*(?:of|around|near|at|about|=)?\s*@?\s*(\d+(?:\.\d+)?)", s
        ) or re.search(r"\b(\d+\.\d+)\b", s)
        if m and 1.01 <= float(m.group(1)) <= 500:
            out["odds"] = float(m.group(1))
    out["acca"] = bool(re.search(r"acca|accumulator|combo|ticket|parlay|multiple|slip|daily", s)) \
        or ("range" in out and out["range"][0] >= 3)
    out["safe"] = bool(re.search(r"safe|sure|likely|banker|certain|best bet|top pick", s))
    for pat, mk in MARKET_WORDS:
        if re.search(pat, s):
            out["markets"] = mk
            break
    out["leagues"] = sorted({div for w, div in LEAGUE_WORDS.items() if w in s})
    days = set(cands["Day"]) if not cands.empty else set()
    if "today" in s or "tonight" in s:
        out["day"] = [today]
    elif "tomorrow" in s:
        out["day"] = [today + pd.Timedelta(days=1)]
    elif "weekend" in s:  # the next weekend only: Friday to Sunday
        wk = sorted(d for d in days if d.weekday() >= 5)
        if wk:
            saturday = wk[0] - pd.Timedelta(days=wk[0].weekday() - 5)
            out["day"] = [d for d in days if -1 <= (d - saturday).days <= 1]
        else:
            out["day"] = []
    else:
        for i, w in enumerate(WEEKDAYS):
            if w in s or re.search(rf"\b{w[:3]}\b", s):
                out["day"] = sorted(d for d in days if d.weekday() == i)[:1]
    teams = set(cands["Home"]) | set(cands["Away"]) if not cands.empty else set()
    out["teams"] = [t for t in teams if re.search(rf"\b{re.escape(t.lower())}\b", s)]
    return out


def answer(q: str, cands: pd.DataFrame, today) -> str:
    """Rule-based answer used when no Claude API key is configured."""
    if cands.empty:
        return "There are no upcoming fixtures loaded right now, so I have nothing to pick from."
    p = parse_question(q, cands, today)
    c = cands
    if p.get("leagues") and not p["teams"]:  # "Spain" may be a national team, not La Liga
        c = c[c["Div"].isin(p["leagues"])]
    if "day" in p:
        if not p["day"]:
            return "There are no fixtures loaded for that day. " + _days_hint(cands)
        c = c[c["Day"].isin(p["day"])]
    if c.empty:
        return "No fixtures match that. " + _days_hint(cands)

    if p["teams"]:
        lines = []
        for team in p["teams"][:3]:
            m = c[(c["Home"] == team) | (c["Away"] == team)]
            if m.empty:
                continue
            first = m[m["Kickoff"] == m["Kickoff"].min()]
            row = first.iloc[0]
            get = lambda name: first[first["Market"] == name].iloc[0]
            lines.append(
                f"**{row['Match']}** ({row['League']}, {_when(row)})\n"
                f"- {row['Home']} {get('Home win')['Prob']:.0%} · draw {get('Draw')['Prob']:.0%} · "
                f"{row['Away']} {get('Away win')['Prob']:.0%}\n"
                f"- Over 2.5: {get('Over 2.5 goals')['Prob']:.0%} · BTTS: {get('Both teams score')['Prob']:.0%}"
                f" · xG {row['xG']} · likely score {row['LikelyScore']}"
            )
        if lines:
            return "\n\n".join(lines)

    markets = p.get("markets") or list(model.MARKETS)
    c = c[c["Market"].isin(markets)]

    if p["acca"]:
        lo, hi = p.get("range", (8.0, 15.0))
        t = best_accumulator(c, lo, hi, max_legs=8, min_leg_prob=0.55)
        if t is None:
            return f"I couldn't build a ticket with combined odds between {lo:g} and {hi:g} from those fixtures."
        s = ticket_summary(t)
        legs = "\n".join(f"- {r['Pick']}: {r['Match']} ({_when(r)}) @ {_fmt_odds(r)}, model {r['Prob']:.0%}"
                         for _, r in t.iterrows())
        return (
            f"**{s['legs']}-leg ticket @ {s['combined_odds']:.2f}** "
            f"(model chance all win: {s['model_probability']:.1%})\n{legs}\n\n"
            + _odds_note(t)
        )

    if "odds" in p:
        target = p["odds"]
        hits = near_odds(c, target).drop_duplicates("Match").head(5)
        if hits.empty:
            return f"Nothing is priced near {target:g} for that selection. " + _days_hint(cands)
        lines = []
        for _, r in hits.iterrows():
            extra = f", edge {r['Edge']:+.0%}" if r["OddsSource"] == "market" else ""
            lines.append(f"- **{r['Pick']}**: {r['Match']} ({r['League']}, {_when(r)}) "
                         f"@ {_fmt_odds(r)}, model {r['Prob']:.0%}{extra}")
        head = f"Best selections around **@{target:g}**:"
        return head + "\n" + "\n".join(lines) + "\n\n" + _odds_note(hits)

    best = best_per_match(c, markets).head(5)
    lines = [f"- **{r['Pick']}**: {r['Match']} ({r['League']}, {_when(r)}), model {r['Prob']:.0%} "
             f"@ {_fmt_odds(r)}" for _, r in best.iterrows()]
    head = "Most likely outcomes:" if p["safe"] or p.get("markets") or p.get("leagues") else HELP + "\n\nMeanwhile, the most likely outcomes:"
    return head + "\n" + "\n".join(lines)


def _odds_note(df: pd.DataFrame) -> str:
    if (df["OddsSource"] == "model").any():
        return ("_Odds marked (model) are the model's fair odds: bookmaker prices for these "
                "fixtures aren't published yet, so there is no value edge to measure._")
    return "_Edge = model probability × market odds − 1._"


def _days_hint(cands: pd.DataFrame) -> str:
    days = sorted(set(cands["Day"]))[:5]
    return "Fixtures are loaded for: " + ", ".join(d.strftime("%a %d %b") for d in days) + "."
