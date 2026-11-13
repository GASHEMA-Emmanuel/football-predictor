"""Past daily picks, rebuilt honestly and graded against real results.

For every earlier date we rebuild what the Daily pick would have been, using only what was
known before that day:
- domestic leagues: goal markets from a Dixon-Coles model fitted on matches before the date,
  result markets from the deep model's pre-kick-off signals (classifier trained on earlier
  seasons only, see DeepEngine._past_probs);
- European cups and the Nations League: their models refitted on matches before the date.
Then each leg is settled with the final score.

Nothing has to be stored, so the track record also works on hosts without a disk.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

import competitions
import data
import model
import picks

JOINT_L2 = 1.0


def leg_won(market: str, hg: float, ag: float) -> bool | None:
    """Did a selection win, given the final score? None if the match isn't settled."""
    if pd.isna(hg) or pd.isna(ag):
        return None
    total = hg + ag
    return {
        "Home win": hg > ag, "Draw": hg == ag, "Away win": ag > hg,
        "Home or draw": hg >= ag, "Away or draw": ag >= hg,
        "Over 1.5 goals": total >= 2, "Over 2.5 goals": total >= 3,
        "Under 2.5 goals": total <= 2, "Under 3.5 goals": total <= 3,
        "Both teams score": hg > 0 and ag > 0,
    }[market]


def _with_scores(cands: pd.DataFrame, scores: dict) -> pd.DataFrame:
    if cands.empty:
        return cands
    s = [scores.get((d, h, a), (np.nan, np.nan)) for d, h, a in zip(cands["Div"], cands["Home"], cands["Away"])]
    return cands.assign(FTHG=[x[0] for x in s], FTAG=[x[1] for x in s])


def past_pools(
    days_back: int,
    domestic: dict[str, pd.DataFrame],
    deep_past: dict,
    half_life: int,
    today: dt.date,
    min_matches: int = 10,
    competitions_set: list[str] | None = None,
) -> dict[dt.date, pd.DataFrame]:
    """date -> candidate selections for that day (with final scores), for the last `days_back`
    days before `today`. `domestic` holds every league's results (all 18 are needed for the
    joint European model)."""
    comps = competitions_set or competitions.DEFAULT_PICK_SET
    dates = [today - dt.timedelta(days=i) for i in range(days_back, 0, -1)]
    first = pd.Timestamp(dates[0])
    pools: dict[dt.date, list] = {d: [] for d in dates}
    scores: dict[dt.date, dict] = {d: {} for d in dates}

    # ---- domestic leagues
    for div in [c for c in comps if c in data.LEAGUES]:
        res = domestic.get(div)
        if res is None or res.empty:
            continue
        recent = res[res["Date"] >= first].copy()
        if recent.empty:
            continue
        recent["Kickoff"] = data.kickoffs(recent, data.UK)
        for d, day_rows in recent.groupby(recent["Kickoff"].dt.date):
            if d not in pools:
                continue
            mdl = model.DixonColes.fit(res[res["Date"] < pd.Timestamp(d)], half_life_days=half_life)
            fn = lambda f, div=div: deep_past.get((div, f["Date"], f["HomeTeam"], f["AwayTeam"]))
            pools[d].append((div, mdl, day_rows, fn))
            for r in day_rows.itertuples():
                scores[d][(div, r.HomeTeam, r.AwayTeam)] = (r.FTHG, r.FTAG)

    # ---- European cups (one joint model, refitted per date)
    cups = [c for c in comps if c in competitions.CUPS]
    if cups:
        joint, mapping = competitions.europe_matches(list(domestic.values()), seasons=2)
        for code in cups:
            _, slug = competitions.CUPS[code]
            fx = competitions.load_fixturedownload(slug, data.season_start_year(today))
            fx = fx[fx["Kickoff"] >= first.tz_localize(data.UK)].copy()
            fx["HomeTeam"] = fx["HomeTeam"].replace(mapping)
            fx["AwayTeam"] = fx["AwayTeam"].replace(mapping)
            for d, day_rows in fx.groupby(fx["Kickoff"].dt.date):
                if d not in pools:
                    continue
                mdl = model.DixonColes.fit(joint[joint["Date"] < pd.Timestamp(d)],
                                           half_life_days=half_life, l2=JOINT_L2)
                pools[d].append((code, mdl, day_rows))
                for r in day_rows.itertuples():
                    scores[d][(code, r.HomeTeam, r.AwayTeam)] = (r.FTHG, r.FTAG)

    # ---- international (Nations League)
    intl_codes = [c for c in comps if c in competitions.INTERNATIONAL]
    if intl_codes:
        intl = competitions.international_results(8)
        known = set(intl["HomeTeam"]) | set(intl["AwayTeam"])
        for code in intl_codes:
            _, slug = competitions.INTERNATIONAL[code]
            fx = competitions.load_fixturedownload(slug, data.season_start_year(today))
            fx = fx[fx["Kickoff"] >= first.tz_localize(data.UK)].copy()
            fx["HomeTeam"] = fx["HomeTeam"].map(lambda t: competitions.nation_name(t, known))
            fx["AwayTeam"] = fx["AwayTeam"].map(lambda t: competitions.nation_name(t, known))
            for d, day_rows in fx.groupby(fx["Kickoff"].dt.date):
                if d not in pools:
                    continue
                mdl = model.DixonColes.fit(intl[intl["Date"] < pd.Timestamp(d)],
                                           half_life_days=730, l2=JOINT_L2)
                pools[d].append((code, mdl, day_rows))
                for r in day_rows.itertuples():
                    scores[d][(code, r.HomeTeam, r.AwayTeam)] = (r.FTHG, r.FTAG)

    out = {}
    for d in dates:
        if pools[d]:
            cands = picks.build_candidates(pools[d], min_matches=min_matches)
            if not cands.empty:
                out[d] = _with_scores(cands, scores[d])
    return out


def grade(ticket: pd.DataFrame) -> pd.DataFrame:
    """Add a per-leg result to a ticket."""
    won = [leg_won(m, h, a) for m, h, a in zip(ticket["Market"], ticket["FTHG"], ticket["FTAG"])]
    return ticket.assign(
        Won=won,
        Score=[f"{int(h)}-{int(a)}" if pd.notna(h) and pd.notna(a) else "—"
               for h, a in zip(ticket["FTHG"], ticket["FTAG"])],
    )


def ticket_status(graded: pd.DataFrame) -> str:
    if (graded["Won"] == False).any():  # noqa: E712 (None means pending)
        return "Lost"
    if graded["Won"].isna().any() or (graded["Won"].map(lambda w: w is None)).any():
        return "Pending"
    return "Won"


def track_record(pools: dict, **ticket_kw) -> list[dict]:
    """Best ticket per past day (same rules as today's Daily pick), graded."""
    out = []
    for d in sorted(pools, reverse=True):
        t = picks.best_accumulator(pools[d], **ticket_kw)
        if t is None:
            continue
        g = grade(t)
        s = picks.ticket_summary(t)
        out.append({"date": d, "ticket": g, "status": ticket_status(g), **s,
                    "legs_won": int((g["Won"] == True).sum())})  # noqa: E712
    return out
