"""Claude-powered chat over the predictor's own data (used when an Anthropic API key is set).

Claude never invents numbers: it calls the tools below, which read the same candidates table
the app shows (model probabilities + bookmaker or model-fair odds).
"""
from __future__ import annotations

import json
import os

import anthropic
import pandas as pd
from anthropic import beta_tool

import model
import picks

MODEL = "claude-opus-5-5"

SYSTEM = """You are the assistant inside a football prediction app. Answer questions about \
upcoming matches in Europe's top leagues, the Champions League, Europa League, Conference League \
and UEFA Nations League using ONLY the tools; never make up fixtures, odds or probabilities. \
Other international matches (qualifiers, friendlies) have no fixture feed: say the user can \
predict any two national teams on the app's International page.

How the data works:
- Probabilities come from a Dixon-Coles statistical model fitted on recent results.
- Each selection has odds. odds_source "market" = bookmaker average odds; "model" = the model's \
fair odds (1/probability), used when bookmakers haven't published the round yet. With model odds \
there is no value edge to speak of, so say so instead of calling something "value".
- edge = model probability x market odds - 1 (only for market odds).

When asked for "the best prediction at odds X", search near X and prefer the highest edge when \
market odds exist; otherwise explain that the picks are priced by the model. For accumulators, \
use build_accumulator and always report the combined odds and the model's chance that every leg \
wins. Be concise: short bullet lists with match, kick-off (UK time), pick, odds, probability. \
Never promise wins; mention that bets can lose when users talk about staking. Today is {today}. \
Fixtures are loaded for these days: {days}."""


def api_key() -> str | None:
    try:
        import streamlit as st
        key = st.secrets.get("ANTHROPIC_API_KEY")
        if key:
            return key
    except Exception:  # no secrets.toml
        pass
    return os.environ.get("ANTHROPIC_API_KEY")


def _rows(df: pd.DataFrame, limit: int) -> list[dict]:
    out = []
    for _, r in df.head(limit).iterrows():
        out.append({
            "match": r["Match"], "league": r["League"],
            "kickoff_uk": r["Kickoff"].strftime("%a %d %b %H:%M" if r["HasTime"] else "%a %d %b (time TBC)"),
            "pick": r["Pick"], "market": r["Market"],
            "probability": round(r["Prob"], 3), "odds": round(r["Odds"], 2),
            "odds_source": r["OddsSource"],
            "edge": None if pd.isna(r["Edge"]) else round(r["Edge"], 3),
            "xg": r["xG"], "likely_score": r["LikelyScore"],
        })
    return out


def make_tools(cands: pd.DataFrame):
    def _filter(league: str | None, day: str | None, markets: list[str] | None) -> pd.DataFrame:
        c = cands
        if league:
            c = c[c["League"].str.contains(league, case=False) | (c["Div"] == league)]
        if day:
            c = c[c["Day"].astype(str) == day]
        if markets:
            c = c[c["Market"].isin(markets)]
        return c

    @beta_tool
    def search_selections(
        target_odds: float | None = None,
        min_probability: float | None = None,
        league: str | None = None,
        day: str | None = None,
        markets: list[str] | None = None,
        limit: int = 8,
    ) -> str:
        """Find single selections, either near target odds or the most probable per match.

        With target_odds: selections priced near those odds, best edge first. Without: the most
        probable outcome of each match. Market names: Home win,
        Draw, Away win, Home or draw, Away or draw, Over 1.5 goals, Over 2.5 goals,
        Under 2.5 goals, Under 3.5 goals, Both teams score.

        Args:
            target_odds: Decimal odds to search around, e.g. 1.7.
            min_probability: Only selections the model rates at least this likely (0-1).
            league: League name fragment such as "Premier League" or "Serie A".
            day: ISO date (YYYY-MM-DD) of kick-off, UK time.
            markets: Restrict to these market names.
            limit: Maximum rows to return.
        """
        c = _filter(league, day, markets)
        if min_probability:
            c = c[c["Prob"] >= min_probability]
        res = picks.near_odds(c, target_odds) if target_odds else picks.best_per_match(c, markets or list(model.MARKETS))
        res = res.drop_duplicates("Match") if not res.empty else res
        return json.dumps(_rows(res, limit) or "no matching selections")

    @beta_tool
    def build_accumulator(
        min_total_odds: float,
        max_total_odds: float,
        day: str | None = None,
        league: str | None = None,
        markets: list[str] | None = None,
        max_legs: int = 8,
        min_leg_probability: float = 0.55,
        tickets: int = 1,
    ) -> str:
        """Build the most probable accumulator whose combined odds fall in a range.

        One leg per match; combined odds within [min_total_odds, max_total_odds]. Market names: Home win, Draw, Away win, Home or draw,
        Away or draw, Over 1.5 goals, Over 2.5 goals, Under 2.5 goals, Under 3.5 goals,
        Both teams score.

        Args:
            min_total_odds: Lowest acceptable combined odds, e.g. 8.
            max_total_odds: Highest acceptable combined odds, e.g. 15.
            day: ISO date (YYYY-MM-DD) to restrict legs to one day.
            league: League name fragment.
            markets: Restrict legs to these market names.
            max_legs: Maximum number of legs.
            min_leg_probability: Every leg must be at least this likely (0-1); keeps the ticket
                from being a single long shot.
            tickets: How many alternative tickets (no shared matches) to return, 1-3.
        """
        c = _filter(league, day, markets)
        ts = picks.accumulators(c, n=max(1, min(tickets, 3)), min_odds=min_total_odds,
                                max_odds=max_total_odds, max_legs=max_legs,
                                min_leg_prob=min_leg_probability)
        if not ts:
            return json.dumps("no ticket fits that odds range")
        return json.dumps([{**picks.ticket_summary(t), "legs_detail": _rows(t, 20)} for t in ts])

    @beta_tool
    def team_next_match(team: str) -> str:
        """All market probabilities for a team's next loaded fixture.

        Args:
            team: Team name as the data spells it, e.g. "Man City", "Bayern Munich".
        """
        c = cands[cands["Home"].str.contains(team, case=False) | cands["Away"].str.contains(team, case=False)]
        if c.empty:
            known = sorted(set(cands["Home"]) | set(cands["Away"]))
            return json.dumps({"error": f"no fixture for '{team}'", "known_teams": known})
        first = c[c["Kickoff"] == c["Kickoff"].min()]
        return json.dumps(_rows(first.sort_values("Prob", ascending=False), 20))

    return [search_selections, build_accumulator, team_next_match]


def ask(history: list[dict], cands: pd.DataFrame, today) -> str:
    """history: [{"role": "user"|"assistant", "content": str}], ending with the user's question."""
    days = ", ".join(d.isoformat() for d in sorted(set(cands["Day"]))[:14]) or "none"
    client = anthropic.Anthropic(api_key=api_key())
    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM.format(today=today.isoformat(), days=days),
        tools=make_tools(cands),
        messages=history,
        output_config={"effort": "low"},  # chat: quick answers; raise if answers get sloppy
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",  # if a request is declined, the API retries on a fallback model
        max_iterations=8,
    )
    final = runner.until_done()
    if final.stop_reason == "refusal":
        return "Sorry, I can't help with that one."
    text = "".join(b.text for b in final.content if b.type == "text").strip()
    return text or "I couldn't find an answer in the loaded data."
