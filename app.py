from concurrent.futures import ThreadPoolExecutor

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import assistant
import data
import model
import picks

st.set_page_config(page_title="Football Predictor", page_icon="⚽", layout="wide")
ss = st.session_state

st.markdown(
    """
    <style>
    /* sidebar menu: flat collapsible groups like an admin portal */
    [data-testid="stSidebar"] [data-testid="stExpander"] details {
        border: none; background: transparent;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary p { font-weight: 600; font-size: 1rem; }
    [data-testid="stSidebar"] [data-testid="stPageLink"] a { padding-left: 2.1rem; }
    .brand { display: flex; align-items: center; gap: .7rem; margin: .2rem 0 1rem; }
    .brand .logo { font-size: 2.2rem; line-height: 1; }
    .brand .name { font-weight: 800; font-size: 1.25rem; letter-spacing: .02em; line-height: 1.1; }
    .brand .sub { font-weight: 600; font-size: .78rem; letter-spacing: .08em; opacity: .75; }
    /* KPI cards */
    .kpi { border-radius: 10px; padding: 1rem 1.2rem; color: #fff; min-height: 7.5rem;
           box-shadow: 0 2px 6px rgba(0,0,0,.12); margin-bottom: .8rem; }
    .kpi .label { font-size: .95rem; opacity: .92; }
    .kpi .value { font-size: 2rem; font-weight: 700; line-height: 1.25; }
    .kpi .sub { font-size: .82rem; opacity: .85; }
    /* floating chat bubble (bottom-right, every page) */
    .st-key-chat_fab { position: fixed; right: 1.5rem; bottom: calc(1.5rem + 2cm); z-index: 1000; width: auto !important; }
    .st-key-chat_fab button { border-radius: 999px; padding: .65rem 1.15rem; font-weight: 600;
                              box-shadow: 0 6px 18px rgba(0,0,0,.25); }
    [data-testid="stPopoverBody"]:has(.st-key-chat_box) { width: min(400px, 92vw); max-width: 92vw; }
    /* chat history height: shrink on short/narrow screens so the input box stays visible */
    [data-testid="stPopoverBody"] { --chat-h: min(380px, calc(100vh - 300px - 2cm)); }
    .st-key-chat_log, [data-testid="stLayoutWrapper"]:has(> .st-key-chat_log) {
        height: var(--chat-h) !important; max-height: var(--chat-h) !important;
        flex: 0 0 var(--chat-h) !important; }
    @media (max-width: 640px) {
        [data-testid="stPopoverBody"] { --chat-h: max(180px, calc(100vh - 560px - 2cm)); }
        .st-key-chat_fab { right: 1rem; bottom: calc(1rem + 2cm); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

KPI_COLORS = ["#e05a68", "#8b2cf5", "#2f6fe0", "#4fae8c"]


# ---------------- cached loaders ----------------
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_results(div: str, n_seasons: int) -> pd.DataFrame:
    return data.load_results(div, n_seasons)


@st.cache_resource(show_spinner=False)
def get_model(div: str, n_seasons: int, half_life: int, data_version: str) -> model.DixonColes:
    return model.DixonColes.fit(get_results(div, n_seasons), half_life_days=half_life)


@st.cache_data(ttl=300, show_spinner=False)  # downloads are cached in data.py; keep this short
def get_upcoming(div: str, n_seasons: int, data_version: str):  # so a failed schedule retries soon
    return data.load_upcoming(div, get_results(div, n_seasons))


HORIZON_DAYS = 21  # how far ahead cross-league features look


@st.cache_data(ttl=1800, show_spinner="Loading fixtures across leagues…")
def get_candidates(leagues: tuple, n_seasons: int, half_life: int, time_bucket: str):
    """One row per (upcoming match, market) across `leagues`. time_bucket refreshes the cache."""
    horizon = pd.Timestamp.now(tz=data.UK) + pd.Timedelta(days=HORIZON_DAYS)

    def one(div):
        try:
            results = get_results(div, n_seasons)
            version = str(results["Date"].max().date())
            mdl = get_model(div, n_seasons, half_life, version)
            fx, _, sched_err = get_upcoming(div, n_seasons, version)
            note = f"{data.LEAGUES[div]}: {sched_err}" if sched_err else None
            return div, mdl, fx[fx["Kickoff"] < horizon] if not fx.empty else fx, note
        except Exception as e:  # network hiccup etc.: skip this league, keep the rest
            return div, None, None, f"{data.LEAGUES[div]}: {type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=len(leagues) or 1) as pool:  # leagues download in parallel
        loaded = list(pool.map(one, leagues))
    fixtures = [(div, mdl, fx) for div, mdl, fx, _ in loaded if fx is not None and not fx.empty]
    problems = [note for *_, note in loaded if note]
    return picks.build_candidates(fixtures), problems


@st.cache_data(show_spinner="Running backtest…")
def get_backtest(div, n_seasons, half_life, test_days, min_edge, data_version):
    return model.backtest(
        get_results(div, n_seasons), test_days=test_days,
        half_life_days=half_life, min_edge=min_edge,
    )


# ---------------- shared helpers ----------------
ss.setdefault("league", "E0")


def pct(x):
    return f"{x:.0%}"


def settings():
    return ss.get("n_seasons", 3), ss.get("half_life", 180), ss.get("min_edge", 0.05)


def league_context(div: str):
    """Results + fitted model for one league, or None if the data can't be loaded."""
    n_seasons, half_life, _ = settings()
    results = get_results(div, n_seasons)
    version = str(results["Date"].max().date())
    return results, get_model(div, n_seasons, half_life, version), version


def upcoming(div: str, results: pd.DataFrame, version: str):
    fx, unmatched, sched_err = get_upcoming(div, settings()[0], version)
    if not fx.empty:  # cached result may be a few minutes old: re-check kick-off times
        fx = fx[fx["Kickoff"] > pd.Timestamp.now(tz=data.UK)]
    return fx, unmatched, sched_err


def kickoff_label(f) -> str:
    t = f.Kickoff.strftime("%a %d %b")
    return t + (f" {f.Kickoff:%H:%M}" if pd.notna(f.get("Time")) else " (time TBC)")


def _set_league():
    ss.league = ss.league_widget


def header(*crumbs: str, league_picker: bool = True):
    """Breadcrumb on the left, league switcher on the right."""
    left, right = st.columns([3, 1], vertical_alignment="center")
    path = " / ".join(crumbs[:-1])
    left.markdown(f"#### :gray[{path} /] :green[{crumbs[-1]}]" if path else f"#### {crumbs[-1]}")
    if league_picker:
        keys = list(data.LEAGUES)
        right.selectbox(
            "League", keys, index=keys.index(ss.league), format_func=data.LEAGUES.get,
            key="league_widget", on_change=_set_league, label_visibility="collapsed",
        )


def load_league_or_stop(div: str):
    try:
        return league_context(div)
    except Exception as e:  # network / missing league
        st.error(f"Could not load results for {data.LEAGUES[div]}: {e}")
        st.stop()


def kpi_cards(cards):
    cols = st.columns(len(cards))
    for col, (label, value, sub), color in zip(cols, cards, KPI_COLORS):
        col.markdown(
            f'<div class="kpi" style="background:{color}"><div class="label">{label}</div>'
            f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
            unsafe_allow_html=True,
        )


def prob_col(label):
    return st.column_config.ProgressColumn(label, format="percent", min_value=0, max_value=1)


def load_candidates(leagues) -> tuple[pd.DataFrame, list[str]]:
    n_seasons, half_life, _ = settings()
    now = pd.Timestamp.now(tz=data.UK)
    cands, failed = get_candidates(tuple(leagues), n_seasons, half_life, now.floor("30min").isoformat())
    if failed:
        st.warning("Some data could not be loaded, so picks may be incomplete. " + "; ".join(failed))
    return cands[cands["Kickoff"] > now], failed


def show_legs(df: pd.DataFrame, rank: bool = False):
    view = pd.DataFrame({
        "Kick-off (UK)": [k.strftime("%a %d %b %H:%M" if t else "%a %d %b (time TBC)")
                          for k, t in zip(df["Kickoff"], df["HasTime"])],
        "League": df["League"], "Match": df["Match"], "Pick": df["Pick"],
        "Probability": df["Prob"], "Odds": df["Odds"], "Odds from": df["OddsSource"],
        "xG": df["xG"], "Likely score": df["LikelyScore"],
    })
    if rank:
        view.index = range(1, len(view) + 1)
    st.dataframe(
        view, hide_index=not rank, width="stretch",
        column_config={
            "Probability": prob_col("Probability"),
            "Odds": st.column_config.NumberColumn(format="%.2f"),
            "Odds from": st.column_config.TextColumn(
                help="market = bookmaker average odds; model = model fair odds (1 / probability), "
                     "used until bookmakers' prices for that round are published"),
        },
    )


def in_window(fx: pd.DataFrame, days: int) -> pd.DataFrame:
    if fx.empty:
        return fx
    start = fx["Kickoff"].min().normalize()
    return fx[fx["Kickoff"] < start + pd.Timedelta(days=days)]


# ---------------- pages ----------------
def page_top_picks():
    header("Top picks", "Most likely results", league_picker=False)
    st.caption(
        "The model's single most likely outcome for every upcoming match across Europe's top "
        "leagues, ranked by probability. One pick per match."
    )
    c1, c2, c3, c4 = st.columns([3, 2, 1, 1])
    leagues = c1.multiselect(
        "Leagues", list(data.LEAGUES), default=data.TOP10, format_func=data.LEAGUES.get,
    )
    market_set = c2.selectbox("Markets", list(model.MARKET_SETS))
    n_picks = c3.number_input("Picks", 3, 20, 10)
    days = c4.number_input("Days ahead", 1, HORIZON_DAYS, 7, help="Counted from the first upcoming kick-off.")
    if not leagues:
        st.info("Choose at least one league.")
        return

    cands, _ = load_candidates(leagues)
    if cands.empty:
        st.info(
            "No upcoming fixtures found for the selected leagues. The football-data feed publishes "
            "each round a few days before kick-off, so check back closer to the weekend."
        )
        return
    start = cands["Kickoff"].min().normalize()
    end = start + pd.Timedelta(days=days)
    cands = cands[cands["Kickoff"] < end]
    best = picks.best_per_match(cands, model.MARKET_SETS[market_set]).head(int(n_picks))
    n = len(best)
    probs = best["Prob"].to_numpy()
    all_win = float(np.prod(probs))
    market = best["MarketOdds"].to_numpy(dtype=float)
    combined_market = f"market {np.prod(market):.2f}" if np.isfinite(market).all() else "market odds n/a"

    kpi_cards([
        ("Average pick probability", pct(probs.mean()), f"lowest {pct(probs.min())}"),
        ("Expected correct", f"{probs.sum():.1f} / {n}", "sum of probabilities"),
        (f"Chance all {n} win", f"{all_win:.1%}", "treating matches as independent"),
        ("Combined odds", f"{1 / all_win:.2f}", f"fair · {combined_market}"),
    ])
    show_legs(best, rank=True)
    empty = sorted(set(data.LEAGUES[d] for d in leagues) - set(cands["League"]))
    st.caption(
        f"Fixtures from {start:%a %d %b} to {end - pd.Timedelta(days=1):%a %d %b}. "
        "Market odds are bookmaker averages, available only once football-data publishes the round "
        "(1X2 and over/under 2.5 only)."
        + (f" No fixtures in this window: {', '.join(empty)}." if empty else "")
    )

    st.subheader("How many picks are likely to come in?")
    dist = model.hits_distribution(probs)
    chart_df = pd.DataFrame({
        "Correct picks": np.arange(n + 1),
        "Probability": dist,
        "At least this many": dist[::-1].cumsum()[::-1],
    })
    st.altair_chart(
        alt.Chart(chart_df).mark_bar(color=KPI_COLORS[2]).encode(
            x=alt.X("Correct picks:O"),
            y=alt.Y("Probability:Q", axis=alt.Axis(format="%")),
            tooltip=["Correct picks", alt.Tooltip("Probability:Q", format=".1%"),
                     alt.Tooltip("At least this many:Q", format=".1%")],
        ).properties(height=260),
        width="stretch",
    )
    st.info(
        f"Each pick is likely on its own, but all {n} landing together happens only about "
        f"{all_win:.1%} of the time (roughly 1 in {1 / all_win:.0f}). "
        "Short-priced favourites also rarely offer value: see the Backtest page.",
        icon=":material/info:",
    )


def page_daily_pick():
    header("Top picks", "Daily pick", league_picker=False)
    st.caption(
        "The most probable accumulator for a day whose combined odds land in your target range. "
        "One leg per match, picked across Europe's top leagues."
    )
    cands, _ = load_candidates(data.TOP10)
    if cands.empty:
        st.info("No upcoming fixtures loaded yet.")
        return

    days = sorted(set(cands["Day"]))
    per_day = cands.groupby("Day")["Match"].nunique()
    busy = [i for i, d in enumerate(days) if per_day[d] >= 6]  # default: first proper match day
    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
    day = c1.selectbox(
        "Day", ["Next 3 days"] + days, index=busy[0] + 1 if busy else 0,
        format_func=lambda d: d if isinstance(d, str) else f"{d:%A %d %b} ({per_day[d]} matches)",
    )
    lo, hi = c2.slider("Combined odds", 2.0, 50.0, (8.0, 15.0), step=0.5)
    max_legs = c3.number_input("Max legs", 2, 12, 8)
    min_leg = c4.number_input(
        "Min leg chance", 0.30, 0.95, 0.60, step=0.05, format="%.2f",
        help="Every leg must be at least this likely. Stops the ticket from being one long shot.",
    )
    market_set = st.radio("Markets", list(model.MARKET_SETS), index=2, horizontal=True)

    if isinstance(day, str):
        pool = cands[cands["Day"].isin(days[:3])]
    else:
        pool = cands[cands["Day"] == day]
    pool = pool[pool["Market"].isin(model.MARKET_SETS[market_set])]
    tickets = picks.accumulators(pool, n=3, min_odds=lo, max_odds=hi,
                                 max_legs=int(max_legs), min_leg_prob=min_leg)
    if not tickets:
        st.info(
            f"No ticket between {lo:g} and {hi:g} fits these settings ({pool['Match'].nunique()} matches "
            "available). Try more legs, a lower minimum leg chance, or another day."
        )
        return

    for i, t in enumerate(tickets):
        s = picks.ticket_summary(t)
        st.subheader("Daily pick" if i == 0 else f"Alternative {i}")
        kpi_cards([
            ("Combined odds", f"{s['combined_odds']:.2f}", f"{s['odds_source']} odds"),
            ("Model chance all win", f"{s['model_probability']:.1%}",
             f"about 1 in {1 / s['model_probability']:.0f}"),
            ("Legs", str(s["legs"]), f"weakest leg {t['Prob'].min():.0%}"),
            ("Expected return", f"{s['expected_return_per_unit']:+.0%}",
             "per unit staked, if the model is right"),
        ])
        show_legs(t)
    st.info(
        "When the odds come from the model (bookmakers haven't priced the round yet), the expected "
        "return is 0% by construction: the ticket is the *most likely* way to reach those odds, not a "
        "value bet. Real bookmaker odds include a margin, so the real return is usually negative.",
        icon=":material/info:",
    )


EXAMPLES = [
    "Best prediction at odds 1.7",
    "Acca with odds 8 to 15 for Saturday",
    "Safest Premier League picks this weekend",
]


def _queue_example(text: str):
    ss.chat_pending = text


def _clear_chat():
    ss.chat = []


def chat_widget():
    """Floating chat bubble, bottom-right on every page."""
    ss.setdefault("chat", [])
    use_claude = bool(assistant.api_key())
    with st.container(key="chat_fab"):
        with st.popover("Ask", icon=":material/chat:", type="primary", key="chat_pop"):
            top, clear = st.columns([5, 1], vertical_alignment="center")
            top.markdown("**Ask the predictor**")
            if ss.chat:
                clear.button("", icon=":material/delete:", help="Clear chat", on_click=_clear_chat,
                             type="tertiary", key="chat_clear")
            st.caption(
                "Powered by Claude, answering from the app's predictions." if use_claude else
                "Basic mode: ask about odds, odds ranges, markets, leagues, days or teams."
            )
            log = st.container(height=380, border=False, autoscroll=True, key="chat_log")
            question = st.chat_input("Ask about picks, odds or teams…", key="chat_box")
            question = question or ss.pop("chat_pending", None)

            if question:
                ss.chat.append({"role": "user", "content": question})
                with log, st.spinner("Thinking…"):
                    cands, _ = load_candidates(data.TOP10)  # loaded only when someone asks
                    today = pd.Timestamp.now(tz=data.UK).date()
                    reply = None
                    if use_claude:
                        try:
                            reply = assistant.ask(ss.chat, cands, today)
                        except Exception as e:  # API/network error: fall back to rule-based
                            reply = f"_(Claude unavailable: {type(e).__name__}; basic answer below.)_\n\n"
                            reply += picks.answer(question, cands, today)
                    if reply is None:
                        reply = picks.answer(question, cands, today)
                ss.chat.append({"role": "assistant", "content": reply})

            with log:
                if not ss.chat:
                    st.markdown("Hi! Try one of these:")
                    for i, ex in enumerate(EXAMPLES):
                        st.button(ex, key=f"chat_ex_{i}", on_click=_queue_example, args=(ex,),
                                  width="stretch")
                for m in ss.chat:
                    with st.chat_message(m["role"]):
                        st.markdown(m["content"])


def page_fixtures():
    header("Fixtures", "Upcoming")
    div = ss.league
    results, mdl, version = load_league_or_stop(div)
    min_edge = settings()[2]
    fx, unmatched, sched_err = upcoming(div, results, version)
    if sched_err:
        st.warning(f"Only fixtures with published odds are shown: {sched_err}.")

    days = st.slider("Days ahead", 1, 21, 7, help="Counted from the first upcoming kick-off.")
    fx = in_window(fx, days)
    if fx.empty:
        st.info(
            "No upcoming fixtures for this league yet. The football-data feed publishes each round "
            "a few days before kick-off."
            + ("" if div in data.OPENFOOTBALL else " This league has no full-season schedule source.")
        )
        return

    rows, skipped = [], 0
    for rec in fx.to_dict("records"):  # not itertuples: it renames "Avg>2.5"
        f = pd.Series(rec)
        if not (mdl.knows(f.HomeTeam) and mdl.knows(f.AwayTeam)):
            skipped += 1
            continue
        p = mdl.predict(f.HomeTeam, f.AwayTeam)
        picks = {
            "Home": (p["home"], f.get("AvgH")),
            "Draw": (p["draw"], f.get("AvgD")),
            "Away": (p["away"], f.get("AvgA")),
            "Over 2.5": (p["over_2_5"], f.get("Avg>2.5")),
            "Under 2.5": (p["under_2_5"], f.get("Avg<2.5")),
        }
        edges = {k: model.edge(pr, o) for k, (pr, o) in picks.items()}
        best = max(edges, key=lambda k: -1 if np.isnan(edges[k]) else edges[k])
        value = (
            f"{best} @ {picks[best][1]:.2f} (+{edges[best]:.0%})"
            if not np.isnan(edges[best]) and edges[best] > min_edge else "—"
        )
        has_odds = pd.notna(f.get("AvgH"))
        rows.append({
            "Kick-off (UK)": kickoff_label(f),
            "Match": f"{f.HomeTeam} vs {f.AwayTeam}",
            "xG": f"{p['xG_home']:.2f} – {p['xG_away']:.2f}",
            "Home": p["home"], "Draw": p["draw"], "Away": p["away"],
            "Over 2.5": p["over_2_5"], "BTTS": p["btts"],
            "Likely score": p["top_scores"][0][0],
            "Fair odds (H/D/A)": f"{1/p['home']:.2f} / {1/p['draw']:.2f} / {1/p['away']:.2f}",
            "Market odds (H/D/A)": f"{f.AvgH:.2f} / {f.AvgD:.2f} / {f.AvgA:.2f}" if has_odds else "—",
            "Value bet": value,
        })
    if not rows:
        st.info("Fixtures found, but the teams are not in the model's history.")
        return
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, width="stretch",
        column_config={k: prob_col(k) for k in ["Home", "Draw", "Away", "Over 2.5", "BTTS"]},
    )
    n_odds = int(fx["AvgH"].notna().sum()) if "AvgH" in fx else 0
    notes = [
        "Matches that have already kicked off are hidden.",
        "Fair odds = 1 / model probability. Value bet = best market where the model edge exceeds your threshold.",
    ]
    if n_odds < len(fx):
        notes.append(
            f"Market odds are available for {n_odds} of {len(fx)} fixtures; the rest come from the "
            "season schedule and get odds once football-data publishes that round."
        )
    if skipped:
        notes.append(f"{skipped} fixture(s) skipped: team not in the model's history.")
    if unmatched:
        notes.append("Unmatched schedule names: " + ", ".join(unmatched) + ".")
    st.caption(" ".join(notes))


def page_predictor():
    header("Predictor", "Match predictor")
    results, mdl, _ = load_league_or_stop(ss.league)
    c1, c2 = st.columns(2)
    teams = mdl.teams
    home = c1.selectbox("Home team", teams, index=0)
    away = c2.selectbox("Away team", teams, index=min(1, len(teams) - 1))
    if home == away:
        st.info("Pick two different teams.")
        return
    p = mdl.predict(home, away)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(f"{home} win", pct(p["home"]), f"fair {1/p['home']:.2f}", delta_color="off")
    m2.metric("Draw", pct(p["draw"]), f"fair {1/p['draw']:.2f}", delta_color="off")
    m3.metric(f"{away} win", pct(p["away"]), f"fair {1/p['away']:.2f}", delta_color="off")
    m4.metric("Over 2.5 goals", pct(p["over_2_5"]))
    m5.metric("Both teams score", pct(p["btts"]))
    st.write(f"**Expected goals:** {home} {p['xG_home']:.2f} – {p['xG_away']:.2f} {away}")
    st.write("**Most likely scores:** " + " · ".join(f"{s} ({pr:.0%})" for s, pr in p["top_scores"]))

    mat = p["matrix"][:6, :6]
    grid = pd.DataFrame(
        [(i, j, mat[i, j]) for i in range(6) for j in range(6)],
        columns=[f"{home} goals", f"{away} goals", "Probability"],
    )
    base = alt.Chart(grid).encode(
        x=alt.X(f"{away} goals:O"), y=alt.Y(f"{home} goals:O", sort="descending"),
    )
    heat = base.mark_rect().encode(
        color=alt.Color("Probability:Q", scale=alt.Scale(scheme="greens"), legend=None),
        tooltip=[alt.Tooltip("Probability:Q", format=".1%")],
    )
    text = base.mark_text().encode(text=alt.Text("Probability:Q", format=".0%"))
    st.altair_chart((heat + text).properties(height=360, title="Scoreline probabilities"),
                    width="stretch")

    st.subheader("Recent head-to-head")
    h2h = results[
        ((results.HomeTeam == home) & (results.AwayTeam == away))
        | ((results.HomeTeam == away) & (results.AwayTeam == home))
    ].sort_values("Date", ascending=False)
    if h2h.empty:
        st.caption("No meetings in the loaded history.")
    else:
        st.dataframe(
            h2h[["Date", "HomeTeam", "FTHG", "FTAG", "AwayTeam"]].head(10)
            .assign(Date=lambda d: d.Date.dt.date),
            hide_index=True, width="stretch",
        )


def page_ratings():
    header("Analysis", "Team ratings")
    results, mdl, version = load_league_or_stop(ss.league)
    st.caption(
        f"{len(results)} matches · {len(mdl.teams)} teams · latest result {version} · "
        f"home advantage ×{np.exp(mdl.home_adv):.2f} goals"
    )
    r = mdl.ratings()
    r["Goals for /game (vs avg)"] = np.exp(r["Attack"])
    r["Goals against /game (vs avg)"] = np.exp(-r["Defence"])
    st.dataframe(
        r[["Team", "Overall", "Attack", "Defence", "Goals for /game (vs avg)",
           "Goals against /game (vs avg)", "Matches"]].round(2),
        hide_index=True, width="stretch",
    )
    st.caption("Ratings are time-weighted. Attack > 0 scores more than average; Defence > 0 concedes less than average.")


def page_form():
    header("Analysis", "Form")
    results, mdl, _ = load_league_or_stop(ss.league)
    recent = []
    for t in mdl.teams:
        g = results[(results.HomeTeam == t) | (results.AwayTeam == t)].tail(5)
        form = "".join(
            ("W" if (row.HomeTeam == t and row.FTR == "H") or (row.AwayTeam == t and row.FTR == "A")
             else "D" if row.FTR == "D" else "L")
            for row in g.itertuples()
        )
        recent.append({"Team": t, "Last 5 (oldest → newest)": form, "Last match": g.Date.max().date()})
    st.dataframe(pd.DataFrame(recent), hide_index=True, width="stretch")


def page_backtest():
    header("Analysis", "Backtest")
    results, mdl, version = load_league_or_stop(ss.league)
    n_seasons, half_life, min_edge = settings()
    st.write(
        "Walk-forward test: the model is refit every 14 days using only past data, then predicts the "
        "next matches. Compared against bookmaker market-average odds (margin removed)."
    )
    test_days = st.slider("Test window (days)", 30, 365, 120, step=30)
    if not st.button("Run backtest", type="primary"):
        return
    bt, s = get_backtest(ss.league, n_seasons, half_life, test_days, min_edge, version)
    if not s:
        st.info("Not enough data for a backtest in this window.")
        return
    c = st.columns(4)
    c[0].metric("Matches tested", s["matches"])
    c[1].metric("Model accuracy", pct(s["accuracy"]),
                f"{s['accuracy'] - s.get('bookie_accuracy', s['accuracy']):+.1%} vs bookies")
    c[2].metric("Model log-loss (lower better)", f"{s['log_loss']:.3f}",
                f"{s['log_loss'] - s.get('bookie_log_loss', s['log_loss']):+.3f} vs bookies",
                delta_color="inverse")
    if "roi" in s:
        c[3].metric(f"Value bets (edge > {min_edge:.0%})", s["value_bets"],
                    f"ROI {s['roi']:+.1%} ({s['profit_units']:+.1f} units)")
    if s.get("bookie_log_loss") and s["log_loss"] > s["bookie_log_loss"]:
        st.info("The bookmakers' odds were more accurate than the model in this window, "
                "so positive 'edges' are more likely model error than real value.")
    st.dataframe(bt.assign(Date=bt.Date.dt.date), hide_index=True, width="stretch")


# ---------------- navigation ----------------
P = {
    "top": st.Page(page_top_picks, title="Most likely 10", icon=":material/star:",
                   url_path="most-likely", default=True),
    "daily": st.Page(page_daily_pick, title="Daily pick (8–15)", icon=":material/confirmation_number:",
                     url_path="daily-pick"),
    "fixtures": st.Page(page_fixtures, title="Upcoming fixtures", icon=":material/event:",
                        url_path="fixtures"),
    "predictor": st.Page(page_predictor, title="Match predictor", icon=":material/sports_soccer:",
                         url_path="predictor"),
    "ratings": st.Page(page_ratings, title="Team ratings", icon=":material/leaderboard:",
                       url_path="ratings"),
    "form": st.Page(page_form, title="Form", icon=":material/trending_up:", url_path="form"),
    "backtest": st.Page(page_backtest, title="Backtest", icon=":material/science:",
                        url_path="backtest"),
}
MENU = [
    ("Top picks", ":material/emoji_events:", ["top", "daily"]),
    ("Fixtures", ":material/calendar_month:", ["fixtures"]),
    ("Predictor", ":material/online_prediction:", ["predictor"]),
    ("Analysis", ":material/analytics:", ["ratings", "form", "backtest"]),
]

current = st.navigation(list(P.values()), position="hidden")

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="logo">⚽</div><div>'
        '<div class="name">FOOTBALL PREDICTOR</div><div class="sub">MATCH ANALYTICS</div>'
        "</div></div>",
        unsafe_allow_html=True,
    )
    query = st.text_input(
        "Search", placeholder="Search", icon=":material/search:", label_visibility="collapsed",
    ).strip().lower()
    for label, icon, keys in MENU:
        items = [P[k] for k in keys
                 if not query or query in P[k].title.lower() or query in label.lower()]
        if not items:
            continue
        open_ = bool(query) or any(P[k].url_path == current.url_path for k in keys)
        with st.expander(label, icon=icon, expanded=open_):
            for page in items:
                st.page_link(page)

    with st.expander("Settings", icon=":material/settings:"):  # always rendered: keeps state
        st.slider("Seasons of history", 1, 5, 3, key="n_seasons")
        st.slider(
            "Form half-life (days)", 30, 720, 180, step=30, key="half_life",
            help="How quickly old matches lose influence. Lower = more weight on recent form.",
        )
        st.slider(
            "Value-bet threshold (edge)", 0.0, 0.30, 0.05, step=0.01, key="min_edge",
            help="Flag a bet when model probability × odds − 1 exceeds this.",
        )
    st.caption(
        "Data: [football-data.co.uk](https://www.football-data.co.uk) and "
        "[openfootball](https://github.com/openfootball/football.json), both free."
    )
    st.caption(
        ":material/warning: Predictions are estimates, not guarantees. "
        "Only bet what you can afford to lose."
    )

chat_widget()  # before the page: pages may st.stop() early
current.run()
