from concurrent.futures import ThreadPoolExecutor

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import assistant
import competitions
import data
import deep
import history
import model
import picks
import teamnews

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
JOINT_L2 = 1.0  # pulls low-data teams (e.g. 3 cup games) towards average instead of extremes
MIN_MATCHES_FOR_PICKS = 10  # teams with less history are shown, but left out of pick features


@st.cache_resource(ttl=6 * 3600, show_spinner="Fitting the European model (all leagues + cups)…")
def get_europe_model(n_seasons: int, half_life: int):
    """One model over every domestic league plus European cup results, so clubs from different
    leagues can be compared. Returns (model, cup name -> model name mapping)."""
    with ThreadPoolExecutor(max_workers=len(data.LEAGUES)) as pool:
        domestic = list(pool.map(lambda d: get_results(d, n_seasons), data.LEAGUES))
    matches, mapping = competitions.europe_matches(domestic, seasons=2)
    return model.DixonColes.fit(matches, half_life_days=half_life, l2=JOINT_L2), mapping


@st.cache_resource(ttl=6 * 3600, show_spinner="Fitting the international model…")
def get_intl_model():
    """National teams, last 8 years of internationals; slower decay since teams play rarely."""
    return model.DixonColes.fit(competitions.international_results(8), half_life_days=730, l2=JOINT_L2)


@st.cache_resource(ttl=6 * 3600, show_spinner="Training the deep model (18 leagues, 8 seasons)…")
def get_deep_engine(version: int = deep.ENGINE_VERSION) -> deep.DeepEngine:
    """`version` is part of the cache key: a model change forces a rebuild."""
    n = data.season_start_year() - deep.WARMUP_FROM + 1
    with ThreadPoolExecutor(max_workers=6) as pool:
        raw = dict(zip(data.LEAGUES, pool.map(lambda d: data.load_results(d, n), data.LEAGUES)))
    return deep.DeepEngine(raw)


@st.cache_data(ttl=1800, show_spinner=False)
def get_team_news() -> dict:
    try:
        return teamnews.team_news()
    except Exception:  # FPL API down: predictions still work without team news
        return {}


def deep_result(eng, div, home, away, kickoff, odds=None, dc=None, news=None) -> dict | None:
    """Deep model probabilities, then a cautious team-news adjustment (Premier League)."""
    r = eng.predict(div, home, away, kickoff, odds)
    if r is None:
        return None
    final = np.array(r["best"], dtype=float)
    r["news_adj"] = None
    if news and dc is not None and dc.knows(home) and dc.knows(away):
        mh = teamnews.xg_multiplier(news.get(home))
        ma = teamnews.xg_multiplier(news.get(away))
        if mh < 1 or ma < 1:  # shift by how much the absences move the goals model
            lam, mu = dc.expected_goals(home, away)
            delta = dc.result_probs_from_xg(lam * mh, mu * ma) - dc.result_probs_from_xg(lam, mu)
            final = np.clip(final + delta, 0.005, None)
            final /= final.sum()
            r["news_adj"] = final
    r["final"] = final
    return r


def comp_fixtures(code: str, n_seasons: int, half_life: int) -> tuple[model.DixonColes, pd.DataFrame]:
    """Model + upcoming fixtures for a cup or international competition code."""
    if code in competitions.CUPS:
        mdl, mapping = get_europe_model(n_seasons, half_life)
        return mdl, competitions.cup_fixtures(code, mapping)
    mdl = get_intl_model()
    return mdl, competitions.international_fixtures(code, set(mdl.teams))


@st.cache_data(ttl=1800, show_spinner="Loading fixtures across competitions…")
def get_candidates(leagues: tuple, n_seasons: int, half_life: int, time_bucket: str):
    """One row per (upcoming match, market) across `leagues`. time_bucket refreshes the cache."""
    horizon = pd.Timestamp.now(tz=data.UK) + pd.Timedelta(days=HORIZON_DAYS)

    def one(div):
        name = ALL_NAMES[div]
        try:
            if div in competitions.COMPETITIONS:
                mdl, fx = comp_fixtures(div, n_seasons, half_life)
                note = None
            else:
                results = get_results(div, n_seasons)
                version = str(results["Date"].max().date())
                mdl = get_model(div, n_seasons, half_life, version)
                fx, _, sched_err = get_upcoming(div, n_seasons, version)
                note = f"{name}: {sched_err}" if sched_err else None
            return div, mdl, fx[fx["Kickoff"] < horizon] if not fx.empty else fx, note
        except Exception as e:  # network hiccup etc.: skip this competition, keep the rest
            return div, None, None, f"{name}: {type(e).__name__}: {e}"

    try:  # deep model + team news for domestic leagues (falls back to Dixon-Coles if it fails)
        eng = get_deep_engine(deep.ENGINE_VERSION) if any(d in data.LEAGUES for d in leagues) else None
        news = get_team_news()
    except Exception:
        eng, news = None, {}

    def result_fn(div, mdl):
        if eng is None or div not in data.LEAGUES:
            return None

        def fn(f):
            odds = tuple(f.get(c) for c in ("AvgH", "AvgD", "AvgA"))
            odds = odds if all(o is not None and pd.notna(o) and o > 1 for o in odds) else None
            r = deep_result(eng, div, f["HomeTeam"], f["AwayTeam"], f["Kickoff"], odds, mdl,
                            news if div == "E0" else None)
            return None if r is None else r["final"]
        return fn

    # the joint European model is shared by three cups: build it once before fanning out
    if any(d in competitions.CUPS for d in leagues):
        try:
            get_europe_model(n_seasons, half_life)
        except Exception:
            pass  # each cup reports its own error below

    with ThreadPoolExecutor(max_workers=len(leagues) or 1) as pool:  # leagues download in parallel
        loaded = list(pool.map(one, leagues))
    fixtures = [(div, mdl, fx, result_fn(div, mdl)) for div, mdl, fx, _ in loaded
                if fx is not None and not fx.empty]
    problems = [note for *_, note in loaded if note]
    return picks.build_candidates(fixtures, min_matches=MIN_MATCHES_FOR_PICKS), problems


@st.cache_data(ttl=3 * 3600, show_spinner="Rebuilding past daily picks…")
def get_past_pools(days_back: int, n_seasons: int, half_life: int, today_iso: str) -> dict:
    """Candidate selections for each past day, rebuilt with only earlier data (see history.py)."""
    eng = get_deep_engine(deep.ENGINE_VERSION)
    with ThreadPoolExecutor(max_workers=6) as pool:
        domestic = dict(zip(data.LEAGUES, pool.map(lambda d: get_results(d, n_seasons), data.LEAGUES)))
    return history.past_pools(days_back, domestic, eng.past, half_life,
                              pd.Timestamp(today_iso).date(), MIN_MATCHES_FOR_PICKS)


@st.cache_data(show_spinner="Running backtest…")
def get_backtest(div, n_seasons, half_life, test_days, min_edge, data_version):
    return model.backtest(
        get_results(div, n_seasons), test_days=test_days,
        half_life_days=half_life, min_edge=min_edge,
    )


# ---------------- shared helpers ----------------
ss.setdefault("league", "E0")
ALL_NAMES = {**data.LEAGUES, **competitions.COMPETITIONS}


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
        "leagues, the European cups and the Nations League, ranked by probability. One pick per match. "
        f"Matches involving a team with fewer than {MIN_MATCHES_FOR_PICKS} games of history are left out."
    )
    c1, c2, c3, c4 = st.columns([3, 2, 1, 1])
    leagues = c1.multiselect(
        "Competitions", list(ALL_NAMES), default=competitions.DEFAULT_PICK_SET, format_func=ALL_NAMES.get,
    )
    market_set = c2.selectbox("Markets", list(model.MARKET_SETS))
    n_picks = c3.number_input("Picks", 3, 20, 10)
    days = c4.number_input("Days ahead", 1, HORIZON_DAYS, 7, help="Counted from the first upcoming kick-off.")
    if not leagues:
        st.info("Choose at least one competition.")
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
    empty = sorted(set(ALL_NAMES[d] for d in leagues) - set(cands["League"]))
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
        "One leg per match, picked across Europe's top leagues, the European cups and the Nations League."
    )
    cands, _ = load_candidates(competitions.DEFAULT_PICK_SET)

    days = sorted(set(cands["Day"])) if not cands.empty else []
    per_day = cands.groupby("Day")["Match"].nunique() if not cands.empty else pd.Series(dtype=int)
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
    markets = model.MARKET_SETS[market_set]
    ticket_kw = dict(min_odds=lo, max_odds=hi, max_legs=int(max_legs), min_leg_prob=min_leg)

    # ---------- today's / upcoming pick
    if cands.empty:
        st.info("No upcoming fixtures loaded yet.")
    else:
        pool = cands[cands["Day"].isin(days[:3])] if isinstance(day, str) else cands[cands["Day"] == day]
        pool = pool[pool["Market"].isin(markets)]
        tickets = picks.accumulators(pool, n=3, **ticket_kw)
        if not tickets:
            st.info(
                f"No ticket between {lo:g} and {hi:g} fits these settings ({pool['Match'].nunique()} "
                "matches available). Try more legs, a lower minimum leg chance, or another day."
            )
        for i, t in enumerate(tickets):
            s = picks.ticket_summary(t)
            st.subheader("Daily pick" if i == 0 else f"Alternative {i}")
            kpi_cards([
                ("Combined odds", f"{s['combined_odds']:.2f}", f"{s['odds_source']} odds"),
                ("Model chance all win", f"{s['model_probability']:.1%}",
                 f"about 1 in {1 / s['model_probability']:.0f}"),
                ("Legs", str(s["legs"]), f"weakest leg {t['Prob'].min():.0%}"),
                ("Expected return", f"{(s['expected_return_per_unit'] if abs(s['expected_return_per_unit']) > 1e-6 else 0.0):+.0%}",
                 "per unit staked, if the model is right"),
            ])
            show_legs(t)
        if tickets:
            st.info(
                "When the odds come from the model (bookmakers haven't priced the round yet), the "
                "expected return is 0% by construction: the ticket is the *most likely* way to reach "
                "those odds, not a value bet. Real bookmaker odds include a margin, so the real return "
                "is usually negative.",
                icon=":material/info:",
            )

    # ---------- track record
    st.subheader("Previous daily picks")
    lookback = st.select_slider("Look back", [7, 14, 21, 30, 45, 60], value=21,
                                format_func=lambda d: f"{d} days")
    n_seasons, half_life, _ = settings()
    today = pd.Timestamp.now(tz=data.UK).date()
    try:
        pools = get_past_pools(lookback, n_seasons, half_life, today.isoformat())
    except Exception as e:
        st.warning(f"Could not rebuild past picks: {e}")
        return
    pools = {d: p[p["Market"].isin(markets)] for d, p in pools.items()}
    record = history.track_record(pools, **ticket_kw)
    if not record:
        st.info("No past days in this window had enough matches for a ticket with these settings.")
        return

    settled = [r for r in record if r["status"] != "Pending"]
    won = [r for r in settled if r["status"] == "Won"]
    profit = sum(r["combined_odds"] - 1 if r["status"] == "Won" else -1 for r in settled)
    expected = sum(r["model_probability"] for r in settled)
    kpi_cards([
        ("Tickets won", f"{len(won)} / {len(settled)}",
         f"{len(won) / len(settled):.0%} hit rate" if settled else "none settled yet"),
        ("Model expected", f"{expected:.1f} wins", "sum of each ticket's chance"),
        ("Profit", f"{profit:+.1f} units", "1 unit per ticket at its combined odds"),
        ("Legs won", f"{sum(r['legs_won'] for r in settled)} / {sum(r['legs'] for r in settled)}",
         "individual selections"),
    ])

    icon = {"Won": "✅ Won", "Lost": "❌ Lost", "Pending": "⏳ Pending"}
    summary = pd.DataFrame([{
        "Date": r["date"].strftime("%a %d %b"), "Legs": r["legs"],
        "Legs won": f"{r['legs_won']} / {r['legs']}",
        "Combined odds": r["combined_odds"], "Model chance": r["model_probability"],
        "Result": icon[r["status"]],
    } for r in record])
    st.dataframe(summary, hide_index=True, width="stretch", column_config={
        "Combined odds": st.column_config.NumberColumn(format="%.2f"),
        "Model chance": st.column_config.NumberColumn(format="percent"),
    })

    for r in record:
        t = r["ticket"]
        with st.expander(f"{r['date']:%A %d %b} · {r['legs']} legs @ {r['combined_odds']:.2f} · "
                         f"{icon[r['status']]}"):
            st.dataframe(pd.DataFrame({
                "Kick-off (UK)": t["Kickoff"].dt.strftime("%H:%M"), "League": t["League"],
                "Match": t["Match"], "Pick": t["Pick"], "Probability": t["Prob"],
                "Odds": t["Odds"], "Score": t["Score"],
                "Result": t["Won"].map(lambda w: "⏳ Pending" if w is None or pd.isna(w)
                                       else "✅ Won" if w else "❌ Lost"),
            }), hide_index=True, width="stretch", column_config={
                "Probability": prob_col("Probability"),
                "Odds": st.column_config.NumberColumn(format="%.2f"),
            })
    st.caption(
        "Each past ticket is rebuilt with only the information available before that day (models "
        "refitted on earlier matches) and the same settings as above, then graded with the final "
        "scores. Odds are bookmaker averages where the results feed has them, otherwise the model's "
        "fair odds. Results feeds can lag a few days, so the most recent days may be missing "
        "matches or show as pending."
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
                    cands, _ = load_candidates(competitions.DEFAULT_PICK_SET)  # loaded only when someone asks
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


def last_results(results: pd.DataFrame, team: str, n: int = 6) -> str:
    g = results[(results.HomeTeam == team) | (results.AwayTeam == team)].tail(n)
    out = []
    for r in g.itertuples():
        home = r.HomeTeam == team
        gf, ga = (r.FTHG, r.FTAG) if home else (r.FTAG, r.FTHG)
        res = "W" if gf > ga else "L" if gf < ga else "D"
        out.append(f"{res} {gf}-{ga} {'v' if home else '@'} {r.AwayTeam if home else r.HomeTeam}")
    return " · ".join(reversed(out)) or "—"


def news_summary(news: dict | None, limit: int = 3) -> str:
    if not news or news["players"].empty:
        return ""
    p = news["players"]
    key = p[(p["Chance to play"].isin(["0%", "25%"]))].head(limit)
    return ", ".join(f"{r.Player} ({r.Status.lower()})" for r in key.itertuples())


def odds_tuple(f) -> tuple | None:
    o = (f.get("AvgH"), f.get("AvgD"), f.get("AvgA"))
    return o if all(x is not None and pd.notna(x) and float(x) > 1 for x in o) else None


def value_signal(r: dict, odds: tuple | None, threshold: float = 0.05):
    """(outcome index, edge) where the deep model + market disagrees most with the price."""
    if odds is None or r.get("with_market") is None:
        return None
    edges = r["final"] * np.array(odds, dtype=float) - 1
    i = int(edges.argmax())
    return (i, float(edges[i])) if edges[i] > threshold else None


def deep_dive(div: str, home: str, away: str, kickoff: pd.Timestamp, odds: tuple | None = None):
    """Everything the model knows about one match, and why it leans the way it does."""
    eng = get_deep_engine(deep.ENGINE_VERSION)
    results, dc, _ = league_context(div)
    news = get_team_news() if div == "E0" else {}
    r = deep_result(eng, div, home, away, kickoff, odds, dc, news)
    if r is None:
        st.info("Not enough league history for one of these teams to run the deep model.")
        return
    labels = [f"{home} win", "Draw", f"{away} win"]

    # 1) every view of the probabilities side by side
    rows = [("Deep model (own signals)", r["own"])]
    if r["market"] is not None:
        rows += [("Bookmakers (margin removed)", r["market"]),
                 ("Deep model + bookmaker price", r["with_market"])]
    if r["news_adj"] is not None:
        rows.append(("After team news", r["news_adj"]))
    pdc = dc.predict(home, away)
    rows.append(("Season strength model (older model)", np.array([pdc["home"], pdc["draw"], pdc["away"]])))
    st.dataframe(
        pd.DataFrame([{"Source": n, labels[0]: p[0], labels[1]: p[1], labels[2]: p[2]} for n, p in rows]),
        hide_index=True, width="stretch", column_config={c: prob_col(c) for c in labels},
    )

    # 2) verdict against the market
    sig = value_signal(r, odds)
    vb = next((v for v in eng.report["value_bets"] if v[0] == "edge > 5%" and v[1] == "all odds"), None)
    track = (f" In testing on {eng.report['test_matches']:,} unseen matches, bets with an edge above 5% "
             f"returned {vb[3]:+.1%} (±{vb[4]:.1%}), which is not yet proof of a real edge."
             if vb else "")
    if sig:
        i, e = sig
        st.success(
            f"**Possible value: {labels[i]} at {odds[i]:.2f}.** The model gives {r['final'][i]:.0%}, "
            f"the bookmakers {r['market'][i]:.0%} (edge {e:+.0%})." + track, icon=":material/target:")
    elif odds is not None:
        st.info("The deep model broadly agrees with the bookmakers here: no value signal.",
                icon=":material/balance:")
    else:
        st.info("Bookmaker odds for this match aren't published yet, so this is the model's own view. "
                "Check back when odds appear to see whether it disagrees with the market.",
                icon=":material/schedule:")

    # 3) why: contribution of each signal
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Why the model leans this way**")
        why = r["why"].copy()
        why["Direction"] = np.where(why["Home vs away"] >= 0, f"favours {home}", f"favours {away}")
        st.altair_chart(
            alt.Chart(why).mark_bar().encode(
                x=alt.X("Home vs away:Q", title=f"← {away}    |    {home} →"),
                y=alt.Y("Signal:N", sort="-x", title=None),
                color=alt.Color("Direction:N", legend=None,
                                scale=alt.Scale(range=[KPI_COLORS[2], KPI_COLORS[0]])),
                tooltip=["Signal", "Direction", alt.Tooltip("Home vs away:Q", format="+.2f")],
            ).properties(height=230),
            width="stretch",
        )
        st.caption("Bar length = how strongly each signal pushes the prediction (log-odds). Signals "
                   "overlap (Elo already counts recent results), so a small bar can point the "
                   "other way as a correction once the others are counted.")
    with c2:
        st.markdown("**Form and context**")
        x = r["signals"]
        fh, fa = eng.recent(div, home), eng.recent(div, away)
        table = pd.DataFrame({
            "": ["Elo rating", "Shots on target for (form)", "Shots on target against (form)",
                 "Points per game (form)", "Points per game (season)", "Days since last league match"],
            home: [fh["Elo"], fh["Shots on target for"], fh["Shots on target against"],
                   fh["Points per game (form)"], round(x["ppg_h"], 2), x["rest_h"]],
            away: [fa["Elo"], fa["Shots on target for"], fa["Shots on target against"],
                   fa["Points per game (form)"], round(x["ppg_a"], 2), x["rest_a"]],
        })
        st.dataframe(table.astype(str), hide_index=True, width="stretch")

    st.markdown(f"**Last 6 league matches** (newest first)  \n{home}: {last_results(results, home)}  \n"
                f"{away}: {last_results(results, away)}")

    # 4) head-to-head
    meetings = eng.head_to_head(div, home, away)
    if meetings:
        st.markdown(f"**Head-to-head** (last {len(meetings)} league meetings): " + " · ".join(
            f"{d:%b %Y} {h} {gh}-{ga} {away if h == home else home}" for d, h, gh, ga in reversed(meetings)))
    else:
        st.caption("No recent league meetings between these teams.")

    # 5) team news (Premier League only)
    if div == "E0":
        st.markdown("**Team news** (official Fantasy Premier League data)")
        if not news:
            st.caption("Team news is unavailable right now.")
        for team in (home, away):
            n = news.get(team)
            if n is None or n["players"].empty:
                st.caption(f"{team}: no injuries or suspensions reported.")
                continue
            st.markdown(f"{team}: about **{n['missing_attack']:.0%}** of the team's xG+xA is likely missing "
                        f"(expected goals × {teamnews.xg_multiplier(n):.2f})")
            st.dataframe(n["players"], hide_index=True, width="stretch")
        st.caption("The team-news adjustment can't be backtested (no free injury history exists), "
                   "so it is deliberately cautious.")
    else:
        st.caption("Injury data: only available for the Premier League (no free feed for other leagues).")


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

    eng = get_deep_engine(deep.ENGINE_VERSION)
    news = get_team_news() if div == "E0" else {}
    rows, keys, skipped = [], [], 0
    for rec in fx.to_dict("records"):  # not itertuples: it renames "Avg>2.5"
        f = pd.Series(rec)
        if not (mdl.knows(f.HomeTeam) and mdl.knows(f.AwayTeam)):
            skipped += 1
            continue
        p = mdl.predict(f.HomeTeam, f.AwayTeam)
        odds = odds_tuple(rec)
        r = deep_result(eng, div, f.HomeTeam, f.AwayTeam, f.Kickoff, odds, mdl, news)
        ph, pd_, pa = r["final"] if r else (p["home"], p["draw"], p["away"])
        value = "—"
        sig = value_signal(r, odds, min_edge) if r else None
        if sig:
            name = ["Home", "Draw", "Away"][sig[0]]
            value = f"{name} @ {odds[sig[0]]:.2f} (+{sig[1]:.0%})"
        row = {
            "Kick-off (UK)": kickoff_label(f),
            "Match": f"{f.HomeTeam} vs {f.AwayTeam}",
            "Home": ph, "Draw": pd_, "Away": pa,
            "Over 2.5": p["over_2_5"], "BTTS": p["btts"],
            "xG": f"{p['xG_home']:.2f} – {p['xG_away']:.2f}",
            "Fair odds (H/D/A)": f"{1/ph:.2f} / {1/pd_:.2f} / {1/pa:.2f}",
            "Market odds (H/D/A)": " / ".join(f"{o:.2f}" for o in odds) if odds else "—",
            "Value": value,
        }
        if div == "E0":
            row["Team news"] = "; ".join(s for s in (news_summary(news.get(f.HomeTeam)),
                                                      news_summary(news.get(f.AwayTeam))) if s) or "—"
        rows.append(row)
        keys.append((f.HomeTeam, f.AwayTeam, f.Kickoff, odds))
    if not rows:
        st.info("Fixtures found, but the teams are not in the model's history.")
        return
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, width="stretch",
        column_config={k: prob_col(k) for k in ["Home", "Draw", "Away", "Over 2.5", "BTTS"]},
    )
    n_odds = sum(k[3] is not None for k in keys)
    notes = [
        "Home/Draw/Away come from the deep model (form, shots, Elo, rest, head-to-head"
        + (", team news" if div == "E0" else "") + ", plus the bookmaker price when published). "
        "Over 2.5, BTTS and xG come from the season strength model.",
        f"Value = the deep model's edge over the market is above your threshold ({min_edge:.0%}).",
    ]
    if n_odds < len(keys):
        notes.append(f"Bookmaker odds are available for {n_odds} of {len(keys)} fixtures.")
    if skipped:
        notes.append(f"{skipped} fixture(s) skipped: team not in the model's history.")
    if unmatched:
        notes.append("Unmatched schedule names: " + ", ".join(unmatched) + ".")
    st.caption(" ".join(notes))

    st.subheader("Deep dive")
    i = st.selectbox("Match", range(len(keys)), format_func=lambda i: f"{rows[i]['Match']} · {rows[i]['Kick-off (UK)']}",
                     label_visibility="collapsed")
    home, away, kickoff, odds = keys[i]
    deep_dive(div, home, away, kickoff, odds)


def page_predictor():
    header("Predictor", "Match predictor")
    div = ss.league
    results, mdl, _ = load_league_or_stop(div)
    c1, c2 = st.columns(2)
    teams = mdl.teams
    home = c1.selectbox("Home team", teams, index=0)
    away = c2.selectbox("Away team", teams, index=min(1, len(teams) - 1))
    if home == away:
        st.info("Pick two different teams.")
        return
    st.caption("Predicts the match as if it were played now, using each team's current form.")
    deep_dive(div, home, away, pd.Timestamp.now(tz=data.UK))

    st.subheader("Goals and scorelines")
    p = mdl.predict(home, away)
    m1, m2, m3 = st.columns(3)
    m1.metric("Over 2.5 goals", pct(p["over_2_5"]))
    m2.metric("Both teams score", pct(p["btts"]))
    m3.metric("Expected goals", f"{p['xG_home']:.2f} – {p['xG_away']:.2f}")
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


def page_model_report():
    header("Analysis", "Model report", league_picker=False)
    eng = get_deep_engine(deep.ENGINE_VERSION)
    rep = eng.report
    st.markdown(
        f"The deep model was trained on matches from {deep.TRAIN_FROM}-{(deep.TRAIN_FROM + 1) % 100:02d} "
        f"to {deep.TEST_FROM - 1}-{deep.TEST_FROM % 100:02d} ({rep['train_matches']:,} matches, 18 leagues), "
        f"then tested on **{rep['test_matches']:,} later matches it had never seen**."
    )
    st.subheader("Accuracy against the bookmakers")
    st.dataframe(
        pd.DataFrame(rep["metrics"], columns=["Model", "Log-loss (lower is better)", "Accuracy"]),
        hide_index=True, width="stretch",
        column_config={"Log-loss (lower is better)": st.column_config.NumberColumn(format="%.4f"),
                       "Accuracy": st.column_config.NumberColumn(format="percent")},
    )
    st.caption("Log-loss rewards giving high probability to what actually happens; it's the "
               "standard way to compare forecasters. Closing odds are the sharpest benchmark.")
    st.subheader("If you had bet every value signal")
    vb = pd.DataFrame(rep["value_bets"], columns=["Rule", "Odds range", "Bets", "Return per bet", "± (95%)"])
    st.dataframe(vb, hide_index=True, width="stretch", column_config={
        "Return per bet": st.column_config.NumberColumn(format="percent"),
        "± (95%)": st.column_config.NumberColumn(format="percent")})
    st.caption("1 unit per bet at the best available price, using the deep model + bookmaker price. "
               "If the ± range includes zero, the result could be luck.")
    st.subheader("What this means")
    st.markdown(
        "- The deep model is clearly better than the app's older season-strength model.\n"
        "- On its own signals it is still a little less accurate than the bookmakers: they already "
        "price in form, injuries and head-to-head records.\n"
        "- Combined with the bookmaker price it matches or slightly beats the opening odds, which is "
        "what makes its value signals worth looking at. It does not reliably beat the closing odds.\n"
        "- Treat value signals as a shortlist to research, not as sure bets. Stake small."
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


def prediction_table(fx: pd.DataFrame, mdl: model.DixonColes):
    """Upcoming matches with model probabilities; flags teams with little history."""
    rows = []
    for f in fx.to_dict("records"):
        home, away = f["HomeTeam"], f["AwayTeam"]
        if not (mdl.knows(home) and mdl.knows(away)):
            continue
        p = mdl.predict(home, away)
        games = min(mdl.n_matches.get(home, 0), mdl.n_matches.get(away, 0))
        rows.append({
            "Kick-off (UK)": f["Kickoff"].strftime("%a %d %b %H:%M"),
            "Match": f"{home} vs {away}",
            "Home": p["home"], "Draw": p["draw"], "Away": p["away"],
            "Over 2.5": p["over_2_5"], "BTTS": p["btts"],
            "xG": f"{p['xG_home']:.2f} – {p['xG_away']:.2f}",
            "Likely score": p["top_scores"][0][0],
            "Fair odds (H/D/A)": f"{1/p['home']:.2f} / {1/p['draw']:.2f} / {1/p['away']:.2f}",
            "History": "⚠ limited" if games < MIN_MATCHES_FOR_PICKS else "ok",
        })
    if not rows:
        st.info("No upcoming matches with known teams.")
        return
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, width="stretch",
        column_config={
            **{k: prob_col(k) for k in ["Home", "Draw", "Away", "Over 2.5", "BTTS"]},
            "History": st.column_config.TextColumn(
                help=f"⚠ = a team has fewer than {MIN_MATCHES_FOR_PICKS} matches in the data, "
                     "so this prediction is much less reliable."),
        },
    )


def match_card(mdl: model.DixonColes, home: str, away: str, neutral: bool):
    p = mdl.predict(home, away, neutral=neutral)
    c = st.columns(5)
    c[0].metric(f"{home} win", pct(p["home"]), f"fair {1/p['home']:.2f}", delta_color="off")
    c[1].metric("Draw", pct(p["draw"]), f"fair {1/p['draw']:.2f}", delta_color="off")
    c[2].metric(f"{away} win", pct(p["away"]), f"fair {1/p['away']:.2f}", delta_color="off")
    c[3].metric("Over 2.5 goals", pct(p["over_2_5"]))
    c[4].metric("Both teams score", pct(p["btts"]))
    st.write(f"**Expected goals:** {home} {p['xG_home']:.2f} – {p['xG_away']:.2f} {away} · "
             "**Most likely scores:** " + " · ".join(f"{s} ({pr:.0%})" for s, pr in p["top_scores"][:4]))
    few = [t for t in (home, away) if mdl.n_matches.get(t, 0) < MIN_MATCHES_FOR_PICKS]
    if few:
        st.caption(f"⚠ Limited history for {', '.join(few)}: treat this prediction with extra caution.")


def page_cups():
    header("Europe & International", "European cups", league_picker=False)
    code = st.segmented_control(
        "Competition", list(competitions.CUPS), default="UCL",
        format_func=lambda c: competitions.CUPS[c][0].split("– ")[1], key="cup_code",
    ) or "UCL"
    n_seasons, half_life, _ = settings()
    try:
        mdl, mapping = get_europe_model(n_seasons, half_life)
        fx = competitions.cup_fixtures(code, mapping)
    except Exception as e:
        st.error(f"Could not load European cup data: {e}")
        return

    st.subheader("Upcoming matches")
    days = st.slider("Days ahead", 1, 60, 14, key="cup_days", help="Counted from the next kick-off.")
    prediction_table(in_window(fx, days), mdl)
    st.caption(
        "One model is fitted on all 18 domestic leagues plus two seasons of European cup results, "
        "so clubs from different countries are rated on the same scale. Clubs from leagues the app "
        "doesn't cover (e.g. Norway, Austria, Czechia) are rated from their cup games only. "
        "Bookmaker odds aren't available for cup matches."
    )

    st.subheader("Predict any tie")
    played = competitions.cup_played(code, mapping)
    teams = sorted((set(fx["HomeTeam"]) | set(fx["AwayTeam"]) | set(played["HomeTeam"])
                    | set(played["AwayTeam"])) & set(mdl.teams))
    if len(teams) >= 2:
        c1, c2, c3 = st.columns([2, 2, 1], vertical_alignment="bottom")
        home = c1.selectbox("Home team", teams, key="cup_home")
        away = c2.selectbox("Away team", teams, index=1, key="cup_away")
        neutral = c3.toggle("Neutral venue", help="For a final.", key="cup_neutral")
        if home == away:
            st.info("Pick two different teams.")
        else:
            match_card(mdl, home, away, neutral)

    with st.expander(f"Results so far this season ({len(played)})"):
        st.dataframe(
            pd.DataFrame({
                "Date": played["Kickoff"].dt.strftime("%a %d %b"), "Home": played["HomeTeam"],
                "Score": played["FTHG"].astype(int).astype(str) + " – " + played["FTAG"].astype(int).astype(str),
                "Away": played["AwayTeam"],
            }),
            hide_index=True, width="stretch",
        )


def page_international():
    header("Europe & International", "International", league_picker=False)
    try:
        mdl = get_intl_model()
    except Exception as e:
        st.error(f"Could not load international results: {e}")
        return

    st.subheader("UEFA Nations League: upcoming")
    try:
        fx = competitions.international_fixtures("UNL", set(mdl.teams))
        if fx.empty:
            st.info("No upcoming Nations League matches in the feed right now.")
        else:
            prediction_table(fx, mdl)
    except Exception as e:
        st.warning(f"Could not load Nations League fixtures: {e}")

    st.subheader("Any two national teams")
    st.caption(
        "For World Cup and Euro qualifiers, AFCON qualifiers, friendlies and other matches that "
        "have no free fixture feed: pick the two teams and the venue."
    )
    teams = sorted(t for t in mdl.teams if mdl.n_matches.get(t, 0) >= 5)
    c1, c2, c3 = st.columns([2, 2, 1], vertical_alignment="bottom")
    home = c1.selectbox("Team 1 (home)", teams, index=teams.index("England") if "England" in teams else 0,
                        key="intl_home")
    away = c2.selectbox("Team 2 (away)", teams, index=teams.index("France") if "France" in teams else 1,
                        key="intl_away")
    neutral = c3.toggle("Neutral venue", key="intl_neutral",
                        help="Tournament finals are usually at a neutral venue.")
    if home == away:
        st.info("Pick two different teams.")
    else:
        match_card(mdl, home, away, neutral)

    with st.expander("Strongest national teams (model rating)"):
        r = mdl.ratings()
        r = r[r["Matches"] >= 10].head(40).reset_index(drop=True)
        r.index += 1
        st.dataframe(r[["Team", "Overall", "Attack", "Defence", "Matches"]].round(2), width="stretch")
    st.caption(
        "Fitted on the last 8 years of full internationals (source: "
        "[international_results](https://github.com/martj42/international_results)), recent "
        f"matches weighted more, friendlies at {competitions.FRIENDLY_WEIGHT:.0%} weight, and no "
        "home advantage at neutral venues. The results file is updated by volunteers, so the most "
        "recent window can be missing for a while."
    )


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
    "cups": st.Page(page_cups, title="European cups", icon=":material/trophy:", url_path="cups"),
    "intl": st.Page(page_international, title="International", icon=":material/public:",
                    url_path="international"),
    "report": st.Page(page_model_report, title="Model report", icon=":material/fact_check:",
                      url_path="model-report"),
    "form": st.Page(page_form, title="Form", icon=":material/trending_up:", url_path="form"),
    "backtest": st.Page(page_backtest, title="Backtest", icon=":material/science:",
                        url_path="backtest"),
}
MENU = [
    ("Top picks", ":material/emoji_events:", ["top", "daily"]),
    ("Fixtures", ":material/calendar_month:", ["fixtures"]),
    ("Europe & International", ":material/language:", ["cups", "intl"]),
    ("Predictor", ":material/online_prediction:", ["predictor"]),
    ("Analysis", ":material/analytics:", ["report", "ratings", "form", "backtest"]),
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
