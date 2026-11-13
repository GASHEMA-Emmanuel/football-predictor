"""Deep match model: many pre-match signals combined by a trained classifier.

Signals (all computed from information available BEFORE kick-off):
- Elo rating that reacts fast to results
- Exponentially weighted form: shots, shots on target and points
- Rest days since each team's previous league match
- Head-to-head goal difference in recent meetings
- Points per game this season
- Optionally the bookmakers' opening odds, so the model learns where its own
  signals should move away from the market price

The classifier is a multinomial logistic regression on standardised features. It is
interpretable: every prediction can be broken down into how much each signal pushed it.
It is trained on past seasons and judged on later seasons it never saw.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import data

ALPHA = 0.15            # EWM weight per match (~ last 6-8 matches dominate)
ELO_K, ELO_HOME = 20.0, 60.0
STATS = [("gf", "FTHG", "FTAG"), ("sf", "HS", "AS"), ("stf", "HST", "AST"), ("cf", "HC", "AC")]

# Signal groups, chosen by out-of-sample testing (deep_eval.py + ablation): a season-strength
# model, corners and home/away-specific form were dropped because they added nothing or hurt.
GROUPS = {
    "Elo rating": ["elo_diff"],
    "Shots on target (form)": ["stf_h", "sta_h", "stf_a", "sta_a"],
    "Shots (form)": ["sf_h", "sa_h", "sf_a", "sa_a"],
    "Points (form)": ["pts_h", "pts_a"],
    "Rest days": ["rest_h", "rest_a"],
    "Head-to-head": ["h2h_gd", "h2h_n"],
    "Season points per game": ["ppg_h", "ppg_a"],
    "Bookmaker price": ["mkt_logit", "mkt_draw"],
}
BASE_FEATURES = [f for g, fs in GROUPS.items() if g != "Bookmaker price" for f in fs]
MARKET_FEATURES = BASE_FEATURES + GROUPS["Bookmaker price"]


class LeagueState:
    """Everything the model knows about one league at a point in time."""

    def __init__(self):
        self.form: dict[str, dict] = {}
        self.elo: dict[str, float] = {}
        self.last: dict[str, pd.Timestamp] = {}
        self.h2h: dict[frozenset, deque] = defaultdict(lambda: deque(maxlen=6))
        self.season: dict[str, list] = {}          # team -> [points, played] this season
        self.season_code: str | None = None
        self.avg = {"gf": 1.4, "sf": 12.0, "stf": 4.3, "cf": 5.0}  # league running averages

    def _team(self, t: str) -> dict:
        if t not in self.form:
            a = self.avg
            self.form[t] = {
                "gf": a["gf"], "ga": a["gf"], "sf": a["sf"], "sa": a["sf"],
                "stf": a["stf"], "sta": a["stf"], "cf": a["cf"], "ca": a["cf"], "pts": 1.35,
                "hgf": a["gf"], "hga": a["gf"], "agf": a["gf"], "aga": a["gf"], "n": 0,
            }
            # newcomers (usually promoted) start below the league's average Elo
            self.elo[t] = (np.mean(list(self.elo.values())) - 50) if self.elo else 1500.0
        return self.form[t]

    def features(self, home: str, away: str, date: pd.Timestamp) -> dict:
        h, a = self._team(home), self._team(away)
        rest = lambda t: min((date - self.last[t]).days, 21) if t in self.last else 14
        meetings = [m for m in self.h2h[frozenset((home, away))] if (date - m[0]).days < 4 * 365]
        gd = [(m[2] - m[3]) if m[1] == home else (m[3] - m[2]) for m in meetings]
        ppg = lambda t: (self.season[t][0] / self.season[t][1]) if self.season.get(t, [0, 0])[1] else 1.35
        return {
            "elo_diff": self.elo[home] - self.elo[away],
            "gf_h": h["gf"], "ga_h": h["ga"], "gf_a": a["gf"], "ga_a": a["ga"],
            "stf_h": h["stf"], "sta_h": h["sta"], "stf_a": a["stf"], "sta_a": a["sta"],
            "sf_h": h["sf"], "sa_h": h["sa"], "sf_a": a["sf"], "sa_a": a["sa"],
            "cf_h": h["cf"], "ca_h": h["ca"], "cf_a": a["cf"], "ca_a": a["ca"],
            "pts_h": h["pts"], "pts_a": a["pts"],
            "vgf_h": h["hgf"], "vga_h": h["hga"], "vgf_a": a["agf"], "vga_a": a["aga"],
            "rest_h": rest(home), "rest_a": rest(away),
            "h2h_gd": float(np.mean(gd)) if gd else 0.0, "h2h_n": len(gd),
            "ppg_h": ppg(home), "ppg_a": ppg(away),
        }

    def update(self, m) -> None:
        """Feed one finished match (a row with HomeTeam, AwayTeam, FTHG, FTAG, stats...)."""
        home, away = m["HomeTeam"], m["AwayTeam"]
        h, a = self._team(home), self._team(away)
        gh, ga = int(m["FTHG"]), int(m["FTAG"])
        if m.get("Season") != self.season_code:  # new season: reset the table
            self.season_code, self.season = m.get("Season"), {}

        def ewm(d, key, x):
            if x is not None and not (isinstance(x, float) and math.isnan(x)):
                d[key] += ALPHA * (x - d[key])

        for key, hc, ac in STATS:
            hv, av = m.get(hc), m.get(ac)
            against = "ga" if key == "gf" else key[:-1] + "a"
            ewm(h, key, hv); ewm(h, against, av)
            ewm(a, key, av); ewm(a, against, hv)
            if hv is not None and not pd.isna(hv):  # league averages drift slowly
                self.avg[key] += 0.01 * ((hv + av) / 2 - self.avg[key])
        ph, pa = (3, 0) if gh > ga else (0, 3) if gh < ga else (1, 1)
        ewm(h, "pts", ph); ewm(a, "pts", pa)
        ewm(h, "hgf", gh); ewm(h, "hga", ga); ewm(a, "agf", ga); ewm(a, "aga", gh)
        h["n"] += 1; a["n"] += 1

        # Elo with a goal-difference multiplier
        exp_h = 1 / (1 + 10 ** (-(self.elo[home] + ELO_HOME - self.elo[away]) / 400))
        score = 1.0 if gh > ga else 0.0 if gh < ga else 0.5
        k = ELO_K * (1 + math.log1p(abs(gh - ga)))
        self.elo[home] += k * (score - exp_h)
        self.elo[away] -= k * (score - exp_h)

        self.last[home] = self.last[away] = m["Date"]
        self.h2h[frozenset((home, away))].append((m["Date"], home, gh, ga))
        for t, p in ((home, ph), (away, pa)):
            s = self.season.setdefault(t, [0, 0])
            s[0] += p; s[1] += 1


def market_probs(df: pd.DataFrame, cols=("AvgH", "AvgD", "AvgA")) -> np.ndarray:
    """Bookmaker probabilities with the margin removed (NaN rows where odds are missing)."""
    odds = df[list(cols)].to_numpy(dtype=float)
    inv = 1 / odds
    return inv / inv.sum(1, keepdims=True)


def _add_derived(F: pd.DataFrame) -> pd.DataFrame:
    F = F.copy()
    if "mkt_ph" in F:
        F["mkt_logit"] = np.log(F["mkt_ph"] / F["mkt_pa"])
        F["mkt_draw"] = F["mkt_pd"]
    return F


def build_history(matches: pd.DataFrame) -> tuple[pd.DataFrame, LeagueState]:
    """Pre-match features for every match in `matches` (one league), in date order, plus the
    league state after the last match (used to predict upcoming fixtures)."""
    matches = matches.sort_values("Date").reset_index(drop=True)
    state = LeagueState()
    rows = []
    for m in matches.to_dict("records"):
        rows.append(state.features(m["HomeTeam"], m["AwayTeam"], m["Date"]))
        state.update(m)
    F = pd.DataFrame(rows)
    F[["mkt_ph", "mkt_pd", "mkt_pa"]] = market_probs(matches)
    keep = ["Div", "Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
            "AvgH", "AvgD", "AvgA", "MaxH", "MaxD", "MaxA", "AvgCH", "AvgCD", "AvgCA"]
    out = pd.concat([matches[[c for c in keep if c in matches]].reset_index(drop=True), F], axis=1)
    return _add_derived(out), state


def train(hist: pd.DataFrame, features: list[str], C: float = 0.5):
    d = hist.dropna(subset=features + ["FTR"])
    y = d["FTR"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    clf.fit(d[features].to_numpy(), y)
    return clf


def predict_proba(clf, F: pd.DataFrame, features: list[str]) -> np.ndarray:
    return clf.predict_proba(F[features].to_numpy())


def explain(clf, x: dict, features: list[str]) -> pd.DataFrame:
    """How much each signal group pushes towards home win (+) / away win (-) and draw."""
    scaler, lr = clf[0], clf[1]
    z = (np.array([x[f] for f in features]) - scaler.mean_) / scaler.scale_
    home_vs_away = (lr.coef_[0] - lr.coef_[2]) * z
    draw = (lr.coef_[1] - (lr.coef_[0] + lr.coef_[2]) / 2) * z
    rows = []
    for g, fs in GROUPS.items():
        idx = [features.index(f) for f in fs if f in features]
        if idx:
            rows.append({"Signal": g, "Home vs away": float(home_vs_away[idx].sum()),
                         "Draw": float(draw[idx].sum())})
    return pd.DataFrame(rows)


# ---------------- engine used by the app ----------------
# Bump whenever DeepEngine gains or changes attributes: the app caches the trained engine for
# hours, and a new version number forces a rebuild instead of serving a stale object.
ENGINE_VERSION = 2
TRAIN_FROM = 2019   # bookmaker average/max odds columns exist from 2019-20
TEST_FROM = 2025    # held-out seasons for the honest report
WARMUP_FROM = 2018  # one season to warm up Elo/form before training rows start


def _logloss(P, y):
    return float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)).mean())


class DeepEngine:
    """Trained deep model + the current state of every league."""

    def __init__(self, raw_by_league: dict[str, pd.DataFrame]):
        parts, self.states = [], {}
        for div, raw in raw_by_league.items():
            hist, state = build_history(raw)
            parts.append(hist)
            self.states[div] = state
        hist = pd.concat(parts, ignore_index=True)
        hist["Start"] = hist["Season"].str[:2].astype(int) + 2000
        hist = hist[hist["Start"] >= TRAIN_FROM]
        self.report = self._report(hist)
        # deployed models are trained on everything, including the test seasons
        self.own = train(hist, BASE_FEATURES)
        self.mkt = train(hist.dropna(subset=MARKET_FEATURES), MARKET_FEATURES)
        self.n_train = len(hist)
        self.past = self._past_probs(hist)

    @staticmethod
    def _past_probs(hist: pd.DataFrame) -> dict:
        """Leak-free probabilities for this season's played matches, for grading past picks:
        a classifier trained only on earlier seasons, fed each match's pre-kick-off signals.
        Key: (Div, Date, HomeTeam, AwayTeam) -> (home, draw, away)."""
        current = hist["Start"].max()
        before, now = hist[hist["Start"] < current], hist[hist["Start"] == current]
        if before.empty or now.empty:
            return {}
        own = train(before, BASE_FEATURES)
        mkt = train(before.dropna(subset=MARKET_FEATURES), MARKET_FEATURES)
        P = predict_proba(own, now, BASE_FEATURES)
        has_mkt = now[MARKET_FEATURES].notna().all(axis=1).to_numpy()
        if has_mkt.any():
            P[has_mkt] = predict_proba(mkt, now[has_mkt], MARKET_FEATURES)
        keys = zip(now["Div"], now["Date"], now["HomeTeam"], now["AwayTeam"])
        return {k: p for k, p in zip(keys, P)}

    @staticmethod
    def _report(hist: pd.DataFrame) -> dict:
        """Out-of-sample comparison with the bookmakers (train < TEST_FROM, test >= TEST_FROM)."""
        d = hist.dropna(subset=MARKET_FEATURES + ["MaxH", "FTR"])
        tr, te = d[d["Start"] < TEST_FROM], d[d["Start"] >= TEST_FROM]
        y = te["FTR"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
        P_own = predict_proba(train(tr, BASE_FEATURES), te, BASE_FEATURES)
        P_mkt = predict_proba(train(tr, MARKET_FEATURES), te, MARKET_FEATURES)
        book = market_probs(te)
        rep = {"test_matches": len(te), "train_matches": len(tr), "metrics": [
            ("Deep model (own signals)", _logloss(P_own, y), float((P_own.argmax(1) == y).mean())),
            ("Deep model + bookmaker price", _logloss(P_mkt, y), float((P_mkt.argmax(1) == y).mean())),
            ("Bookmakers, opening odds", _logloss(book, y), float((book.argmax(1) == y).mean())),
        ]}
        if te["AvgCH"].notna().all():
            close = market_probs(te, ("AvgCH", "AvgCD", "AvgCA"))
            rep["metrics"].append(("Bookmakers, closing odds", _logloss(close, y),
                                   float((close.argmax(1) == y).mean())))
        odds = te[["MaxH", "MaxD", "MaxA"]].to_numpy(dtype=float)
        win = np.eye(3)[y]
        profit_all = np.where(win == 1, odds - 1, -1.0)
        rows = []
        for thr in (0.03, 0.05, 0.10):
            bet = P_mkt * odds - 1 > thr
            p = profit_all[bet]
            if len(p):
                rows.append((f"edge > {thr:.0%}", "all odds", len(p), float(p.mean()),
                             float(1.96 * p.std() / np.sqrt(len(p)))))
        bet = P_mkt * odds - 1 > 0.05
        for lo, hi in ((1, 2), (2, 3), (3, 5), (5, 100)):
            p = profit_all[bet & (odds >= lo) & (odds < hi)]
            if len(p):
                rows.append(("edge > 5%", f"odds {lo}–{hi if hi < 100 else '+'}", len(p),
                             float(p.mean()), float(1.96 * p.std() / np.sqrt(len(p)))))
        rep["value_bets"] = rows
        return rep

    def knows(self, div: str, team: str) -> bool:
        return div in self.states and team in self.states[div].form

    def predict(self, div: str, home: str, away: str, when: pd.Timestamp,
                odds: tuple | None = None) -> dict | None:
        """Probabilities (home, draw, away) from own signals, and with the market when odds
        are given; plus the raw signals and an explanation table."""
        if not (self.knows(div, home) and self.knows(div, away)):
            return None
        state = self.states[div]
        when = pd.Timestamp(when).tz_localize(None) if pd.Timestamp(when).tzinfo else pd.Timestamp(when)
        x = state.features(home, away, when)
        out = {"signals": x, "own": predict_proba(self.own, pd.DataFrame([x]), BASE_FEATURES)[0],
               "why": explain(self.own, x, BASE_FEATURES), "market": None, "with_market": None}
        if odds is not None and all(o is not None and np.isfinite(o) and o > 1 for o in odds):
            inv = 1 / np.array(odds, dtype=float)
            mp = inv / inv.sum()
            xm = {**x, "mkt_logit": float(np.log(mp[0] / mp[2])), "mkt_draw": float(mp[1])}
            out["market"] = mp
            out["with_market"] = predict_proba(self.mkt, pd.DataFrame([xm]), MARKET_FEATURES)[0]
            out["why_market"] = explain(self.mkt, xm, MARKET_FEATURES)
        out["best"] = out["with_market"] if out["with_market"] is not None else out["own"]
        return out

    def recent(self, div: str, team: str) -> dict:
        """Readable form summary for one team."""
        s = self.states[div]
        f = s.form[team]
        return {"Elo": round(s.elo[team]), "Shots on target for": round(f["stf"], 1),
                "Shots on target against": round(f["sta"], 1), "Points per game (form)": round(f["pts"], 2),
                "Last league match": s.last.get(team)}

    def head_to_head(self, div: str, home: str, away: str) -> list:
        return list(self.states[div].h2h[frozenset((home, away))])
