"""Dixon-Coles (1997) Poisson model with exponential time decay.

log(lambda_home) = c + home_adv + attack[home] - defence[away]
log(mu_away)     = c +            attack[away] - defence[home]

plus the Dixon-Coles low-score correction (rho) for 0-0, 1-0, 0-1, 1-1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson


def _tau(x, y, lam, mu, rho):
    t = np.ones_like(lam, dtype=float)
    t = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + mu * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return np.clip(t, 1e-10, None)


@dataclass
class DixonColes:
    teams: list[str]
    attack: np.ndarray
    defence: np.ndarray
    intercept: float
    home_adv: float
    rho: float
    n_matches: dict

    # ---------- fitting ----------
    @classmethod
    def fit(
        cls,
        matches: pd.DataFrame,
        half_life_days: float = 180,
        ref_date: pd.Timestamp | None = None,
        l2: float = 0.01,
    ) -> "DixonColes":
        """Optional columns: `Neutral` (bool, no home advantage) and `Weight` (per-match weight,
        e.g. lower for friendlies), multiplied into the time-decay weight."""
        teams = sorted(set(matches["HomeTeam"]) | set(matches["AwayTeam"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        h = matches["HomeTeam"].map(idx).to_numpy()
        a = matches["AwayTeam"].map(idx).to_numpy()
        x = matches["FTHG"].to_numpy()
        y = matches["FTAG"].to_numpy()
        hf = 1.0 - matches["Neutral"].astype(float).to_numpy() if "Neutral" in matches else np.ones(len(x))

        ref_date = ref_date or matches["Date"].max()
        days = (ref_date - matches["Date"]).dt.days.to_numpy().clip(min=0)
        w = np.exp(-np.log(2) / half_life_days * days)
        if "Weight" in matches:
            w = w * matches["Weight"].to_numpy(dtype=float)
        const = gammaln(x + 1) + gammaln(y + 1)
        m00, m01 = (x == 0) & (y == 0), (x == 0) & (y == 1)
        m10, m11 = (x == 1) & (y == 0), (x == 1) & (y == 1)

        def unpack(p):
            return p[:n], p[n : 2 * n], p[2 * n], p[2 * n + 1], p[2 * n + 2]

        def nll_and_grad(p):
            """Negative log-likelihood and its exact gradient (needed for big joint fits)."""
            att, dfn, c, home, rho = unpack(p)
            log_lam = c + home * hf + att[h] - dfn[a]
            log_mu = c + att[a] - dfn[h]
            lam, mu = np.exp(log_lam), np.exp(log_mu)
            tau = _tau(x, y, lam, mu, rho)
            ll = (x * log_lam - lam) + (y * log_mu - mu) - const + np.log(tau)
            # d ll / d log_lam, d log_mu, d rho
            g_lam = x - lam
            g_mu = y - mu
            g_rho = np.zeros_like(lam)
            g_lam = g_lam + np.where(m00, -lam * mu * rho / tau, 0) + np.where(m01, lam * rho / tau, 0)
            g_mu = g_mu + np.where(m00, -lam * mu * rho / tau, 0) + np.where(m10, mu * rho / tau, 0)
            g_rho = (np.where(m00, -lam * mu / tau, 0) + np.where(m01, lam / tau, 0)
                     + np.where(m10, mu / tau, 0) + np.where(m11, -1 / tau, 0))
            wl, wm = w * g_lam, w * g_mu
            d_att = -(np.bincount(h, wl, n) + np.bincount(a, wm, n)) + 2 * l2 * att
            d_dfn = (np.bincount(a, wl, n) + np.bincount(h, wm, n)) + 2 * l2 * dfn
            grad = np.concatenate([d_att, d_dfn, [-(wl.sum() + wm.sum()), -(wl * hf).sum(),
                                                  -(w * g_rho).sum()]])
            value = -np.sum(w * ll) + l2 * (np.sum(att**2) + np.sum(dfn**2))
            return value, grad

        p0 = np.concatenate([np.zeros(2 * n), [0.2, 0.25, -0.05]])
        bounds = [(-3, 3)] * (2 * n) + [(-3, 3), (-1, 1), (-0.25, 0.25)]
        res = minimize(nll_and_grad, p0, jac=True, method="L-BFGS-B", bounds=bounds)
        att, dfn, c, home, rho = unpack(res.x)
        # centre ratings (absorb means into the intercept)
        c += att.mean() - dfn.mean()
        att, dfn = att - att.mean(), dfn - dfn.mean()

        counts = pd.concat([matches["HomeTeam"], matches["AwayTeam"]]).value_counts().to_dict()
        return cls(teams, att, dfn, float(c), float(home), float(rho), counts)

    # ---------- prediction ----------
    def _index(self, team: str) -> int:
        if not hasattr(self, "_idx"):
            self._idx = {t: i for i, t in enumerate(self.teams)}
        return self._idx[team]

    def knows(self, team: str) -> bool:
        return team in self.n_matches

    def expected_goals(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        i, j = self._index(home), self._index(away)
        home_adv = 0.0 if neutral else self.home_adv
        lam = np.exp(self.intercept + home_adv + self.attack[i] - self.defence[j])
        mu = np.exp(self.intercept + self.attack[j] - self.defence[i])
        return float(lam), float(mu)

    def score_matrix(self, home: str, away: str, max_goals: int = 10, neutral: bool = False) -> np.ndarray:
        lam, mu = self.expected_goals(home, away, neutral)
        return self.matrix_from_xg(lam, mu, max_goals)

    def result_probs_from_xg(self, lam: float, mu: float) -> np.ndarray:
        """(home, draw, away) for given expected goals, e.g. after a team-news adjustment."""
        m = self.matrix_from_xg(lam, mu)
        return np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])

    def matrix_from_xg(self, lam: float, mu: float, max_goals: int = 10) -> np.ndarray:
        g = np.arange(max_goals + 1)
        m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
        m[0, 0] *= 1 - lam * mu * self.rho
        m[0, 1] *= 1 + lam * self.rho
        m[1, 0] *= 1 + mu * self.rho
        m[1, 1] *= 1 - self.rho
        return m / m.sum()

    def predict(self, home: str, away: str, neutral: bool = False) -> dict:
        m = self.score_matrix(home, away, neutral=neutral)
        lam, mu = self.expected_goals(home, away, neutral)
        goals = np.add.outer(np.arange(m.shape[0]), np.arange(m.shape[1]))
        flat = [(m[i, j], f"{i}-{j}") for i in range(m.shape[0]) for j in range(m.shape[1])]
        top = sorted(flat, reverse=True)[:5]
        return {
            "xG_home": lam,
            "xG_away": mu,
            "home": float(np.tril(m, -1).sum()),
            "draw": float(np.trace(m)),
            "away": float(np.triu(m, 1).sum()),
            "over_1_5": float(m[goals > 1.5].sum()),
            "over_2_5": float(m[goals > 2.5].sum()),
            "under_2_5": float(m[goals < 2.5].sum()),
            "under_3_5": float(m[goals < 3.5].sum()),
            "btts": float(m[1:, 1:].sum()),
            "top_scores": [(s, float(p)) for p, s in top],
            "matrix": m,
        }

    def ratings(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "Team": self.teams,
                "Attack": self.attack,
                "Defence": self.defence,
                "Matches": [self.n_matches.get(t, 0) for t in self.teams],
            }
        )
        df["Overall"] = df["Attack"] + df["Defence"]
        return df.sort_values("Overall", ascending=False).reset_index(drop=True)


# ---------- "most likely" picks ----------
# market name -> (probability key(s) summed, market-average odds column or None)
MARKETS = {
    "Home win": (("home",), "AvgH"),
    "Draw": (("draw",), "AvgD"),
    "Away win": (("away",), "AvgA"),
    "Home or draw": (("home", "draw"), None),
    "Away or draw": (("away", "draw"), None),
    "Over 1.5 goals": (("over_1_5",), None),
    "Over 2.5 goals": (("over_2_5",), "Avg>2.5"),
    "Under 2.5 goals": (("under_2_5",), "Avg<2.5"),
    "Under 3.5 goals": (("under_3_5",), None),
    "Both teams score": (("btts",), None),
}
MARKET_SETS = {
    "Match result (1X2)": ["Home win", "Draw", "Away win"],
    "Result + double chance": ["Home win", "Draw", "Away win", "Home or draw", "Away or draw"],
    "All markets": list(MARKETS),
}


def best_pick(p: dict, markets: list[str], fixture=None) -> dict:
    """Most probable outcome for one match among `markets`."""
    def prob(name):
        return sum(p[k] for k in MARKETS[name][0])

    name = max(markets, key=prob)
    col = MARKETS[name][1]
    odds = fixture.get(col) if (fixture is not None and col) else None
    odds = float(odds) if odds is not None and pd.notna(odds) else np.nan
    return {"pick": name, "prob": prob(name), "market_odds": odds}


def hits_distribution(probs) -> np.ndarray:
    """P(exactly k of the picks win), k = 0..n, assuming independent matches."""
    dist = np.array([1.0])
    for q in probs:
        dist = np.convolve(dist, [1 - q, q])
    return dist


def fair_odds(p: float) -> float:
    return float("inf") if p <= 0 else 1.0 / p


def edge(p: float, odds: float) -> float:
    """Expected value per 1 unit staked."""
    if odds is None or not np.isfinite(odds) or odds <= 1:
        return np.nan
    return p * odds - 1


def backtest(
    matches: pd.DataFrame,
    test_days: int = 90,
    refit_every_days: int = 14,
    half_life_days: float = 180,
    min_edge: float = 0.05,
) -> tuple[pd.DataFrame, dict]:
    """Walk-forward test: refit periodically, predict only matches after each fit date."""
    end = matches["Date"].max()
    start = end - pd.Timedelta(days=test_days)
    rows = []
    cut = start
    while cut < end:
        nxt = cut + pd.Timedelta(days=refit_every_days)
        train = matches[matches["Date"] < cut]
        test = matches[(matches["Date"] >= cut) & (matches["Date"] < nxt)]
        if len(train) > 50 and len(test):
            mdl = DixonColes.fit(train, half_life_days=half_life_days, ref_date=cut)
            for _, r in test.iterrows():
                if not (mdl.knows(r.HomeTeam) and mdl.knows(r.AwayTeam)):
                    continue
                p = mdl.predict(r.HomeTeam, r.AwayTeam)
                rows.append(
                    {
                        "Date": r.Date, "HomeTeam": r.HomeTeam, "AwayTeam": r.AwayTeam,
                        "FTR": r.FTR, "pH": p["home"], "pD": p["draw"], "pA": p["away"],
                        "AvgH": r.get("AvgH"), "AvgD": r.get("AvgD"), "AvgA": r.get("AvgA"),
                    }
                )
        cut = nxt
    df = pd.DataFrame(rows)
    if df.empty:
        return df, {}

    outcome = df["FTR"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    P = df[["pH", "pD", "pA"]].to_numpy()
    Y = np.eye(3)[outcome]
    stats = {
        "matches": len(df),
        "accuracy": float((P.argmax(1) == outcome).mean()),
        "log_loss": float(-np.log(P[np.arange(len(df)), outcome]).mean()),
        "brier": float(((P - Y) ** 2).sum(1).mean()),
    }

    odds = df[["AvgH", "AvgD", "AvgA"]].to_numpy(dtype=float)
    has_odds = np.isfinite(odds).all(1)
    if has_odds.any():
        B = 1 / odds[has_odds]
        B = B / B.sum(1, keepdims=True)  # remove bookmaker margin
        o, Yb = outcome[has_odds], Y[has_odds]
        stats["bookie_accuracy"] = float((B.argmax(1) == o).mean())
        stats["bookie_log_loss"] = float(-np.log(B[np.arange(len(B)), o]).mean())
        stats["bookie_brier"] = float(((B - Yb) ** 2).sum(1).mean())

        ev = P[has_odds] * odds[has_odds] - 1
        bets = ev > min_edge
        profit = np.where(Yb == 1, odds[has_odds] - 1, -1.0)
        n_bets = int(bets.sum())
        stats["value_bets"] = n_bets
        stats["profit_units"] = float(profit[bets].sum())
        stats["roi"] = float(profit[bets].sum() / n_bets) if n_bets else float("nan")
    return df, stats
