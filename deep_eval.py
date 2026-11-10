"""Backtest the deep model against the bookmakers on seasons it never saw.

Run: python deep_eval.py   (downloads ~100 season files the first time; a few minutes)
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import data
import deep

FIRST_SEASON = 2017   # warm-up seasons for form/Elo/DC state
TRAIN_FROM = 2019     # Avg/Max odds columns exist from 2019-20
TEST_FROM = 2025      # 2025-26 and the current season are the out-of-sample test


def load_history(leagues=data.TOP10) -> pd.DataFrame:
    n = data.season_start_year() - FIRST_SEASON + 1
    with ThreadPoolExecutor(len(leagues)) as pool:
        raw = list(pool.map(lambda d: data.load_results(d, n), leagues))
    t = time.time()
    cache = data.CACHE_DIR / f"hist_{'_'.join(leagues)}.pkl"
    if "--cached" in sys.argv and cache.exists():
        return pd.read_pickle(cache)
    hist = pd.concat([deep.build_history(r) for r in raw], ignore_index=True)
    print(f"features for {len(hist)} matches in {time.time() - t:.0f}s", file=sys.stderr)
    hist["Start"] = hist["Season"].str[:2].astype(int) + 2000
    data.CACHE_DIR.mkdir(exist_ok=True)
    hist.to_pickle(cache)
    return hist


def logloss(P, y):
    return float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)).mean())


def main():
    leagues = list(data.LEAGUES) if "--all" in sys.argv else data.TOP10
    hist = load_history(leagues)
    hist = hist.dropna(subset=deep.MARKET_FEATURES + ["AvgCH", "MaxH", "FTR"])
    tr = hist[(hist.Start >= TRAIN_FROM) & (hist.Start < TEST_FROM)]
    te = hist[hist.Start >= TEST_FROM].copy()
    y_te = te["FTR"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    print(f"train {len(tr)} matches, test {len(te)} matches\n")

    preds = {
        "Dixon-Coles only": te[["dc_ph", "dc_pd", "dc_pa"]].to_numpy(),
        "Bookmakers (opening avg)": deep.market_probs(te),
        "Bookmakers (closing avg)": deep.market_probs(te, ("AvgCH", "AvgCD", "AvgCA")),
    }
    for C in (0.05, 0.5):
        clf = deep.train(tr, deep.BASE_FEATURES, C)
        preds[f"Deep (no market), C={C}"] = deep.predict_proba(clf, te, deep.BASE_FEATURES)
        clf = deep.train(tr, deep.MARKET_FEATURES, C)
        preds[f"Deep + market, C={C}"] = deep.predict_proba(clf, te, deep.MARKET_FEATURES)
    ytr = tr["FTR"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_leaf_nodes=15,
                                        l2_regularization=1.0, random_state=0)
    gb.fit(tr[deep.MARKET_FEATURES].to_numpy(), ytr)
    preds["Gradient boosting + market"] = gb.predict_proba(te[deep.MARKET_FEATURES].to_numpy())

    print(f"{'model':32} {'log-loss':>9} {'accuracy':>9}")
    for name, P in preds.items():
        print(f"{name:32} {logloss(P, y_te):9.4f} {(P.argmax(1) == y_te).mean():9.1%}")

    # value betting at the best available pre-match price
    odds_best = te[["MaxH", "MaxD", "MaxA"]].to_numpy(dtype=float)
    odds_avg = te[["AvgH", "AvgD", "AvgA"]].to_numpy(dtype=float)
    win = np.eye(3)[y_te]
    for name in ["Deep (no market), C=0.5", "Deep + market, C=0.5", "Gradient boosting + market"]:
        P = preds[name]
        print(f"\nValue bets: {name}")
        for label, odds in (("best price", odds_best), ("average price", odds_avg)):
            for thr in (0.03, 0.05, 0.10):
                bet = P * odds - 1 > thr
                profit = np.where(win == 1, odds - 1, -1.0)[bet]
                n = int(bet.sum())
                if n:
                    se = profit.std() / np.sqrt(n)
                    print(f"  {label:13} edge>{thr:.0%}: {n:5} bets, ROI {profit.mean():+6.1%} (±{1.96 * se:.1%})")
        bet = P * odds_best - 1 > 0.05
        if "--all" in sys.argv:
            print("  by league (best price, edge>5%):")
            for div in leagues:
                lg = bet & (te["Div"].to_numpy() == div)[:, None]
                profit = np.where(win == 1, odds_best - 1, -1.0)[lg]
                ll_m = logloss(P[te["Div"].to_numpy() == div], y_te[te["Div"].to_numpy() == div])
                ll_b = logloss(preds["Bookmakers (opening avg)"][te["Div"].to_numpy() == div], y_te[te["Div"].to_numpy() == div])
                if len(profit):
                    print(f"    {div:4} log-loss model {ll_m:.4f} vs book {ll_b:.4f} | {len(profit):4} bets, ROI {profit.mean():+6.1%} (±{1.96 * profit.std() / np.sqrt(len(profit)):.1%})")
        print("  by odds band (best price, edge>5%):")
        for lo, hi in ((1, 2), (2, 3), (3, 5), (5, 100)):
            band = bet & (odds_best >= lo) & (odds_best < hi)
            profit = np.where(win == 1, odds_best - 1, -1.0)[band]
            if len(profit):
                print(f"    odds {lo}-{hi}: {len(profit):5} bets, ROI {profit.mean():+6.1%} (±{1.96 * profit.std() / np.sqrt(len(profit)):.1%})")


if __name__ == "__main__":
    main()
