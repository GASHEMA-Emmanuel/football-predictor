# ⚽ Football Predictor

A Streamlit app that predicts football match outcomes (1X2, over/under 2.5, both teams to score, scorelines) and flags possible value bets against bookmaker odds.

## Data source
[football-data.co.uk](https://www.football-data.co.uk) is free and needs no API key or sign-up:
- Results for the last N seasons across 18 European leagues, including market-average odds and xG for the current season
- `fixtures.csv`: the current round, with odds. It's only refreshed a couple of times a week, so the app hides matches that have already kicked off.

[openfootball/football.json](https://github.com/openfootball/football.json) supplies full-season schedules (no odds) for the Premier League, Championship, La Liga, Serie A, Bundesliga, Ligue 1, Eredivisie and Primeira Liga. The app shows those fixtures before football-data publishes the round, matching team names automatically.

## Pages
- **Most likely 10**: the single most likely outcome of every upcoming match across Europe's top 10 leagues, ranked by probability, plus the chance that all of them come in.
- **Daily pick (8–15)**: for a chosen day, the most probable accumulator (one leg per match) whose combined odds fall in your range, plus two alternatives that share no matches.
- **Ask (chat bubble, bottom-right on every page)**: a chat box for questions like "best prediction at odds 1.7" or "acca 8 to 15 for Saturday". Without an API key it runs in basic mode, which understands odds, odds ranges, markets, leagues, days and team names. With an Anthropic API key, Claude answers free-form questions by calling the app's own pick functions, so it never invents odds. To turn it on, set `ANTHROPIC_API_KEY` as an environment variable, or in `.streamlit/secrets.toml` (already gitignored; on Streamlit Cloud use the app's Secrets settings).
- **Upcoming fixtures**, **Match predictor**, **Team ratings**, **Form** and **Backtest** work on one league at a time (switch it at the top right).

## Model
A Dixon-Coles Poisson model with time decay. Each team gets an attack rating and a defence rating, plus a home-advantage term and a correction for low scorelines. Recent matches count more, controlled by the "form half-life" slider.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud (free)
1. Push this folder to a GitHub repo.
2. Go to https://share.streamlit.io, choose **New app**, and pick the repo with `app.py` as the main file.
3. Deploy. No secrets are needed.

## Honest note
Bookmaker odds are very efficient. Use the **Backtest** tab to compare the model with the market before betting. In testing, a pure-statistics model usually comes close to the bookmakers but doesn't beat them.
