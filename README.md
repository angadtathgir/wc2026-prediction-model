# World Cup 2026 Prediction Model

A forecasting model for the 2026 World Cup, rebuilt from scratch to test whether it has any real predictive skill — and mostly to find out where it doesn't.

## The result I didn't want

Out-of-sample across 296 World Cup matches (2014, 2018, 2022, 2026):

| Model | Log loss | RPS | Accuracy |
|---|---|---|---|
| Dixon-Coles + Elo | 0.9420 | 0.1906 | 59.5% |
| **Fitted Elo alone** | **0.9396** | 0.1910 | **59.8%** |
| Climatology (base rates) | 1.0696 | 0.2381 | 44.3% |
| Uniform (1/3 each) | 1.0986 | 0.2412 | 44.3% |

The bivariate Poisson machinery **does not beat a two-parameter Elo**. Both crush the baselines — going from a uniform guess to a fitted Elo buys 0.159 nats — but essentially all of the skill lives in the rating system, not in the scoreline layer.

Three of the original five layers were removed after ablation. In-tournament form, the layer v3 was built around, takes log loss from 0.77 to **1.49** out-of-sample once the blend weight is chosen walk-forward rather than on the test set. The referee and clutch layers cannot be backtested with available data, so they went too.

Other findings: the fitted Elo divisor is **400**, not the 600 FIFA publishes; match-importance weighting comes out essentially flat; and host advantage is **×1.77 on expected goals**, against the ×1.04 v3 assumed.

## Write-ups

- **v4 (current)** — https://angadtathgir.github.io/wc2026-prediction-model/
- **Part one (v3)** — https://angadtathgir.github.io/wc2026-prediction-model/part-one.html

## Running it

```
python3 wc2026_model_v4.py
```

Downloads its own data and reproduces every published number from scratch.

## Data

[martj42/international_results](https://github.com/martj42/international_results) — every senior men's international since 1872, 49,547 matches, including all 104 matches of the 2026 tournament. Nothing is hand-entered.

## Method

Elo fitted by maximum likelihood over the full match history, plus a bivariate Poisson scoreline model with the Dixon & Coles (1997) correction for low-score dependence. Validation is walk-forward throughout: to predict a tournament, the model may only use matches played before it began.

Dixon, M. & Coles, S. (1997), "Modelling Association Football Scores and Inefficiencies in the Football Betting Market", *Journal of the Royal Statistical Society Series C*, 46(2), 265–280.

## Known gap

No comparison against betting-market odds. The model can be shown to have **skill** — it beats climatology by 0.13 nats out-of-sample. Whether it has **edge** — whether it knows anything a closing price doesn't — is not a question this work has earned an answer to.
