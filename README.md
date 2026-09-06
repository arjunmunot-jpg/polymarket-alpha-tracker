# polymarket-alpha-tracker

A research system that finds consistently profitable traders on [Polymarket](https://polymarket.com),
scores them, and turns their activity into tradeable signal — with a paper-trading engine and a
safety layer wrapped around it.

Built as a personal research project. It runs in **paper mode**; no capital is deployed.

---

## What it does

Prediction markets settle to a known truth, which makes them unusually clean to study: every
position eventually resolves to 0 or 1, so a trader's skill is measurable rather than inferred.

The pipeline is four stages:

1. **Discover** — enumerate wallets from public leaderboards and on-chain activity, then filter to
   those with enough *resolved* trades to say anything statistically.
2. **Score** — rank wallets on realised profit, win rate, profit/loss ratio and volume, and track
   whether an individual wallet's edge is decaying over time.
3. **Signal** — watch the surviving cohort for new positions, match them against open markets, and
   size a hypothetical position under a risk budget.
4. **Evaluate** — simulate fills with realistic slippage, log every decision, and score the strategy
   against a frozen benchmark before any change is kept.

## How it's built

The part I find most interesting isn't the signal — it's the machinery that stops the system from
fooling itself:

- **Eval-gated learning.** Strategy changes are only kept if they beat a frozen held-out metric;
  otherwise they're reverted automatically. No change ships because it looked good in-sample.
- **Adversarial review.** A separate critic process tries to refute each finding before it's trusted.
- **Halt state.** A persistent kill-switch: when a data source goes stale or an invariant breaks, the
  system stops rather than trading on bad inputs. Guards fail loudly, never silently.
- **Fill simulation.** Paper results are computed through a slippage model, so they aren't the
  frictionless fantasy that makes backtests look good.
- **Regime detection.** Signals are evaluated per market regime rather than pooled, because an edge
  that only exists in one regime isn't an edge.

## Stack

Python · SQLite · pandas/numpy · Polymarket public APIs · pytest · LLM-assisted research agents

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env    # add your own API keys
python -m polytrack.market_scanner
```

The curated wallet roster is **not** included — it's the actual research output of the project, and
it's the part worth earning rather than copying. `config.py` loads one from `data/roster.json`
(gitignored) and the selection criteria used to build it are all in that file. Run the discovery
pass to generate your own.

## What's AI-assisted

Most of this was written with Claude Code, working from my design and research direction. I set the
architecture, decided what to measure, and made the calls on what to keep and discard; the model did
a large share of the implementation. The eval-gating and adversarial-review layers exist precisely
because I wanted a check on AI-generated changes that wasn't my own judgement.

## Status

Paper-trading research project, actively developed. Not investment advice.
