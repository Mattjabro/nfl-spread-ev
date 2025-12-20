# NFL Bayesian Spread & EV Model

This project builds an end-to-end NFL betting model that predicts **point spreads**, converts them into **win probabilities**, and evaluates **expected value (EV)** against Vegas lines. The core idea is to model *score margins directly*, quantify uncertainty, and only then decide whether a bet is worth taking.

This is **not** an Elo model and **not** a classifier. It is a **Bayesian margin-of-victory model** trained on historical NFL outcomes, with explicit adjustments for time decay, home-field advantage, and quarterback strength.

---

## High-Level Overview

The pipeline works as follows:

1. Train a Bayesian model on historical score margins  
2. Infer latent team strengths with uncertainty  
3. Adjust expected margins for starting quarterbacks  
4. Optionally blend model spreads with Vegas lines  
5. Convert margins into probabilities  
6. Compute EV and Kelly-optimal bet sizes  
7. Visualize and interact with everything in a Streamlit app  

Vegas lines are **never used to train the model**. They are only used later as a benchmark and for optional blending.

---

## Core Model: How It Works

### 1. Training Data

The model is trained **only on past game score margins**:

margin = home_score - away_score

Older games are **exponentially downweighted**, so recent games matter more. Games from the current season can receive extra weight to reflect roster and scheme stability.

---

### 2. Team Strengths (Latent Ratings)

Each team has an unobserved latent strength parameter:

team_strength[team]

These are learned jointly across all teams using Bayesian inference. There are no manual updates or step-by-step rating changes like Elo.

---

### 3. Home-Field Advantage

A single global home-field advantage parameter is learned:

hfa

This represents the average point advantage of playing at home.

---

### 4. Quarterback Adjustment

Quarterbacks are handled **outside** the Bayesian team-strength model to keep the core stable.

QB values are estimated using historical Vegas spreads:
- Measure how spreads move relative to league average when a QB starts
- Shrink estimates toward zero for QBs with fewer games
- Result is a QB value in **points**

For any matchup:

QB adjustment = home_qb_value − away_qb_value

This adjustment is added directly to the predicted margin.

---

### 5. Expected Margin Formula

For a given game:

μ = hfa
team_strength[home] − team_strength[away]
QB_home − QB_away

This `μ` is the model’s predicted **true spread** from the home team’s perspective.

---

### 6. Uncertainty & Heavy Tails

NFL outcomes are noisy. To reflect this:

- The model learns a global variance parameter `σ`
- Variance is inflated slightly for large spreads (blowout risk)
- A **Student-t–style distribution** is used instead of a pure Gaussian

This produces more realistic tail behavior than a normal distribution.

---

### 7. Probability Conversion

Predicted margins are converted into probabilities:
z = (μ − VegasSpread) / σ
P(cover) = StudentT_CDF(z / temperature)

A temperature parameter is calibrated on historical data to improve probability calibration.

---

## Vegas Blending (Optional)

When model and Vegas disagree only slightly, the two can be blended:

blended_spread = α * Vegas + (1 − α) * Model

The weight `α` depends on how large the disagreement is and is learned empirically from past performance.

---

## Betting & Risk Management

### Expected Value

EV is computed using standard American odds:

EV = p * win − (1 − p) * loss

---

### Kelly Criterion

Bet sizing uses **fractional Kelly** with a hard cap:

- Fractional Kelly reduces sensitivity to model error
- A maximum bankroll percentage prevents overbetting

This keeps the strategy stable even when the model is confident.

---

## How to Run

### Install Dependencies

```bash
pip install pymc arviz streamlit nfl_data_py pandas numpy

### 1. Quarterback Valuation (Run Only When Needed)

These steps only need to be rerun when:
- A meaningful QB change occurs
- You want to refresh QB values using newly completed games
- Early in the season when QB samples are small

**Estimate raw QB values from historical spreads**
```bash
python estimate_qb_values_from_spreads.py #infers QB point value by anchoring to historical Vegas spreads, relative to league average

python build_qb_adjustments.py #produces qb_adjustments.csv, which is what the model actually consumes

2. Pull Current Vegas Lines (Run Weekly)

python get_odds_api_lines.py #saves the current Vegas spreads for the target week into results/vegas_lines_2025.csv

3.Generate Weekly Predictions (Run Weekly) 

python predict_week_blended.py #Train the Bayesian model on all past games and generate predictions for the upcoming slate
#Inside the script, set the target week explicitly EX: preds = predict_week(2025, 16)
#Results are written to: results/week16_blended_lines.csv

4. Interactive Evaluation

streamlit run src/streamlit_nfl_pipeline.py #Streamlit app to explore predictions, EV, and bet sizing
