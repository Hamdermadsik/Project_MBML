# MBML_SOCCER_PROJECT_CONTEXT.md

## Course Context
DTU Course: 42186 Model-Based Machine Learning

---

## Project Requirements (Assignment)

### Core Requirements
- Define a probabilistic graphical model (PGM)
- Specify a generative process
- Implement in Pyro
- Perform inference (SVI or MCMC)
- Analyze and interpret results
- Keep model interpretable
- Start simple, then extend

### Deliverables (Deadline: 15 May)
- Fully self-explanatory notebook (like course notebooks)
- 6-page report (IEEE double-column, research paper style)

### Recommended Approach
- Start with a simple model → extend gradually
- Compare with baseline models if possible
- Try multiple inference methods:
  - Variational Inference (SVI)
  - MCMC (optional extension)
- Validate model using simulated (synthetic) data
- Ensure inference recovers true parameters (sanity check)

### Common Pitfalls to Avoid
- Do NOT put priors on observed variables without parents
- Avoid observed variables that block information flow in the PGM
- Be careful with discrete latent variables (Pyro complexity)
- Think about *how* dependencies are modeled (not just structure)
- Prior choice matters → may require experimentation

---

## Problem

Latent variable:
s_t = team strength

Goal:
Infer team strengths from match data and explain goals.

---

## Data

### Full Dataset (Context)
- ~25,000 matches
- ~10,000 players
- 11 European leagues
- Seasons: 2008–2016
- Includes:
  - Match results
  - Team & player attributes (FIFA)
  - Lineups and formations (X,Y coordinates)
  - Betting odds (up to 10 providers)
  - Detailed match events (goals, fouls, cards, etc.)

### Data Source Notes
- Aggregated from multiple sources:
  - football-data (matches, events)
  - football-data.co.uk (betting odds)
  - sofifa (player/team attributes)
- Some missing values (e.g. player attributes)
- Provided as a relational SQLite database (~313 MB)

---

## Subset Used in This Project

We restrict to:
- English Premier League (EPL)
- Stored in: data/epl_matches.csv

Verified dataset facts for `data/epl_matches.csv`:
- 3040 rows
- 7 columns
- 8 seasons: `2008/2009` to `2015/2016`
- 380 matches per season
- 34 unique teams overall
- No missing values
- No duplicate matches on: `season`, `stage`, `date`, `home_team`, `away_team`
- Stages run from 1 to 38 in every season

### Data Format (CSV)

Columns:
- season
- stage
- date
- home_team
- away_team
- home_team_goal
- away_team_goal

---

## Data Preprocessing

Current preprocessing contract for the baseline notebook:
- Parse `date` as datetime
- Standardize dtypes:
  - `season` as string
  - `stage` as integer
  - `home_team_goal` and `away_team_goal` as integer
- Sort matches chronologically by:
  - `date`
  - `stage`
  - `home_team`
  - `away_team`
- Keep raw goal columns unchanged
- Define a mid-season forecasting split:
  - `TEST_SEASON = "2015/2016"`
  - `FIRST_HALF_MAX_STAGE = 19`
  - `historical_df` = all seasons except `2015/2016`
  - `first_half_2015_df` = `2015/2016` stages `1-19`
  - `second_half_2015_df` = `2015/2016` stages `20-38`
  - `train_df` = `historical_df` + `first_half_2015_df`
  - `test_df` = `second_half_2015_df`
- Convert team names → integer indices using training teams only
- Create arrays:
  - `home_team[n]`
  - `away_team[n]`
  - `goals_home[n]`
  - `goals_away[n]`

Important:
- The team index mapping is built from training teams only
- The first-half `2015/2016` matches are included intentionally so every second-half forecast team is covered by the training index

---

## Current Notebook Deliverable

Implemented notebook:
- `notebooks/epl_mbml_notebook.ipynb`

Purpose of the notebook:
- First final-style English notebook for the project
- Self-explanatory and structured as a deliverable, not a scratch notebook
- Stops before Pyro implementation
- Covers:
  - project framing
  - dataset loading
  - integrity checks
  - light cleaning
  - focused exploratory analysis
  - baseline modeling assumptions
  - chronological mid-season train/test split
  - model-ready preprocessing outputs

Notebook sections:
1. Introduction and project goal
2. Dataset overview
3. Data quality and cleaning
4. Exploratory analysis
5. Modeling assumptions for the baseline
6. Train/test split
7. Model-ready preprocessing output
8. Short conclusion and next modeling step

---

## Baseline Split Used In Notebook

Chronological split:
- Historical training seasons: `2008/2009` through `2014/2015`
- Additional training window: `2015/2016` stages `1-19`
- Held-out forecast window: `2015/2016` stages `20-38`

Verified split sizes:
- `historical_df`: 2660 matches
- `first_half_2015_df`: 190 matches
- `second_half_2015_df`: 190 matches
- `train_df`: 2850 matches
- `test_df`: 190 matches

Verified season-stage facts for `2015/2016`:
- 10 matches per stage
- 190 matches in stages `1-19`
- 190 matches in stages `20-38`

Baseline evaluation rule:
- The notebook is now framed as a mid-season forecast of the second half of `2015/2016`
- No teams are excluded from evaluation
- All 20 teams in the forecast window are already present in the training index because the first half of `2015/2016` is included in training

Training-team count for the baseline index:
- 34 teams

---

## Model-Ready Outputs Already Defined

The notebook defines these objects for later Pyro code:
- `historical_df`
- `first_half_2015_df`
- `second_half_2015_df`
- `train_df`
- `test_df`
- `team_to_idx`
- `idx_to_team`
- `home_team_train`
- `away_team_train`
- `goals_home_train`
- `goals_away_train`
- `home_team_test`
- `away_team_test`
- `goals_home_test`
- `goals_away_test`
- `num_teams`
- `num_train_matches`
- `num_test_matches`

Array interface:
- Team id arrays have shape `(2850,)` for training and `(190,)` for the second-half forecast set
- Goal arrays have shape `(2850,)` for training and `(190,)` for the second-half forecast set
- Exported arrays use integer dtype
- `team_to_idx` covers every team appearing in the second half of `2015/2016`
- Team names can be recovered through `idx_to_team` for posterior ranking interpretation

---

## Focused Exploratory Findings Already Established

Model-relevant summary from the notebook:
- Mean home goals: `1.551`
- Mean away goals: `1.160`
- Mean total goals: `2.711`
- Home win rate: `0.457`
- Draw rate: `0.258`
- Away win rate: `0.285`

Visuals already included in the notebook:
- Matches per season
- Distribution of home and away goals
- Distribution of total goals
- Average home-vs-away goals by season

Interpretation used in the notebook:
- The first model intentionally assumes one latent strength per team across all seasons
- First-half `2015/2016` data is included intentionally to support a second-half forecast
- This improves team coverage, but it is no longer a pure unseen-season experiment
- The model remains simple and interpretable, but ignores temporal strength drift and promotion/relegation effects

---

## Model

Type:
Bayesian hierarchical latent variable model  
Poisson likelihood

---

## Nodes

Global:
μ_s, σ_s, α_home, α_away

Latent:
s_t

Observed:
y_home,n, y_away,n

---

## Edges

μ_s → s_t  
σ_s → s_t  

s_home → y_home  
s_away → y_home  

s_home → y_away  
s_away → y_away  

α_home → y_home  
α_away → y_away  

---

## Plates

Teams:
t = 1,...,T → s_t  

Matches:
n = 1,...,N → y_home,n, y_away,n  

---

## Generative Process

μ_s ~ Normal(0,1)  
σ_s ~ HalfNormal(1)  

α_home ~ Normal(0,1)  
α_away ~ Normal(0,1)  

For each team:  
s_t ~ Normal(μ_s, σ_s)  

For each match:  
λ_home = exp(α_home + s_home - s_away)  
λ_away = exp(α_away + s_away - s_home)  

y_home ~ Poisson(λ_home)  
y_away ~ Poisson(λ_away)  

---

## Inference

Use Pyro:
- SVI
- AutoNormal
- Adam
- ELBO

(Optional extension: MCMC)

---

## Outputs

- team strengths (s_t)
- ranking
- home advantage
- predictions

---

## Validation

- Generate synthetic data via ancestral sampling
- Run inference on synthetic data
- Check if model recovers true parameters

---

## Key Idea

Infer hidden structure (team strength) to explain observed goals.
