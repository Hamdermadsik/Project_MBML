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

- Convert team names → integer indices
- Create arrays:
  home_team[n]
  away_team[n]
  goals_home[n]
  goals_away[n]

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
