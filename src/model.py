import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule

class FootballModel(PyroModule):
    def __init__(self, n_teams):
        super().__init__()
        self.n_teams = n_teams

    def forward(self, home_idx, away_idx, home_goals=None, away_goals=None):
        # 1. Priors for global parameters
        
        alpha_away = pyro.sample("alpha_away", dist.Normal(-0.2, 1.0))
        alpha_home = pyro.sample("alpha_home", dist.Normal(0.2, 1.0))
        
        # 2. Priors for team strengths (Plate for independence)
        # We use a plate to say "each team has its own independent strength"
        with pyro.plate("teams_plate", self.n_teams):
            team_strengths = pyro.sample("team_strengths", dist.Normal(0, 1.0))

        # 3. The Match Likelihood
        # We use another plate for the number of matches (observations)
        with pyro.plate("matches_plate", len(home_idx)):
            # Calculate the log-rate (linear predictor)
            # home_idx and away_idx are tensors of team IDs
            lambda_h = torch.exp(team_strengths[home_idx]  - team_strengths[away_idx] + alpha_home)
            lambda_a = torch.exp(team_strengths[away_idx] - team_strengths[home_idx] + alpha_away)

            # 4. Observe the actual goals (or sample them if predicting)
            pyro.sample("obs_home", dist.Poisson(lambda_h), obs=home_goals)
            pyro.sample("obs_away", dist.Poisson(lambda_a), obs=away_goals)
            
        return lambda_h, lambda_a