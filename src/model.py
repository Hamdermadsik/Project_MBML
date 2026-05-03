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


class DynamicFootballModel(PyroModule):
    def __init__(self, n_teams, n_time, sigma_rw_scale=0.03, log_rate_bound=3.0):
        super().__init__()
        self.n_teams = n_teams
        self.n_time = n_time
        self.sigma_rw_scale = sigma_rw_scale
        self.log_rate_bound = log_rate_bound

    def forward(self, home_idx, away_idx, time_idx, home_goals=None, away_goals=None):
        alpha_away = pyro.sample("alpha_away", dist.Normal(-0.2, 0.5))
        alpha_home = pyro.sample("alpha_home", dist.Normal(0.2, 0.5))
        sigma_init = pyro.sample("sigma_init", dist.HalfNormal(0.5))
        sigma_rw = pyro.sample("sigma_rw", dist.HalfNormal(self.sigma_rw_scale))

        device = home_idx.device
        initial_loc = torch.zeros(self.n_teams, device=device)
        initial_attack = pyro.sample(
            "attack_initial",
            dist.Normal(initial_loc, sigma_init).to_event(1),
        )
        initial_defense = pyro.sample(
            "defense_initial",
            dist.Normal(initial_loc, sigma_init).to_event(1),
        )

        if self.n_time > 1:
            innovation_loc = torch.zeros(self.n_time - 1, self.n_teams, device=device)
            attack_offsets = pyro.sample(
                "attack_offsets",
                dist.Normal(innovation_loc, 1.0).to_event(2),
            )
            defense_offsets = pyro.sample(
                "defense_offsets",
                dist.Normal(innovation_loc, 1.0).to_event(2),
            )
            attack_innovations = attack_offsets * sigma_rw
            defense_innovations = defense_offsets * sigma_rw
            later_attack = initial_attack.unsqueeze(0) + torch.cumsum(attack_innovations, dim=0)
            later_defense = initial_defense.unsqueeze(0) + torch.cumsum(defense_innovations, dim=0)
            attack = torch.cat([initial_attack.unsqueeze(0), later_attack], dim=0)
            defense = torch.cat([initial_defense.unsqueeze(0), later_defense], dim=0)
        else:
            attack = initial_attack.unsqueeze(0)
            defense = initial_defense.unsqueeze(0)

        attack = attack - attack.mean(dim=1, keepdim=True)
        defense = defense - defense.mean(dim=1, keepdim=True)
        team_quality = attack + defense
        team_quality = team_quality - team_quality.mean(dim=1, keepdim=True)
        pyro.deterministic("attack_time", attack)
        pyro.deterministic("defense_time", defense)
        pyro.deterministic("team_quality_time", team_quality)
        # Backward-compatible alias for older notebook cells.
        pyro.deterministic("team_strengths_time", team_quality)

        with pyro.plate("matches_plate", len(home_idx)):
            home_attack = attack[time_idx, home_idx]
            away_attack = attack[time_idx, away_idx]
            home_defense = defense[time_idx, home_idx]
            away_defense = defense[time_idx, away_idx]
            log_lambda_h = home_attack - away_defense + alpha_home
            log_lambda_a = away_attack - home_defense + alpha_away
            log_lambda_h = torch.clamp(
                log_lambda_h,
                min=-self.log_rate_bound,
                max=self.log_rate_bound,
            )
            log_lambda_a = torch.clamp(
                log_lambda_a,
                min=-self.log_rate_bound,
                max=self.log_rate_bound,
            )
            lambda_h = torch.exp(log_lambda_h)
            lambda_a = torch.exp(log_lambda_a)

            pyro.sample("obs_home", dist.Poisson(lambda_h), obs=home_goals)
            pyro.sample("obs_away", dist.Poisson(lambda_a), obs=away_goals)

        return lambda_h, lambda_a


class StaticAttackDefenseFormModel(PyroModule):
    def __init__(self, n_teams, n_form_features, log_rate_bound=3.0):
        super().__init__()
        self.n_teams = n_teams
        self.n_form_features = n_form_features
        self.log_rate_bound = log_rate_bound

    def forward(
        self,
        home_idx,
        away_idx,
        home_form,
        away_form,
        home_goals=None,
        away_goals=None,
    ):
        alpha_away = pyro.sample("alpha_away", dist.Normal(-0.2, 0.5))
        alpha_home = pyro.sample("alpha_home", dist.Normal(0.2, 0.5))

        with pyro.plate("teams_plate", self.n_teams):
            attack = pyro.sample("attack", dist.Normal(0.0, 0.7))
            defense = pyro.sample("defense", dist.Normal(0.0, 0.7))

        attack = attack - attack.mean()
        defense = defense - defense.mean()
        pyro.deterministic("attack_centered", attack)
        pyro.deterministic("defense_centered", defense)

        beta_form = pyro.sample(
            "beta_form",
            dist.Normal(
                torch.zeros(self.n_form_features, device=home_idx.device),
                0.5,
            ).to_event(1),
        )

        form_diff = home_form - away_form

        with pyro.plate("matches_plate", len(home_idx)):
            form_effect = (form_diff * beta_form).sum(dim=-1)
            log_lambda_h = attack[home_idx] - defense[away_idx] + alpha_home + form_effect
            log_lambda_a = attack[away_idx] - defense[home_idx] + alpha_away - form_effect
            log_lambda_h = torch.clamp(
                log_lambda_h,
                min=-self.log_rate_bound,
                max=self.log_rate_bound,
            )
            log_lambda_a = torch.clamp(
                log_lambda_a,
                min=-self.log_rate_bound,
                max=self.log_rate_bound,
            )
            lambda_h = torch.exp(log_lambda_h)
            lambda_a = torch.exp(log_lambda_a)

            pyro.sample("obs_home", dist.Poisson(lambda_h), obs=home_goals)
            pyro.sample("obs_away", dist.Poisson(lambda_a), obs=away_goals)

        return lambda_h, lambda_a
