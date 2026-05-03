# EPL Data Preparation And Intro Analysis Notebook

  ## Summary

  Create a new English-language notebook at notebooks/epl_mbml_notebook.ipynb as the start of the final deliverable,
  not as a scratch notebook. Its first iteration should cover: project framing, dataset loading, integrity checks,
  light cleaning, exploratory analysis, chronological train/test setup, and final model-ready inputs for the baseline
  Pyro model.

  The notebook should use data/epl_matches.csv as the single source of truth and ignore the outdated README.md dataset
  description. It should be written in a self-explanatory style, with short explanatory markdown before each code
  block.

  ## Key Changes

  - Structure the notebook into these sections:
      1. Introduction and project goal
      2. Dataset overview
      3. Data quality and cleaning
      4. Exploratory analysis
      5. Modeling assumptions for the baseline
      6. Train/test split
      7. Model-ready preprocessing output
      8. Short conclusion and next modeling step
  - In the data quality and cleaning section:
      - Load the CSV with parsed date.
      - Confirm expected columns, row count, season count, team count, and missing-value status.
      - Sort matches chronologically by date, then stage, then team names for stable reproducibility.
      - Keep the raw goal columns unchanged.
      - Standardize dtypes only as needed: season as string/categorical, stage as integer, date as datetime, goals as
        integer.
      - Explicitly state that “cleaning” here mainly means validation, typing, sorting, and preparing model inputs,
        since the subset is already structurally clean.
  - In the exploratory section:
      - Show dataset coverage: 8 seasons from 2008/2009 to 2015/2016, 380 matches per season, 34 unique teams overall.
      - Summarize scoring behavior relevant to the Poisson model: home goals, away goals, total goals, win/draw/loss
        rates.
      - Include a few targeted visuals only:
          - Matches per season
          - Distribution of home and away goals
          - Total goals distribution
          - Home-vs-away average goals by season
      - Keep EDA focused on informing the model, not broad football storytelling.
  - In the baseline modeling assumptions section:
      - State clearly that the first model assumes one latent strength per team across all seasons.
      - Explain the limitation: this is interpretable and simple, but ignores strength drift across seasons and
        promotions/relegations.
      - Note that this is intentional for the first MBML version.
  - In the train/test section:
      - Use 2015/2016 as the held-out season.
      - Use 2008/2009 through 2014/2015 as training data.
      - Build the baseline evaluation set only from held-out matches where both teams were seen in training.
      - Explicitly flag held-out matches involving unseen promoted teams (Bournemouth, Watford) as excluded from
        baseline evaluation, with a brief justification in markdown.
  - In the model-ready preprocessing section, define and display these notebook outputs:
      - train_df, test_df, test_seen_df, test_unseen_df
      - team_to_idx and idx_to_team, built from training teams only
      - num_teams, num_train_matches, num_test_matches

  ## Public Interfaces / Outputs

  The notebook should establish the exact data interface that later Pyro code will consume:

  - Team indexing based on training teams only
  - Integer arrays of shape (N_train,) and (N_test_seen,) for home/away team ids
  - Integer goal arrays of shape (N_train,) and (N_test_seen,)
  - A documented mapping from team name to index for later ranking interpretation

  This becomes the preprocessing contract for the later model notebook sections.

  ## Test Plan

  - Verify the CSV has exactly 3040 rows and the 7 expected columns.
  - Verify no missing values and no duplicate match rows on season, stage, date, home_team, away_team.
  - Verify each season has 380 matches and stages 1–38.
  - Verify chronological sorting is applied before any split or preparation.
  - Verify the training team index contains only teams observed before 2015/2016.
  - Verify the unseen-team filter identifies the held-out matches involving Bournemouth and Watford.
  - Verify all exported arrays have matching lengths and integer dtypes.

  ## Assumptions

  - The notebook should be written in English from the start.
  - This should be a new final-style notebook, not a continuation of the existing EPL notebooks.
  - The first iteration stops at EDA plus model-ready inputs; it does not yet implement the Pyro model.
  - The baseline remains intentionally simple: one latent strength per team overall, despite the known realism
    tradeoff.