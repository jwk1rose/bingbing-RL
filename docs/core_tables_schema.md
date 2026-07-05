# Core Tables Schema

`core_tables.v1` is the persistent JSONL schema for the five data-engineering tables required by the v4 spec. Each row has:

- `schema_version`: always `core_tables.v1`.
- `table`: stable table name.

Rows are written as compact JSONL. Each table can be appended independently and later loaded by `masked_team_league.data_engineering.load_table_jsonl()`.

## LoadoutTable

One row per loadout/resource variant.

Key fields:

- `loadout_id`
- `hero_id`
- `unique_equip_id`
- `unique_equip_star`
- `normal_equip_ids`
- `normal_equip_features`
- `level_features`
- `final_stats`
- `final_power`
- `standing_rank`
- `standing_bucket`
- `rarity_cost`
- `season`
- `data_version`

## SingleMatchupTable

One row per single-team attack/defense matchup.

Key fields:

- `attack_team_hash`
- `defense_team_hash`
- `sim_or_real`
- `num_games`
- `wins`
- `losses`
- `empirical_winrate`
- `confidence_lower`
- `confidence_upper`
- `mean_duration`
- `mean_margin`
- `simulator_version`
- `model_version`
- `cache_key_hash`

## PlanMatchTable

One row per full BO3/BO5 plan matchup.

Key fields:

- `attack_plan_hash`
- `defense_plan_hash`
- `defense_roster_hash`
- `format_teams`
- `win_required`
- `sim_or_real`
- `num_games`
- `round_win_rates`
- `empirical_winrate`
- `simulator_version`
- `model_version`

## ObservationTable

One row per mask observation.

Key fields:

- `observation_hash`
- `format_teams`
- `hidden_slots`
- `visible_heroes`
- `visible_unique_equip_ids`
- `visible_unique_equip_stars`
- `position_bounds`
- `domain_sizes`
- `real_frequency`
- `belief_candidate_count`
- `belief_entropy`
- `season`
- `rank_segment`

## LeagueStrategyTable

One row per attack/defense strategy in the league pool.

Key fields:

- `strategy_id`
- `strategy_type`
- `role`
- `plan_hash`
- `created_iteration`
- `sim_score`
- `real_score`
- `cluster_id`
- `resource_cost`
- `underdog_gap`
- `active`
- `retired_reason`
- `source`

## Runtime Location

`LeagueRoundRunner` writes these tables under:

```text
<round_dir>/tables/
  loadouts.jsonl
  observations.jsonl
  single_matchups.jsonl
  plan_matches.jsonl
  league_strategies.jsonl
```
