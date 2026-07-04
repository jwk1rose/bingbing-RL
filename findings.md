# Findings

- The v4 spec mandates that legality is structural, not a post-filter around free network output.
- The atomic action object is `Loadout`, not a bare hero ID. It must include hero, unique equipment ID, unique equipment star, normal equipment summary/features, final power, and standing rank.
- Unique equipment uniqueness is keyed by equipment ID, not by equipment star.
- Team slot order is position-aware and must be strictly increasing by `standing_rank`.
- The minimum deliverable is data structures, legal generator, SingleTeamWinrateModel interface, complete-defense AttackOracle, simulation cache, successive halving, diversity selection, and output explanation.
- Neural networks should not be trained first. Oracle search and legal data generation come before distillation.
- The existing oracle backend already exposes the needed decoupled API: `POST /jobs`, `GET /jobs/{job_id}`, `GET /jobs/{job_id}/results`, and `GET /api/status`.
- In backend result labels, `battle_result == 0` maps to attack win and `battle_result == 1` maps to attack loss; `battle_result == 3` should be treated as a mixed/soft result when unit state is available.
- Real smoke submission should require an already-ready persistent worker pool. A stopped backend is not enough for this repository because worker lifecycle ownership stays outside the algorithm package.
- The decoded `heroes.json` `equipIds` field contains ordinary item IDs such as `875` and `422`. These must not be inferred as unique legend equipment.
- New battle requests must match the old random-lineup proto shape closely. Several heroes can fail in Lua when `_legend_equip`, `_shard`, or `_astrolabe` runtime fields are missing.
- Resource selection for league rounds should use `PeakArenaCampGroup`/`PeakArenaCampList` rather than the full decoded hero catalog.
- Old random-lineup requests assign all unique legend equips once per side, then fill remaining slots with normal legend equips. The league runner now mirrors that side-level rule.
