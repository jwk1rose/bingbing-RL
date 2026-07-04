# Findings

- The v4 spec mandates that legality is structural, not a post-filter around free network output.
- The atomic action object is `Loadout`, not a bare hero ID. It must include hero, unique equipment ID, unique equipment star, normal equipment summary/features, final power, and standing rank.
- Unique equipment uniqueness is keyed by equipment ID, not by equipment star.
- Team slot order is position-aware and must be strictly increasing by `standing_rank`.
- The minimum deliverable is data structures, legal generator, SingleTeamWinrateModel interface, complete-defense AttackOracle, simulation cache, successive halving, diversity selection, and output explanation.
- Neural networks should not be trained first. Oracle search and legal data generation come before distillation.
