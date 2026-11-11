# Changelog

## v9.0 — 2026-11-11

- Added game-accurate turn-radius planning and checked boosted trajectories.
- Added length and growth estimates, stake-scaled risk, and per-snake temperament.
- Reworked coiling around geometry and retained target/direction.
- Replaced the large enemy model with a 31,500-weight model and duty-cycled training.
- Reduced planner cost with obstacle filtering, loot bucketing, and short-lived caches.
- Kept the bot CPU-only; the WebGPU path was not shipped.

## Earlier development

The project progressed from an imitation-trained V5 policy through V6 planning, V7 self-improvement, V8 enemy prediction and hunting, and v8.3 loot prioritization. Earlier snapshots and raw development records are intentionally not part of this repository.
