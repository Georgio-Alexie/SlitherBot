# Slither bot

A browser-console bot for slither.io. The current release is v9.0. It combines a learned steering policy with a short-horizon planner that checks hazards, food, opponents, and the arena boundary.

## Quick start

1. Open slither.io in a desktop browser and open its developer console.
2. Paste the full contents of [`bot/slither-bot-v9.bundle.js`](bot/slither-bot-v9.bundle.js).
3. Start a single-tab bot with `V8.solo()`. Use `V8.hub()` in one tab and `V8.bot()` in additional tabs for multi-tab operation.
4. Use `V8.report()` or `__dbg` to inspect status. Run `V8.stop()` to stop play, or `V8.shutdown()` to stop and release all V8 workers in the tab.

The script depends on slither.io's live page globals and is not a browser extension. Its compatibility can change when the game changes.

## What it does

- Evaluates 56 candidate headings with a 14-step lookahead and a collision safety check.
- Uses a learned steering model together with hand-tuned food, hunting, loot, and escape scores.
- Accounts for the game's turn radius at normal and boosted speed.
- Scales risk and boost decisions with snake length, and uses enemy temperament estimates.
- Trains an enemy movement model in the browser and stores it in IndexedDB.

The v9 enemy model has architecture 104 → 128 → 128 → 12 (31,500 weights). The included `model/weights.json` is the earlier V5/V6 steering policy; the enemy model is learned in-browser.

## Repository layout

| Path | Purpose |
|---|---|
| `bot/slither-bot-v9.bundle.js` | Ready-to-paste current release bundle. |
| `bot/slither-bot-v6.js` | Base planner and policy implementation. |
| `bot/v7-learning.js` | Learning and self-improvement layer. |
| `bot/v8-hunter.js` | Enemy model, hunting, and v9 planner updates. |
| `training/recorder-and-trainer.js` | Browser-side V5 data recorder and trainer. |
| `model/weights.json` | Base steering policy weights. |
| `tools/build-bundle.py` | Rebuilds the paste-in bundles from source and weights. |
| `tools/test-offline.mjs` | Offline test harness. |

## Build and verify

Requires Python 3 and Node.js.

```bash
python3 tools/build-bundle.py
node tools/test-offline.mjs
```

The build creates separate V6 and V7 bundles and the current V9 bundle. The training recorder is independent of the v9 enemy-model training loop. Training code expects to run in the game page console.

## Data and privacy

The private raw export, browser state, transcripts, personal memory notes, gameplay dataset, and historical backup snapshot are not included. The bot trains its v9 enemy model from browser gameplay and saves model state locally.

## License

No license is included. Choose a license before making the repository public; without one, standard copyright restrictions apply.
