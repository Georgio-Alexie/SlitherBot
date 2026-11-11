"""Build the paste-in bundles so one paste works in any browser.
  bot/slither-bot-v6.bundle.js = model/weights.json + bot/slither-bot-v6.js
  bot/slither-bot-v7.bundle.js = the above + model/v7-state.json (if present) + bot/v7-learning.js
  bot/slither-bot-v9.bundle.js = the v7 bundle + bot/v8-hunter.js
Run:  python3 tools/build-bundle.py"""
import json, pathlib
root = pathlib.Path(__file__).resolve().parent.parent
model = json.loads((root / "model/weights.json").read_text())
keep = {k: model[k] for k in ("format", "arch", "tensors", "hand", "trainInfo")}
v6 = "window.SLITHER_MODEL=" + json.dumps(keep, separators=(",", ":")) + ";\n" + (root / "bot/slither-bot-v6.js").read_text()
(root / "bot/slither-bot-v6.bundle.js").write_text(v6)
v7 = v6
st = root / "model/v7-state.json"
if st.exists():
    s = json.loads(st.read_text()); s.pop("buffer", None)
    v7 += "window.SLITHER_V7_STATE=" + json.dumps(s, separators=(",", ":")) + ";\n"
v7 += (root / "bot/v7-learning.js").read_text()
(root / "bot/slither-bot-v7.bundle.js").write_text(v7)
v8 = v7 + (root / "bot/v8-hunter.js").read_text()
(root / "bot/slither-bot-v9.bundle.js").write_text(v8)
print(f"wrote v6 bundle ({len(v6)//1024} KB), v7 bundle ({len(v7)//1024} KB, saved V7 state {'included' if st.exists() else 'not found, starts fresh'}), v8 bundle ({len(v8)//1024} KB)")
