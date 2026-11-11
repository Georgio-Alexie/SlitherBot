// Offline check, no browser: runs the bot, the trainer and the V7 learning loop in Node against a fake slither.io page.
// Usage: node tools/test-offline.mjs
import fs from 'node:fs'; import path from 'node:path'; import vm from 'node:vm'; import { fileURLToPath } from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const ok = (c, m) => { if (!c) { console.error('FAIL', m); process.exit(1); } console.log('ok  ', m); };
const sleep = ms => new Promise(r => setTimeout(r, ms));
const data = JSON.parse(fs.readFileSync(path.join(root, 'data/dataset.json'), 'utf8'));
function page() {
  let clock = 1.7e12; const store = { __v5data: data.dataset.b64, __v5n: String(data.dataset.n) };
  class FakeDate extends Date { constructor(...a) { super(...(a.length ? a : [clock])); } static now() { return clock; } }
  const g = { Math, JSON, Date: FakeDate, Float32Array, Uint8Array, Map, Set, Array, Object, String, Number, Error, Promise, setTimeout, clearTimeout,
    atob: s => Buffer.from(s, 'base64').toString('latin1'), btoa: s => Buffer.from(s, 'latin1').toString('base64'),
    setInterval: () => 0, clearInterval: () => {}, console, Blob: class {}, URL: { createObjectURL: () => '' },
    localStorage: new Proxy(store, { get: (t, k) => k === 'getItem' ? (x => t[x] ?? null) : k === 'setItem' ? ((x, v) => { t[x] = String(v); }) : t[k], set: (t, k, v) => { t[k] = String(v); return true; } }),
    document: { body: { innerText: 'Your length: 500', appendChild() {} }, querySelector: () => null, getElementById: () => null, createElement: () => ({ click() {}, remove() {} }) },
    grd: 32550, flux_grd: 13000, real_flux_grd: 13000, fpsls: Array.from({ length: 200 }, (_, i) => i * 0.5), fmlts: Array(200).fill(1),
    setAcceleration: v => { g.__acc = v; } };
  g.window = g; vm.createContext(g);
  g.tick = ms => { clock += ms; }; g.store = store;
  g.run = f => vm.runInContext(fs.readFileSync(path.join(root, f), 'utf8'), g, { filename: f });
  return g;
}
const world = g => {
  const me = { xx: 32550, yy: 32550, ang: 0, sp: 5.78, sc: 1.2, sct: 40, fam: 0.5, kill_count: 0, pts: [] };
  const wall = { xx: 32700, yy: 32300, ang: Math.PI / 2, sp: 5.78, sc: 2, pts: Array.from({ length: 30 }, (_, i) => ({ xx: 32800, yy: 32300 + i * 20 })) };
  g.slither = me; g.slithers = [me, wall];
  g.foods = Array.from({ length: 40 }, (_, i) => ({ xx: 32550 + 200 + (i % 8) * 15, yy: 32550 - 250 + Math.floor(i / 8) * 15, sz: i === 0 ? 14 : 5 }));
  return me;
};

// ---- V6.1 bundle ----
{ const g = page(); g.run('bot/slither-bot-v6.bundle.js'); g.run('training/recorder-and-trainer.js');
  ok(g.V5.src === 'SLITHER_MODEL', 'v6: weights load from the embedded model');
  ok(Object.values(g.V5.P).reduce((a, v) => a + v.length, 0) === 4204, 'v6: 4,204 parameters');
  world(g); g.__mode = 'bot'; g.__botStep6();
  ok(g.__v6err === undefined && typeof g.xm === 'number', `v6: bot steps without error (mode ${g.__dbg.mode})`);
  ok(Math.cos(Math.atan2(g.ym, g.xm)) < 0.95, 'v6: steers off the line into the body wall');
  g.foods = [{ xx: 32550 + 500, yy: 32550 + 400, sz: 12 }, ...Array.from({ length: 30 }, (_, i) => ({ xx: 31500 + i * 20, yy: 31700, sz: 5 }))];
  g.slithers = [g.slither]; g.__botStep6();
  ok(g.__dbg.mode === 'orb' && g.__dbg.orb === 12, `v6.1: targets a lone big orb (mode ${g.__dbg.mode}, orb ${g.__dbg.orb})`);
  ok(g.V5loadData(data) === 3073, 'trainer: 3,073 recorded samples restore');
}
// ---- V7 learning loop ----
{ const g = page(); g.run('bot/slither-bot-v6.js'); g.store.__v5weights = undefined;
  g.SLITHER_MODEL = JSON.parse(fs.readFileSync(path.join(root, 'model/weights.json'), 'utf8')); g.run('bot/slither-bot-v6.js'); g.run('bot/v7-learning.js');
  ok(g.V7.nParams() === 11159, 'v7: 11,159 parameters');
  for (let i = 0; i < 600 && !g.V7.policyReady; i++) await sleep(50);
  ok(g.V7.policyReady, `v7: steering head pre-trained on your play (loss ${g.V7.pretrainLoss})`);
  g.__mode = 'bot';
  const cfg0 = { ...g.V6.cfg }, mu = [];
  for (let life = 1; life <= 3; life++) {
    const me = world(g);
    for (let t = 0; t < 300; t++) { g.tick(40); g.__botStep6(); const a = Math.atan2(g.ym, g.xm); me.ang = a; me.xx += Math.cos(a) * 6; me.yy += Math.sin(a) * 6; me.fam += 0.002 * life; }
    if (life === 1) ok(JSON.stringify(g.V6.cfg) !== JSON.stringify(cfg0), 'v7: ES tries a mutated set of hand weights each life');
    g.slither = null; g.tick(40); g.__botStep6();
    for (let i = 0; i < 400 && g.V7.busy; i++) await sleep(50);
    mu.push(JSON.stringify(g.V7.es.mu));
  }
  const L = g.V7.log;
  ok(L.length === 3 && L.every(r => r.samples > 50), `v7: 3 lives logged (${L.map(r => r.samples).join(', ')} samples)`);
  ok(L[0].learn && L[0].learn.deathSamples > 0, `v7: death window labelled and learned from (${L[0].learn.deathSamples} danger samples, ${L[0].learn.ms} ms)`);
  ok(g.V7.es.gen >= 1 && mu[1] !== mu[0], `v7: ES mean moves after a population of lives (generation ${g.V7.es.gen})`);
  ok(g.V7.deathOn() && typeof g.__dbg.risk === 'number', `v7: death-risk head active in the planner after 2 deaths (risk ${g.__dbg.risk})`);
  const st = JSON.parse(g.store.__v7state); ok(st.log.length === 3 && st.P.W1, 'v7: state saved to localStorage');
}
// ---- V8: 1M-weight behaviour model, hunting, minimap ----
{ const g = page(); g.run('bot/slither-bot-v9.bundle.js'); g.__mode = 'bot';
  for (let i = 0; i < 600 && g.V7.busy; i++) await sleep(50);
  ok(g.__botStep6.toString().includes('V8.kill(') && g.__botStep6.toString().includes('V8.mapGoal('), 'v8: planner patched (predicted-head obstacles, kill score, minimap goal)');
  const me = { id: 1, xx: 32550, yy: 32550, ang: 0, sp: 5.78, sc: 1.4, sct: 60, fam: 0.5, kill_count: 0, pts: [] };
  const en = Array.from({ length: 10 }, (_, i) => ({ id: 100 + i, xx: 32550 + Math.cos(i * 0.63) * (400 + i * 80), yy: 32550 + Math.sin(i * 0.63) * (400 + i * 80),
    ang: i, sp: 5.78, sc: 1 + (i % 3) * 0.4, pts: [], circ: i < 5, w: i < 5 ? (i % 2 ? 1 : -1) * (1.6 + i * 0.4) : 0, v: 185 }));
  g.slither = me; g.slithers = [me, ...en]; g.foods = Array.from({ length: 60 }, (_, i) => ({ xx: 31800 + (i * 137) % 1500, yy: 31800 + (i * 211) % 1500, sz: 5 }));
  const stepWorld = dt => { for (const e of en) { if (!e.circ) e.w = Math.max(-0.7, Math.min(0.7, e.w + (Math.random() - 0.5) * 0.35));
    e.ang += e.w * dt; e.xx += Math.cos(e.ang) * e.v * dt; e.yy += Math.sin(e.ang) * e.v * dt; e.pts.push({ xx: e.xx, yy: e.yy }); if (e.pts.length > 30) e.pts.shift();
    if (!e.circ && Math.hypot(e.xx - me.xx, e.yy - me.yy) > 1400) e.ang = Math.atan2(me.yy - e.yy, me.xx - e.xx) + (Math.random() - 0.5); } };
  await g.V8.solo();
  for (let n = 0; n < 1000; n++) { stepWorld(0.12); g.tick(120); g.V8.sample(); }
  ok([100, 101, 102, 103, 104].every(id => g.V8.circ.has(id)) && ![105, 106, 107, 108, 109].some(id => g.V8.circ.has(id)), `v8: spots the 5 snakes going round in circles, and only those (${[...g.V8.circ].join(', ')})`);
  let st = null; const t0 = Date.now();
  while (Date.now() - t0 < 60000) { await sleep(6000); g.V8.w.postMessage({ t: 'get' }); await sleep(1500); st = g.V8.stats;
    if (st && st.steps >= 150 && st.err096 < 0.8 * st.base096) break; }
  ok(st && st.params === 31500, `v8: behaviour model has ${st && st.params} weights`);
  ok(st.err096 < st.base096, `v8: after ${st.steps} training steps (${Math.round((Date.now() - t0) / 1000)} s) it predicts 0.96 s ahead better than "keeps going the same way": ${st.err096} vs ${st.base096} units (1.44 s: ${st.err144} vs ${st.base144})`);
  g.V8.w.postMessage({ t: 'stats' }); await sleep(400);
  ok(g.V8.stats.mode !== 'full' && g.V8.stats.rateMs >= 50, `v9: trainer throttles once trusted (mode ${g.V8.stats.mode}, one 16-sample step every ${g.V8.stats.rateMs} ms)`);
  g.V8.w.postMessage({ t: 'pause', on: true }); await sleep(700); g.V8.w.postMessage({ t: 'stats' }); await sleep(300); const s1 = g.V8.stats.steps; await sleep(800); g.V8.w.postMessage({ t: 'stats' }); await sleep(300);
  ok(g.V8.stats.paused && g.V8.stats.rateMs === null && g.V8.stats.steps === s1, `v9: pause message stops training (steps stay at ${s1})`);
  g.V8.w.postMessage({ t: 'pause', on: false }); await sleep(300);
  // a cut-off: a smaller snake 100 units to our left and 150 behind; turning across its path puts our body in front of its head
  const prey = { id: 200, xx: 32400, yy: 32450, ang: 0, sp: 5.78, sc: 1.0, pts: Array.from({ length: 20 }, (_, j) => ({ xx: 32400 - 20 * (j + 1), yy: 32450 })) };
  Object.assign(me, { xx: 32550, yy: 32550, ang: 0 }); g.slithers = [me, prey]; g.foods = []; g.mmdata = undefined;
  g.V8.pred.clear(); g.V8.trusted = true; g.V8.err.set(200, 30); g.tick(40);
  const Y = new Float32Array(12); g.V8.HZ.forEach((h, k) => { Y[2 * k] = 185 * h * 0.12 / 200; });
  g.V8.pred.set(200, { t: g.Date.now(), x: prey.xx, y: prey.yy, a: 0, Y }); g.__botStep6();
  ok(g.__v6err === undefined && g.V8.tickMaxKill > 0, `v8: finds the cut-off (best kill score ${g.V8.tickMaxKill.toFixed(2)}; chosen heading kill ${g.__dbg.kill}, boost '${g.__dbg.why}')`);
  // minimap: nothing nearby, the west half of the map is crowded, so it travels east
  g.slithers = [me]; g.V8.pred.clear(); g.mmdata = new Uint8Array(136 * 136); for (let y = 0; y < 136; y++) for (let x = 0; x < 68; x++) g.mmdata[y * 136 + x] = 1;
  g.tick(1500); g.__botStep6();
  ok(g.__dbg.mode === 'travel' && Math.cos(g.V8._mg.a) > 0, `v8: minimap sends it toward the empty side (mode ${g.__dbg.mode}, heading ${g.V8._mg.a && g.V8._mg.a.toFixed(2)} rad)`);
  // training pauses when we have had no snake for more than 5 s, and resumes when one appears (a hidden tab that plays is not a reason)
  g.slither = null; g.V8.sample(); g.tick(3000); g.V8.sample(); ok(!g.V8.paused, 'v9: no pause after 3 s without a snake');
  g.tick(3000); g.V8.sample(); await sleep(300); g.V8.w.postMessage({ t: 'stats' }); await sleep(300); ok(g.V8.paused && g.V8.stats.paused, 'v9: training paused after 6 s without a snake');
  g.slither = me; g.V8.sample(); await sleep(300); g.V8.w.postMessage({ t: 'stats' }); await sleep(300); ok(!g.V8.paused && !g.V8.stats.paused, 'v9: training resumes when the snake is back');
}
// ---- V8.3: loot ----
{ const g = page(); g.run('bot/slither-bot-v9.bundle.js'); g.__mode = 'bot';
  const me = { id: 1, xx: 32550, yy: 32550, ang: 0, sp: 5.78, sc: 1.4, sct: 60, fam: 0.5, kill_count: 0, pts: [] };
  // a smaller snake to hunt ahead-right, plain food ahead, and a dead snake's trail of big orbs behind-left
  const prey = { id: 7, xx: 32950, yy: 32700, ang: 0, sp: 5.78, sc: 0.8, pts: Array.from({ length: 15 }, (_, j) => ({ xx: 32950 - 20 * (j + 1), yy: 32700 })) };
  const trail = Array.from({ length: 18 }, (_, i) => ({ xx: 32100 - i * 25, yy: 32200 - i * 12, sz: 12 }));
  g.slither = me; g.slithers = [me, prey]; g.foods = [...Array.from({ length: 30 }, (_, i) => ({ xx: 32800 + (i % 6) * 15, yy: 32500 + Math.floor(i / 6) * 15, sz: 5 })), ...trail];
  g.tick(1500); g.__botStep6();
  ok(g.__v6err === undefined && g.__dbg.mode === 'loot' && g.__dbg.loot > 150 && Math.cos(Math.atan2(g.ym, g.xm) - Math.atan2(32200 - 32550, 32100 - 32550)) > 0.5,
    `v8.3: goes back for a dead snake's orbs over hunting and plain food (mode ${g.__dbg.mode}, loot ${g.__dbg.loot})`);
  g.foods = g.foods.slice(0, 30).concat(trail.slice(0, 3)); g.__botStep6();
  ok(g.__dbg.mode !== 'loot', `v8.3: ignores a small scatter normally (mode ${g.__dbg.mode})`);
  me.kill_count = 1; g.__botStep6();
  ok(g.__dbg.mode === 'loot', `v8.3: but right after a kill it collects even a small trail (loot ${g.__dbg.loot})`);
}
// ================= v9 =================
const near = (a, b, tol, m) => ok(Math.abs(a / b - 1) <= tol, `${m} (${a.toFixed(1)} vs ${b})`);
const fresh = () => { const g = page(); g.run('bot/slither-bot-v9.bundle.js'); g.__mode = 'bot'; g.V7.applyCfg = () => {}; return g; };   // no random ES mutation, so scenarios are deterministic
const mkMe = (o = {}) => Object.assign({ id: 1, xx: 32550, yy: 32550, ang: 0, sp: 5.78, sc: 1.4, sct: 60, fam: 0.5, kill_count: 0, pts: [] }, o);
const body = (n, gap = 40) => Array.from({ length: n }, (_, i) => ({ xx: 32550 - gap * (i + 1), yy: 32550 }));
// ---- 1. turn radius ----
{ const g = fresh(), V6 = g.V6;
  near(V6.turnRadius(1, 5.79), 44, 0.03, 'v9: turn radius sc1 normal ~ 44'); near(V6.turnRadius(1, 14), 106, 0.03, 'v9: turn radius sc1 boosting ~ 106');
  near(V6.turnRadius(6, 5.78), 284, 0.03, 'v9: turn radius sc6 at 5.78 ~ 284'); near(V6.turnRadius(6, 14), 690, 0.03, 'v9: turn radius sc6 boosting ~ 690');
  near(V6.turnRadius(6, 7.79), 382, 0.03, 'v9: turn radius sc6 at its real normal speed 7.79 = 382 (the brief listed 284, which is the 5.78 figure)');
  ok(V6.turnRadius(2, 6, { scang: 0.5, sp: 6, spang: 1 }) === (6 / 4) / (0.033 * 0.5), 'v9: uses the snake own scang/spang when present');
  // rollout never turns tighter than the radius; the planner uses the speed it will have
  for (const [sc, sp] of [[1.2, 5.87], [3, 6.59], [6, 14]]) { const me = mkMe({ sc, sp, sct: 100 }); g.slither = me; g.slithers = [me]; g.foods = Array.from({ length: 6 }, (_, i) => ({ xx: 32550 + 12 * i, yy: 32550 + 500, sz: 12 })); g.__botStep6();
    const P = g.V6.lastPlan, lim = 50 / g.V6.turnRadius(sc, sp >= 9 ? 14 : sp, me); let worst = 0, prev = me.ang; for (const h of P.hs) { worst = Math.max(worst, Math.abs(h - prev)); prev = h; }
    ok(g.__v6err === undefined && Math.abs(P.maxT - Math.min(2, lim)) < 1e-9 && worst <= P.maxT + 1e-9 && worst > 0.01, `v9: sc ${sc} sp ${sp}: planned turn per 50-unit step ${worst.toFixed(3)} <= radius limit ${P.maxT.toFixed(3)} (r ${P.trN.toFixed(0)})`); g.V6._pb = false; }
  // fit self-check records, never steers
  const me = mkMe(); g.slither = me; g.slithers = [me]; g.foods = []; for (let i = 0; i < 30; i++) { g.tick(40); me.ang += 0.05; me.xx += Math.cos(me.ang) * 7; me.yy += Math.sin(me.ang) * 7; g.__botStep6(); }
  ok(g.V6.turnFit.n > 20 && g.V6.turnFit.measuredR > 0 && g.V6.turnFit.formulaR > 0 && g.__dbg.fit === g.V6.turnFit.ratio, `v9: turnFit reports measured ${g.V6.turnFit.measuredR} vs formula ${g.V6.turnFit.formulaR} units (ratio ${g.V6.turnFit.ratio})`);
}
// ---- boost refused when the boost-radius rollout is blocked ----
{ const g = fresh(), V6 = g.V6; V6.cfg.boostK = 0.05; V6.cfg.stakeL = 1e9;
  const setup = obstacle => { const me = mkMe({ sc: 4, sp: 6.99, sct: 320 }); g.V6._pb = false; g.V6.coil = null;
    const lootAng = 1.5, lx = 32550 + Math.cos(lootAng) * 600, ly = 32550 + Math.sin(lootAng) * 600;
    const sn = [me, { id: 9, xx: lx + 350, yy: ly, ang: 0, sp: 5.78, sc: 0.5, pts: [] }];   // a small snake near the loot makes it contested ("rival")
    if (obstacle) sn.push({ id: 8, xx: 40000, yy: 40000, ang: 0, sp: 5.78, sc: 0.5, pts: [obstacle] });
    g.slither = me; g.slithers = sn; g.foods = Array.from({ length: 10 }, (_, i) => ({ xx: lx + (i % 5) * 12, yy: ly + Math.floor(i / 5) * 12, sz: 12 })); g.tick(500); g.__botStep6(); return me; };
  setup(null); const d0 = g.__dbg, a0 = Math.atan2(g.ym, g.xm);
  ok(d0.mode === 'loot' && d0.why === 'loot' && d0.bchk === true, `v9: clear path -> boosts for contested loot (why ${d0.why}, boost-radius check ${d0.bchk})`);
  const trB = V6.turnRadius(4, 14), mT = 50 / trB; let x = 32550, y = 32550, h = 0, da = a0, pt = null; for (let k = 1; k <= 10; k++) { const t = Math.max(-mT, Math.min(mT, da)); h += t; da -= t; x += Math.cos(h) * 50; y += Math.sin(h) * 50; if (k === 10) pt = { xx: x, yy: y }; }
  const mTn = 50 / V6.turnRadius(4, 6.99); x = 32550; y = 32550; h = 0; da = a0; let minD = 1e9; for (let k = 1; k <= 14; k++) { const t = Math.max(-mTn, Math.min(mTn, da)); h += t; da -= t; x += Math.cos(h) * 50; y += Math.sin(h) * 50; minD = Math.min(minD, Math.hypot(x - pt.xx, y - pt.yy)); }
  ok(minD > 100, `v9: test obstacle sits on the boost arc but ${minD.toFixed(0)} units from the normal arc`);
  setup(pt); const d1 = g.__dbg;
  ok(g.__v6err === undefined && d1.mode === 'loot' && d1.bchk === false && d1.why === '', `v9: boost refused when the wider boost-radius rollout is blocked (why '${d1.why}', check ${d1.bchk}, clear ${d1.clear})`);
}
// ---- stake: boosting must pay for itself, bars scale with length ----
{ const g = fresh(); const run = sct => { const me = mkMe({ sct }); g.V6._pb = false; g.V6.coil = null; const orb = { xx: 32950, yy: 32550, sz: 11 };
    g.slither = me; g.slithers = [me, { id: 9, xx: 32950, yy: 32930, ang: Math.PI, sp: 5.78, sc: 1.3, pts: [] }]; g.foods = [orb]; g.tick(500); g.__botStep6(); return { ...g.__dbg }; };
  const a = run(60), b = run(199);
  ok(a.mode === 'orb' && a.why === 'orb' && a.stake < b.stake, `v9: small length boosts for a contested 24-value orb (stake ${a.stake})`);
  ok(b.mode === 'orb' && b.why === '' && b.stake > 1.25, `v9: bigger length (stake ${b.stake}) will not burn length for the same orb`);
  const me = mkMe({ sct: 199 }); g.slither = me; g.slithers = [me]; g.foods = []; g.__botStep6(); ok(Math.abs(g.__dbg.stake - (1 + g.__dbg.L / 5000)) < 0.01, 'v9: stake = 1 + L/stakeL');
}
// ---- 2. length awareness, growth ----
{ const g = fresh(), V6 = g.V6; g.fpsls = Array.from({ length: 700 }, (_, i) => i * 0.5 + i * i * 0.0004); g.fmlts = Array.from({ length: 700 }, (_, i) => 1 + i * 0.002);
  for (const sct of [30, 100, 250, 400]) { const ex = V6.lenOf({ sct, fam: 0.37, sc: Math.min(6, 1 + (sct - 2) / 106) }), es = V6.lenOf({ sc: Math.min(6, 1 + (sct - 2) / 106) });
    ok(ex.exact && !es.exact && Math.abs(es.len / ex.len - 1) < 0.1 && !es.lowerBound, `v9: lenOf sct ${sct}: width-estimate ${es.len} vs exact ${ex.len} (${(100 * (es.len / ex.len - 1)).toFixed(1)}%)`); }
  const big = V6.lenOf({ sc: 6 }); ok(!big.exact && big.lowerBound && big.sct >= 531, 'v9: lenOf at max width is flagged lowerBound');
  const b0 = V6.grow(100, 0.5, 0), b1 = V6.grow(100, 0.5, 200), b2 = V6.grow(100, 0.5, 2000);
  ok(b0.sct === 100 && b0.fam === 0.5 && b1.sct >= b0.sct && b2.sct > b1.sct && b2.width >= b1.width && b2.turnR >= b1.turnR && b1.fam >= 0 && b1.fam < 1, `v9: grow() monotone (sct ${b0.sct} -> ${b1.sct} -> ${b2.sct}, width ${b0.width.toFixed(1)} -> ${b2.width.toFixed(1)})`);
  ok(V6.cfg.growK > 0, 'v9: growK exposed in cfg (approximate, tune live)');
  ok(V6.bodyLen({ pts: [{ xx: 0, yy: 0 }, { xx: 30, yy: 40 }, { xx: 30, yy: 40, dying: true }, { xx: 30, yy: 140 }] }) === 150, 'v9: bodyLen sums distances between non-dying points');
  ok(V6.bodyLen({ sct: 10, fam: 0, pts: [] }) === 10 * 42, 'v9: bodyLen falls back to sct x 42');
  // kill value grows with the victim's length
  const me = mkMe(); g.slither = me; const mkv = sct => ({ id: 50, xx: 32800, yy: 32550, ang: 0, sp: 5.78, sc: 1, sct, fam: 0.5, pts: [] }); const val = sct => { const v = mkv(sct); g.slithers = [me, v]; g.V8.trusted = true; g.V8.err.set(50, 20);
    const Y = new Float32Array(12); g.V8.pred.set(50, { t: g.Date.now(), x: v.xx, y: v.yy, a: 0, Y }); return g.V8.active(me, 40)[0].val; };
  const v1 = val(20), v2 = val(190); ok(v2 > v1, `v9: a longer victim is worth more (kill value ${v1.toFixed(2)} -> ${v2.toFixed(2)})`);
  const e = g.V8.active(me, 40)[0]; ok(e.tr > 0 && e.len > 0, 'v9: predicted-head obstacles carry the enemy turn radius and length');
  // body radius is a live-tunable cfg
  const w = mkMe({ sc: 1.2, sct: 40 }), wall = { xx: 32700, yy: 32300, ang: Math.PI / 2, sp: 5.78, sc: 2, pts: Array.from({ length: 30 }, (_, i) => ({ xx: 32800, yy: 32300 + i * 20 })) };
  g.slither = w; g.slithers = [w, wall]; g.foods = []; g.__botStep6(); const b100 = g.__dbg.body; g.V6.cfg.bodyRad = 0.5; g.__botStep6(); ok(g.V6.cfg.bodyRad === 0.5 && g.__dbg.body > b100, `v9: bodyRad shrinks the body obstacle radius (${b100} -> ${g.__dbg.body})`);
}
// ---- 4. temperament ----
{ const g = fresh(), V8 = g.V8; const me = mkMe({ xx: 32550, yy: 32550, sc: 1.4 });
  const mk = (id, x, y, ang, sc = 1, sp = 5.78) => ({ id, xx: x, yy: y, ang, sp, sc, pts: [] });
  const greedy = mk(11, 33000, 32000, 0), calm = mk(12, 33000, 33100, Math.PI);   // orbs lie ahead of "greedy" and behind "calm"
  const cruiser = mk(13, 32000, 33000, 0, 1), fleer = mk(14, 32000, 32000, 0, 1), chaser = mk(15, 31000, 33300, 0, 2);   // chaser (bigger) heads at the cruiser/fleer area
  const aggr = mk(16, 33800, 31500, Math.PI, 1, 14), pas = mk(17, 33800, 33600, Math.PI, 1, 5.78), target = mk(18, 33000, 31500, 0, 1), target2 = mk(19, 33000, 33600, 0, 1);
  const panic = mk(20, 31500, 31500, 0, 1, 14), steady = mk(21, 31500, 31900, 0, 1, 14);
  const all = [me, greedy, calm, cruiser, fleer, chaser, aggr, pas, target, target2, panic, steady]; g.slither = me; g.slithers = all;
  g.foods = [{ xx: 33450, yy: 32000, sz: 12 }, { xx: 33000 + 450, yy: 33100, sz: 12 }];   // calm's orb is 450 ahead of it only in x+, it faces -x
  let ph = 0;
  for (let n = 0; n < 60; n++) { ph = n;
    for (const e of all) { if (e === me) continue; const v = (e.sp > 9 ? 52 : 22); e.xx += Math.cos(e.ang) * v; e.yy += Math.sin(e.ang) * v; }
    greedy.xx = 33000; greedy.yy = 32000; calm.xx = 33000; calm.yy = 33100;   // keep them next to their orbs
    // fearful: bigger head 300 away, closing; the fleer boosts straight away from it, the cruiser carries on towards it
    chaser.xx = 31200; chaser.yy = 32000; chaser.ang = 0; fleer.xx = 31600; fleer.yy = 32000; fleer.ang = 0; fleer.sp = 14;
    chaser.sp = 9.5; cruiser.xx = 31600; cruiser.yy = 33000; cruiser.ang = Math.PI; cruiser.sp = 5.78; const ch2 = chaser; ch2.sc = 2;
    // aggressive boosts at a snake ahead of it; passive at the same geometry does not boost
    aggr.xx = 33800; aggr.yy = 31500; aggr.ang = Math.PI; pas.xx = 33800; pas.yy = 33600; pas.ang = Math.PI; target.xx = 33400; target.yy = 31500; target2.xx = 33400; target2.yy = 33600;
    // panic: boosting with an erratic heading; steady boosts straight
    panic.ang = n % 2 ? 0.9 : -0.9; steady.ang = 0; panic.xx = 31500; panic.yy = 31500; steady.xx = 31500; steady.yy = 31900;
    g.tick(120); V8.sample(); }
  const T = id => V8.temper.get(id);
  ok(T(11).greed > 0.6 && T(12).greed < 0.25, `v9: greed rises for a snake heading at a big orb (${T(11).greed.toFixed(2)}) and not for one facing away (${T(12).greed.toFixed(2)})`);
  ok(T(14).fear > 0.6 && T(13).fear < 0.25, `v9: fear rises for a snake that boosts away from a bigger head (${T(14).fear.toFixed(2)}), not for one that does not (${T(13).fear.toFixed(2)})`);
  ok(T(16).aggr > 0.6 && T(17).aggr < 0.25, `v9: aggression rises for a snake boosting at another snake's path (${T(16).aggr.toFixed(2)}), not a passive one (${T(17).aggr.toFixed(2)})`);
  ok(T(20).panic > 0.6 && T(21).panic < 0.25, `v9: panic rises for an erratic boosting snake (${T(20).panic.toFixed(2)}), not a steady one (${T(21).panic.toFixed(2)})`);
  const sm = V8.tsum(); ok(sm.greed >= 1 && sm.fear >= 1 && sm.aggr >= 1 && sm.panic >= 1, `v9: __dbg summary counts snakes above 0.6 (${JSON.stringify(sm)})`);
  const X = new Float32Array(V8.IN), H = V8.hist.get(11); V8.encode(greedy, H, me, V8.context(me), X, 0);
  ok(V8.IN === 104 && [T(11).greed, T(11).fear, T(11).aggr, T(11).panic].every((v, i) => Math.abs(X[100 + i] - v) < 1e-6), 'v9: the 4 temperament values are model inputs 100..103 (IN = 104)');
  // planner: aggressive bigger snake gets a wider margin / longer threat range; panic lowers confidence
  V8.trusted = true; const Y = new Float32Array(12); for (const id of [20, 21]) { const o = all.find(q => q.id === id); V8.err.set(id, 20); V8.pred.set(id, { t: g.Date.now(), x: o.xx, y: o.yy, a: 0, Y }); }
  const act = V8.active(me, 40), cp = act.find(e => e.id === 20).conf, cs = act.find(e => e.id === 21).conf; ok(cp < cs, `v9: panicking snake's predictions are trusted less (${cp.toFixed(2)} < ${cs.toFixed(2)})`);
  g.slither = null; V8.sample(); ok(V8.temper.size === 0, 'v9: temperament cleared with the other per-snake maps');
}
// ---- 5. coiling ----
{ const g = fresh(); const prey = (id, x, y, sc = 0.8) => ({ id, xx: x, yy: y, ang: 0, sp: 5.78, sc, pts: [] });
  const go = (me, sn, foods = []) => { g.slither = me; g.slithers = [me, ...sn]; g.foods = foods; g.__botStep6(); return g.__dbg; };
  let me = mkMe({ pts: body(110) }), d = go(me, [prey(7, 32850, 32650)]);
  ok(d.mode === 'coil' && d.L < 1200, `v9: coils a feasible small target below the old 1200-length gate (L ${d.L}, body ${Math.round(g.V6.bodyLen(me))} units, mode ${d.mode})`);
  g.V6.coil = null; me = mkMe({ pts: body(30) }); d = go(me, [prey(7, 32850, 32650)]);
  ok(d.mode !== 'coil', `v9: no coil when the body is too short to close the ring (body ${Math.round(g.V6.bodyLen(me))} units, mode ${d.mode})`);
  g.V6.coil = null; me = mkMe({ pts: body(110) }); d = go(me, [prey(7, 32850, 32650, 1.3)]); ok(d.mode !== 'coil', `v9: no coil on a target near our own size (mode ${d.mode})`);
  g.V6.coil = null; d = go(me, [prey(7, 32850, 32650), { id: 8, xx: 33300, yy: 32550, ang: Math.PI, sp: 5.78, sc: 3, pts: [] }]); ok(d.mode !== 'coil', `v9: no coil with a much bigger snake within 1000 (mode ${d.mode})`);
  // hysteresis: keep the target (by id) although another becomes nearer; turn direction stays fixed; re-pick after ~6 s
  g.V6.coil = null; me = mkMe({ pts: body(110) }); const p1 = prey(7, 32900, 32700), p2 = prey(8, 33300, 32550); d = go(me, [p1, p2]); const first = g.V6.coil && g.V6.coil.id, sgn = g.V6.coil.sgn;
  p1.xx = 33050; p1.yy = 32800; p2.xx = 32800; p2.yy = 32300; g.tick(1000); d = go(me, [p1, p2]);
  ok(first === 7 && g.V6.coil.id === 7 && g.V6.coil.sgn === sgn, `v9: coil keeps its target (id ${g.V6.coil.id}) while a nearer one appears, direction ${sgn}`);
  p2.xx = 32800; p2.yy = 32650; g.tick(6500); d = go(me, [p1, p2]); ok(g.V6.coil && g.V6.coil.id === 8, `v9: after ~6 s it re-picks the best target (now id ${g.V6.coil && g.V6.coil.id})`);
  // circling / fearful targets get a bonus in the choice
  g.V6.coil = null; me = mkMe({ pts: body(110) }); const near1 = prey(21, 32800, 32650), far1 = prey(22, 32800, 32300); g.V8.circ.add(22); d = go(me, [near1, far1]);
  ok(g.V6.coil && g.V6.coil.id === 22, `v9: a snake going round in circles is preferred over a nearer ordinary one (id ${g.V6.coil && g.V6.coil.id})`);
  g.V8.circ.clear(); g.V6.coil = null; g.V8.temper.set(22, { greed: 0, fear: 0.95, aggr: 0, panic: 0, n: 50 }); d = go(me, [near1, far1]); ok(g.V6.coil && g.V6.coil.id === 22, `v9: a fearful snake is preferred too (id ${g.V6.coil && g.V6.coil.id})`);
  // tightening once closed: sweep around the target accumulates, then the ring shrinks
  g.V8.temper.clear(); g.V6.coil = null; me = mkMe({ pts: body(110) }); const tg = prey(30, 32850, 32550); d = go(me, [tg]); const sg = g.V6.coil.sgn; const R0 = g.V6.coil.Rc;
  for (let i = 1; i <= 36; i++) { const a = Math.PI + sg * i * (Math.PI / 18); me.xx = tg.xx + Math.cos(a) * 250; me.yy = tg.yy + Math.sin(a) * 250; g.tick(40); go(me, [tg]); if (!g.V6.coil) break; }
  ok(g.V6.coil && g.V6.coil.sweep > 5.3 && g.V6.coil.Rc < R0, `v9: after a full lap the ring tightens (sweep ${g.V6.coil && g.V6.coil.sweep.toFixed(1)} rad, radius ${R0.toFixed(0)} -> ${g.V6.coil && g.V6.coil.Rc.toFixed(0)})`);
  // loot still outranks coiling
  g.V6.coil = null; me = mkMe({ pts: body(110) }); const trail = Array.from({ length: 18 }, (_, i) => ({ xx: 32100 - i * 25, yy: 32200 - i * 12, sz: 12 })); g.tick(1500); d = go(me, [prey(7, 32850, 32650)], trail);
  ok(d.mode === 'loot', `v9: loot outranks coiling (mode ${d.mode})`);
  ok(g.V6.cfg.coilW >= 700, 'v9: coilW raised to 700');
  const g2 = fresh(); ok(g2.V7.TUNE.includes('coilW') && g2.V7.TUNE.includes('stakeL'), 'v9: coilW and stakeL are tuned by the evolution strategy');
}
// ---- 6. saved model of the old shape is ignored ----
{ const g = page(); const data = new Map(); const mkReq = r => { setTimeout(() => { r.onsuccess && r.onsuccess(); }, 0); return r; };
  g.indexedDB = { open() { const q = {}; setTimeout(() => { q.result = { createObjectStore() {}, close() {}, transaction() { const tx = {}; tx.objectStore = () => ({ get: k => { const r = { result: data.get(k) }; setTimeout(() => tx.oncomplete && tx.oncomplete(), 0); return r; }, put: (v, k) => { data.set(k, v); const r = { result: k }; setTimeout(() => tx.oncomplete && tx.oncomplete(), 0); return r; } }); return tx; } }; q.onupgradeneeded && q.onupgradeneeded(); q.onsuccess && q.onsuccess(); }, 0); return q; } };
  g.run('bot/slither-bot-v9.bundle.js'); const V8 = g.V8; const LY = V8.LAYERS;
  const old = [[100, 768], [768, 768], [768, 512], [512, 12]].map(([a, b]) => ({ W: new Float32Array(a * b), b: new Float32Array(b) })); data.set('enemy', { P: old, savedAt: 'x' }); data.set('enemy9', { P: old.slice(0, 3), savedAt: 'y' });
  let r, err = null; try { r = await V8.loadModel(); } catch (e) { err = e; }
  ok(!err && r === null && V8.ignoredModel === true, 'v9: a saved model with different layer sizes is ignored without throwing');
  let started = null; try { started = await V8.solo(); } catch (e) { err = e; } ok(!err && /fresh/.test(started), `v9: solo starts fresh after ignoring it (${started})`); V8.shutdown();
  const good = LY.slice(0, -1).map((a, l) => ({ W: new Float32Array(a * LY[l + 1]), b: new Float32Array(LY[l + 1]) })); await V8.storeModel(good, {});
  ok(V8.validModel(data.get('enemy9')) && (await V8.loadModel()).P.length === 3 && data.get('enemy').P === old, 'v9: a model of the new shape saves to key enemy9 and loads; the old key is left untouched');
}
// ---- 8. caches ----
{ const g = fresh(); const me = mkMe(); const trail = Array.from({ length: 18 }, (_, i) => ({ xx: 32100 - i * 25, yy: 32200 - i * 12, sz: 12 })); g.slither = me; g.slithers = [me]; g.foods = trail; g.tick(1500);
  const a = g.V8.loot(me, 32550, 32550, 13000, 330, g.V6.cfg), c1 = g.V8._lc; const b = g.V8.loot(me, 32550, 32550, 13000, 330, g.V6.cfg);
  ok(a && b && g.V8._lc === c1 && a.v === b.v, `v9: loot result cached for 160 ms (value ${a && a.v})`); g.tick(200); g.V8.loot(me, 32550, 32550, 13000, 330, g.V6.cfg); ok(g.V8._lc !== c1, 'v9: loot cache expires');
  me.kill_count = 1; const c2 = g.V8._lc; g.V8.loot(me, 32550, 32550, 13000, 330, g.V6.cfg); ok(g.V8._lc !== c2, 'v9: loot recomputed at once when kill_count rises');
  // bucketed loot sums equal the old O(n^2) pair loop
  g.V8._lc = null; g.foods = trail; g.tick(1); const r = g.V8._loot(me, 32550, 32550, 13000, 330, g.V6.cfg, g.Date.now()); let v = 0; for (const q of trail) if (Math.abs(q.xx - r.x) + Math.abs(q.yy - r.y) < 300) v += q.sz * q.sz / 5; ok(r && Math.abs(r.v - v) < 1e-9, `v9: grid-bucketed loot value equals the pair-loop value (${r && r.v.toFixed(1)})`);
  ok(g.V8.ver === 'v9.0' && g.V8.gpu && g.V8.gpu.enabled === false, 'v9: version v9.0, GPU path deliberately not shipped');
}
process.exit(0);
