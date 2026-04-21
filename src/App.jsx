import { useState, useRef, useEffect, useCallback } from "react";

// === Environment: simplified LunarLander ===
const WORLD_W = 20;
const WORLD_H = 15;
const GRAVITY = 0.006;
const MAIN_THRUST = 0.014;
const SIDE_TORQUE = 0.0014;
const OMEGA_DAMP = 0.985;
const PAD_X = 10, PAD_HALF = 1.9;
const GROUND_Y = 0.55;
const MAX_STEPS = 400;

const N_OBS = 7, N_ACT = 4;

function rand(a, b) { return Math.random() * (b - a) + a; }
function randn() {
  const u = Math.random(), v = Math.random();
  return Math.sqrt(-2 * Math.log(u + 1e-9)) * Math.cos(2 * Math.PI * v);
}

function envReset() {
  return {
    x: WORLD_W / 2 + rand(-3.5, 3.5),
    y: WORLD_H - 2,
    vx: rand(-0.08, 0.08),
    vy: 0,
    theta: rand(-0.08, 0.08),
    omega: 0,
    steps: 0,
    landed: false,
    crashed: false,
    lastAction: 0,
    done: false,
  };
}

function envStep(s, action) {
  if (s.done) return { state: s, reward: 0, done: true };
  let { x, y, vx, vy, theta, omega, steps } = s;
  let tX = 0, tY = 0, torque = 0;
  if (action === 1) {
    tX = Math.sin(theta) * MAIN_THRUST;
    tY = Math.cos(theta) * MAIN_THRUST;
  } else if (action === 2) {
    torque = -SIDE_TORQUE;
  } else if (action === 3) {
    torque = SIDE_TORQUE;
  }
  vy += tY - GRAVITY;
  vx += tX;
  omega += torque;
  omega *= OMEGA_DAMP;
  x += vx;
  y += vy;
  theta += omega;
  steps += 1;

  // reward shaping: stronger gradient toward pad, especially near ground
  const heightFactor = Math.max(0, 1 - Math.max(0, y) / WORLD_H);
  let reward = 0;
  reward -= Math.abs(x - PAD_X) * (0.01 + 0.03 * heightFactor);
  reward -= Math.abs(theta) * 0.08;
  reward -= (Math.abs(vx) * 0.05 + Math.abs(vy) * 0.03);
  reward -= 0.012;
  if (action === 1) reward -= 0.015;

  const ns = { x, y, vx, vy, theta, omega, steps, landed: false, crashed: false, lastAction: action, done: false };

  let done = false;
  if (y <= GROUND_Y) {
    ns.y = GROUND_Y;
    done = true;
    const atPad = Math.abs(x - PAD_X) < PAD_HALF;
    const soft = Math.abs(vx) < 0.35 && Math.abs(vy) < 0.55;
    const upright = Math.abs(theta) < 0.4;
    if (atPad && soft && upright) {
      reward += 150;
      ns.landed = true;
    } else {
      reward -= 40;
      ns.crashed = true;
    }
  }
  if (x < -1 || x > WORLD_W + 1 || y > WORLD_H + 3) {
    done = true;
    reward -= 40;
    ns.crashed = true;
  }
  if (steps >= MAX_STEPS) done = true;
  ns.done = done;
  return { state: ns, reward, done };
}

function obsVec(s) {
  return [
    (s.x - WORLD_W / 2) / (WORLD_W / 2),
    (s.y - WORLD_H / 2) / (WORLD_H / 2),
    s.vx / 1.5,
    s.vy / 1.5,
    s.theta,
    s.omega * 5,
    1,
  ];
}

function policyAction(W, s) {
  const o = obsVec(s);
  let best = 0, bestVal = -Infinity;
  for (let a = 0; a < N_ACT; a++) {
    let v = 0;
    for (let i = 0; i < N_OBS; i++) v += W[a][i] * o[i];
    if (v > bestVal) { bestVal = v; best = a; }
  }
  return best;
}

function makeZeros() {
  return [...Array(N_ACT)].map(() => [...Array(N_OBS)].fill(0));
}
function makeSmallRandom(scale = 0.2) {
  return [...Array(N_ACT)].map(() => [...Array(N_OBS)].map(() => randn() * scale));
}
function makeUniform(val) {
  return [...Array(N_ACT)].map(() => [...Array(N_OBS)].fill(val));
}
function cloneW(W) { return W.map((r) => r.slice()); }

function runEpisode(weights, maxSteps = MAX_STEPS) {
  let s = envReset();
  let total = 0;
  for (let t = 0; t < maxSteps; t++) {
    const a = policyAction(weights, s);
    const { state, reward, done } = envStep(s, a);
    s = state;
    total += reward;
    if (done) break;
  }
  return total;
}

function cemGeneration(mean, std, popSize, elite, evalEpisodes) {
  const pop = [];
  for (let p = 0; p < popSize; p++) {
    const w = mean.map((row, a) => row.map((m, i) => m + std[a][i] * randn()));
    let totalR = 0;
    for (let e = 0; e < evalEpisodes; e++) totalR += runEpisode(w);
    pop.push({ w, reward: totalR / evalEpisodes });
  }
  pop.sort((a, b) => b.reward - a.reward);
  const top = pop.slice(0, elite);
  const newMean = mean.map((row, a) => row.map((_, i) =>
    top.reduce((s, p) => s + p.w[a][i], 0) / top.length
  ));
  // std floor (0.1) keeps exploration alive; multiply by 0.97 to gently anneal
  const STD_FLOOR = 0.1;
  const ANNEAL = 0.97;
  const newStd = mean.map((row, a) => row.map((_, i) => {
    const m = newMean[a][i];
    const v = top.reduce((s, p) => s + (p.w[a][i] - m) ** 2, 0) / top.length;
    return Math.max(STD_FLOOR, Math.sqrt(v)) * ANNEAL;
  }));
  return {
    mean: newMean,
    std: newStd,
    best: pop[0],
    avg: pop.reduce((s, p) => s + p.reward, 0) / pop.length,
  };
}

// === Rendering helpers ===
const font = "'JetBrains Mono', 'Fira Code', 'SF Mono', monospace";

function LanderView({ state, subtitle, badgeColor, bodyColor, epReward, lastEpReward }) {
  const pxW = 340, pxH = 360;
  const toPx = (wx, wy) => [
    (wx / WORLD_W) * pxW,
    pxH - ((wy - GROUND_Y) / (WORLD_H - GROUND_Y)) * (pxH - 40) - 40,
  ];
  const [lx, ly] = toPx(state.x, state.y);
  const [padL] = toPx(PAD_X - PAD_HALF, 0);
  const [padR] = toPx(PAD_X + PAD_HALF, 0);
  const [, padY] = toPx(0, GROUND_Y);
  const groundY = padY;

  const thetaDeg = (state.theta * 180 / Math.PI).toFixed(1);
  const mainOn = state.lastAction === 1 && !state.crashed && !state.landed;
  const leftOn = state.lastAction === 2 && !state.crashed && !state.landed;
  const rightOn = state.lastAction === 3 && !state.crashed && !state.landed;

  return (
    <svg viewBox={`0 0 ${pxW} ${pxH}`} style={{ width: "100%", display: "block", background: "#070914", borderRadius: 8 }}>
      {/* stars */}
      {[...Array(22)].map((_, i) => {
        const sx = (i * 137.5) % pxW;
        const sy = (i * 47.3) % (pxH - 50);
        const r = (i % 3 === 0) ? 0.9 : 0.5;
        return <circle key={i} cx={sx} cy={sy} r={r} fill="#cbd5e1" opacity={0.4 + (i % 3) * 0.15} />;
      })}
      {/* ground */}
      <rect x={0} y={groundY} width={pxW} height={pxH - groundY} fill="#1e293b" />
      <rect x={0} y={groundY} width={pxW} height={2} fill="#475569" />
      {/* landing pad */}
      <rect x={padL} y={groundY - 6} width={padR - padL} height={6} fill="#fbbf24" />
      <line x1={padL} y1={groundY - 6} x2={padL} y2={groundY - 16} stroke="#fbbf24" strokeWidth="1.5" />
      <line x1={padR} y1={groundY - 6} x2={padR} y2={groundY - 16} stroke="#fbbf24" strokeWidth="1.5" />
      {/* lander */}
      <g transform={`translate(${lx.toFixed(2)},${ly.toFixed(2)}) rotate(${thetaDeg})`}>
        {mainOn && (
          <polygon points="-4,10 4,10 0,22" fill="#f97316" opacity={0.85}>
            <animate attributeName="opacity" values="0.7;1;0.7" dur="0.15s" repeatCount="indefinite" />
          </polygon>
        )}
        {leftOn && <polygon points="6,-2 6,4 14,1" fill="#f97316" opacity={0.9} />}
        {rightOn && <polygon points="-6,-2 -6,4 -14,1" fill="#f97316" opacity={0.9} />}
        {/* body */}
        <polygon points="-9,8 9,8 6,-9 -6,-9" fill={bodyColor} stroke="#0b1225" strokeWidth="1" />
        <circle cx={0} cy={-3} r={2.5} fill="#0ea5e9" opacity={0.85} />
        {/* legs */}
        <line x1={-9} y1={8} x2={-13} y2={12} stroke="#94a3b8" strokeWidth="1.5" />
        <line x1={9} y1={8} x2={13} y2={12} stroke="#94a3b8" strokeWidth="1.5" />
      </g>
      {/* status badge */}
      <rect x={8} y={8} width={150} height={20} fill="rgba(11, 18, 37, 0.8)" stroke={badgeColor} rx={3} />
      <text x={83} y={22} textAnchor="middle" fill={badgeColor} fontSize="10" fontFamily={font} fontWeight={600}>
        {subtitle}
      </text>
      {/* live episode reward */}
      <rect x={pxW - 120} y={8} width={112} height={38} fill="rgba(11, 18, 37, 0.8)" stroke="#334155" rx={3} />
      <text x={pxW - 114} y={22} fill="#64748b" fontSize="9" fontFamily={font}>this ep</text>
      <text x={pxW - 14} y={22} textAnchor="end" fill={epReward >= 0 ? "#34d399" : "#f87171"} fontSize="11" fontFamily={font} fontWeight={600}>
        {epReward == null ? "–" : epReward.toFixed(1)}
      </text>
      <text x={pxW - 114} y={38} fill="#64748b" fontSize="9" fontFamily={font}>last</text>
      <text x={pxW - 14} y={38} textAnchor="end" fill={lastEpReward == null ? "#64748b" : (lastEpReward >= 0 ? "#34d399" : "#f87171")} fontSize="11" fontFamily={font}>
        {lastEpReward == null ? "–" : lastEpReward.toFixed(1)}
      </text>
      {/* outcome text */}
      {(state.landed || state.crashed) && (
        <g>
          <rect x={pxW / 2 - 60} y={pxH / 2 - 18} width={120} height={32} fill="rgba(11, 18, 37, 0.85)" stroke={state.landed ? "#34d399" : "#f87171"} rx={4} />
          <text x={pxW / 2} y={pxH / 2 + 4} textAnchor="middle" fill={state.landed ? "#34d399" : "#f87171"} fontSize="13" fontFamily={font} fontWeight={600}>
            {state.landed ? "LANDED" : "CRASHED"}
          </text>
        </g>
      )}
    </svg>
  );
}

function btn(active, color) {
  return {
    background: active ? color : "#1e293b",
    color: active ? "#0b1225" : "#e2e8f0",
    border: `1px solid ${active ? color : "#334155"}`,
    borderRadius: 4,
    padding: "6px 14px",
    fontSize: 11,
    fontFamily: font,
    cursor: "pointer",
    fontWeight: active ? 600 : 400,
  };
}

export default function App() {
  // Policies
  const trainedRef = useRef(makeSmallRandom(0.15));
  const bestEverRef = useRef({ w: makeSmallRandom(0.15), reward: -Infinity });
  const meanRef = useRef(makeSmallRandom(0.15));
  const stdRef = useRef(makeUniform(1.2));

  // Env states (refs so we can animate without re-init each frame)
  const untrainedState = useRef(envReset());
  const trainedState = useRef(envReset());
  const untrainedEpRewardRef = useRef(0);
  const trainedEpRewardRef = useRef(0);
  const untrainedLastRewardRef = useRef(null);
  const trainedLastRewardRef = useRef(null);
  const untrainedHoldUntilRef = useRef(0);
  const trainedHoldUntilRef = useRef(0);

  // UI state (only updated at frame tick)
  const [, forceRender] = useState(0);
  const [training, setTraining] = useState(false);
  const [generation, setGeneration] = useState(0);
  const [bestReward, setBestReward] = useState(null);
  const [avgReward, setAvgReward] = useState(null);
  const [bestEverReward, setBestEverReward] = useState(null);
  const [history, setHistory] = useState([]);
  const [popSize, setPopSize] = useState(40);
  const [eliteFrac, setEliteFrac] = useState(0.25);
  const [evalEps, setEvalEps] = useState(3);

  const trainingRef = useRef(false);
  const stopFlag = useRef(false);

  // Animation loop
  useEffect(() => {
    let running = true;
    let lastTick = 0;
    const tick = (ts) => {
      if (!running) return;
      if (ts - lastTick >= 33) { // ~30 fps
        lastTick = ts;
        // Untrained: random action (pure baseline)
        if (untrainedHoldUntilRef.current > 0) {
          untrainedHoldUntilRef.current -= 1;
          if (untrainedHoldUntilRef.current === 0) {
            untrainedState.current = envReset();
            untrainedEpRewardRef.current = 0;
          }
        } else {
          const a = Math.floor(Math.random() * N_ACT);
          const { state, reward, done } = envStep(untrainedState.current, a);
          untrainedState.current = state;
          untrainedEpRewardRef.current += reward;
          if (done) {
            untrainedLastRewardRef.current = untrainedEpRewardRef.current;
            untrainedHoldUntilRef.current = 24;
          }
        }
        // Trained: policy-based action
        if (trainedHoldUntilRef.current > 0) {
          trainedHoldUntilRef.current -= 1;
          if (trainedHoldUntilRef.current === 0) {
            trainedState.current = envReset();
            trainedEpRewardRef.current = 0;
          }
        } else {
          const a = policyAction(trainedRef.current, trainedState.current);
          const { state, reward, done } = envStep(trainedState.current, a);
          trainedState.current = state;
          trainedEpRewardRef.current += reward;
          if (done) {
            trainedLastRewardRef.current = trainedEpRewardRef.current;
            trainedHoldUntilRef.current = 24;
          }
        }
        forceRender((n) => (n + 1) % 1000000);
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
    return () => { running = false; };
  }, []);

  const resetTraining = () => {
    stopFlag.current = true;
    trainingRef.current = false;
    setTraining(false);
    const initW = makeSmallRandom(0.15);
    meanRef.current = initW;
    stdRef.current = makeUniform(1.2);
    trainedRef.current = cloneW(initW);
    bestEverRef.current = { w: cloneW(initW), reward: -Infinity };
    setGeneration(0); setBestReward(null); setAvgReward(null); setBestEverReward(null); setHistory([]);
  };

  const resetTrainedEp = () => {
    trainedState.current = envReset();
    trainedHoldUntilRef.current = 0;
  };
  const resetUntrainedEp = () => {
    untrainedState.current = envReset();
    untrainedHoldUntilRef.current = 0;
  };

  const startTrain = useCallback(async () => {
    if (trainingRef.current) return;
    trainingRef.current = true;
    stopFlag.current = false;
    setTraining(true);
    const elite = Math.max(2, Math.floor(popSize * eliteFrac));
    while (trainingRef.current && !stopFlag.current) {
      // run one CEM generation (synchronous), yield between
      const { mean: nm, std: ns, best, avg } = cemGeneration(meanRef.current, stdRef.current, popSize, elite, evalEps);
      meanRef.current = nm;
      stdRef.current = ns;
      // Keep best-ever policy on the right panel so it improves monotonically.
      if (best.reward > bestEverRef.current.reward) {
        bestEverRef.current = { w: cloneW(best.w), reward: best.reward };
        trainedRef.current = cloneW(best.w);
        setBestEverReward(best.reward);
      }
      setGeneration((g) => g + 1);
      setBestReward(best.reward);
      setAvgReward(avg);
      setHistory((h) => {
        const next = [...h, { best: best.reward, avg, bestEver: bestEverRef.current.reward }];
        return next.length > 160 ? next.slice(-160) : next;
      });
      // yield to browser
      await new Promise((r) => setTimeout(r, 6));
    }
    trainingRef.current = false;
    setTraining(false);
  }, [popSize, eliteFrac, evalEps]);

  const stopTrain = () => {
    stopFlag.current = true;
    trainingRef.current = false;
    setTraining(false);
  };

  const lossChart = history.length > 1 ? (() => {
    const all = [];
    for (const h of history) { all.push(h.best); all.push(h.avg); all.push(h.bestEver); }
    const lo = Math.min(...all);
    const hi = Math.max(...all);
    const rng = hi - lo || 1;
    const cw = 260, ch = 68;
    const step = cw / (history.length - 1);
    const fmt = (v) => ch - ((v - lo) / rng) * (ch - 4) - 2;
    const bestD = history.map((h, i) => `${i === 0 ? "M" : "L"}${(i * step).toFixed(1)},${fmt(h.best).toFixed(1)}`).join(" ");
    const avgD = history.map((h, i) => `${i === 0 ? "M" : "L"}${(i * step).toFixed(1)},${fmt(h.avg).toFixed(1)}`).join(" ");
    const everD = history.map((h, i) => `${i === 0 ? "M" : "L"}${(i * step).toFixed(1)},${fmt(h.bestEver).toFixed(1)}`).join(" ");
    const zeroY = fmt(0);
    return (
      <svg viewBox={`0 0 ${cw} ${ch}`} style={{ width: "100%", height: 68, display: "block" }}>
        {(zeroY >= 0 && zeroY <= ch) && <line x1={0} y1={zeroY} x2={cw} y2={zeroY} stroke="#334155" strokeWidth="0.6" strokeDasharray="2 3" />}
        <path d={avgD} fill="none" stroke="#94a3b8" strokeWidth="1.2" />
        <path d={bestD} fill="none" stroke="#38bdf8" strokeWidth="1.2" opacity={0.8} />
        <path d={everD} fill="none" stroke="#34d399" strokeWidth="1.8" />
      </svg>
    );
  })() : null;

  const panel = { background: "#11162a", borderRadius: 8, padding: 12, border: "1px solid #1e293b" };

  return (
    <div style={{ minHeight: "100vh", background: "#0c0f1a", color: "#e2e8f0", fontFamily: font, padding: 18 }}>
      <div style={{ maxWidth: 1240, margin: "0 auto" }}>
        <header style={{ marginBottom: 16 }}>
          <h1 style={{ fontSize: 22, fontWeight: 500, letterSpacing: "0.5px" }}>Reinforcement Learning · LunarLander (CEM)</h1>
          <p style={{ color: "#94a3b8", fontSize: 12, marginTop: 4, lineHeight: 1.55 }}>
            A simplified LunarLander environment (gravity, main + side thrusters, rotation, yellow landing pad).
            Left panel = untrained agent (random actions). Right panel = agent controlled by the current best policy from training.
            Algorithm: <span style={{ color: "#34d399" }}>Cross-Entropy Method</span> over a linear policy (state → 4 discrete actions).
          </p>
        </header>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 18 }}>
          <div style={panel}>
            <LanderView
              state={untrainedState.current}
              subtitle="UNTRAINED"
              badgeColor="#f87171"
              bodyColor="#94a3b8"
              epReward={untrainedEpRewardRef.current}
              lastEpReward={untrainedLastRewardRef.current}
            />
            <div style={{ display: "flex", gap: 8, marginTop: 10, alignItems: "center" }}>
              <button onClick={resetUntrainedEp} style={btn(false, "#64748b")}>New episode</button>
              <span style={{ color: "#64748b", fontSize: 10, marginLeft: "auto" }}>random actions</span>
            </div>
          </div>
          <div style={panel}>
            <LanderView
              state={trainedState.current}
              subtitle={generation > 0 ? `TRAINED (gen ${generation})` : "TRAINED"}
              badgeColor="#34d399"
              bodyColor="#38bdf8"
              epReward={trainedEpRewardRef.current}
              lastEpReward={trainedLastRewardRef.current}
            />
            <div style={{ display: "flex", gap: 8, marginTop: 10, alignItems: "center" }}>
              <button onClick={resetTrainedEp} style={btn(false, "#64748b")}>New episode</button>
              <span style={{ color: "#64748b", fontSize: 10, marginLeft: "auto" }}>best-ever policy</span>
            </div>
          </div>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: 18, marginTop: 18 }}>
          <div style={panel}>
            <div style={{ fontSize: 11, color: "#94a3b8", marginBottom: 10 }}>training control</div>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
              {!training
                ? <button onClick={startTrain} style={btn(false, "#34d399")}>Start training</button>
                : <button onClick={stopTrain} style={btn(true, "#f87171")}>Stop</button>}
              <button onClick={resetTraining} style={btn(false, "#64748b")}>Reset training</button>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, marginTop: 14 }}>
              <div>
                <div style={{ fontSize: 10, color: "#64748b", marginBottom: 4 }}>population size</div>
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <input type="range" min={10} max={80} step={2} value={popSize}
                    onChange={(e) => setPopSize(parseInt(e.target.value, 10))}
                    disabled={training}
                    style={{ flex: 1 }} />
                  <span style={{ color: "#67e8f9", fontSize: 11, width: 26, textAlign: "right" }}>{popSize}</span>
                </div>
              </div>
              <div>
                <div style={{ fontSize: 10, color: "#64748b", marginBottom: 4 }}>elite fraction</div>
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <input type="range" min={0.1} max={0.5} step={0.05} value={eliteFrac}
                    onChange={(e) => setEliteFrac(parseFloat(e.target.value))}
                    disabled={training}
                    style={{ flex: 1 }} />
                  <span style={{ color: "#67e8f9", fontSize: 11, width: 34, textAlign: "right" }}>{eliteFrac.toFixed(2)}</span>
                </div>
              </div>
              <div>
                <div style={{ fontSize: 10, color: "#64748b", marginBottom: 4 }}>eval episodes</div>
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <input type="range" min={1} max={5} step={1} value={evalEps}
                    onChange={(e) => setEvalEps(parseInt(e.target.value, 10))}
                    disabled={training}
                    style={{ flex: 1 }} />
                  <span style={{ color: "#67e8f9", fontSize: 11, width: 20, textAlign: "right" }}>{evalEps}</span>
                </div>
              </div>
            </div>
            <div style={{ marginTop: 14 }}>
              <div style={{ fontSize: 10, color: "#64748b", marginBottom: 4 }}>
                reward · <span style={{ color: "#34d399" }}>best-ever</span> / <span style={{ color: "#38bdf8" }}>gen best</span> / <span style={{ color: "#94a3b8" }}>gen avg</span>
              </div>
              {lossChart || <div style={{ height: 68, border: "1px dashed #1e293b", borderRadius: 4, display: "flex", alignItems: "center", justifyContent: "center", color: "#475569", fontSize: 11 }}>no data yet</div>}
            </div>
          </div>

          <div style={panel}>
            <div style={{ fontSize: 11, color: "#94a3b8", marginBottom: 8 }}>metrics</div>
            <div style={{ fontSize: 12, lineHeight: 1.8 }}>
              <div>generation: <span style={{ color: "#e2e8f0" }}>{generation}</span></div>
              <div>best-ever: <span style={{ color: "#34d399" }}>{bestEverReward == null ? "–" : bestEverReward.toFixed(2)}</span></div>
              <div>gen best: <span style={{ color: "#38bdf8" }}>{bestReward == null ? "–" : bestReward.toFixed(2)}</span></div>
              <div>gen avg:&nbsp; <span style={{ color: "#94a3b8" }}>{avgReward == null ? "–" : avgReward.toFixed(2)}</span></div>
            </div>
            <div style={{ marginTop: 10, paddingTop: 10, borderTop: "1px solid #1e293b", fontSize: 11, color: "#94a3b8", lineHeight: 1.6 }}>
              <div style={{ color: "#67e8f9", fontSize: 10, marginBottom: 4 }}>reading the plot</div>
              The green <strong style={{ color: "#34d399" }}>best</strong> curve climbs as CEM concentrates weight mass on strong policies.
              The gray <strong style={{ color: "#94a3b8" }}>mean</strong> curve trails it since CEM still samples around the current mean.
              Reward &gt; 80 usually means the agent is landing on the pad.
            </div>
            <div style={{ marginTop: 10, paddingTop: 10, borderTop: "1px solid #1e293b", fontSize: 11, color: "#94a3b8", lineHeight: 1.6 }}>
              <div style={{ color: "#fbbf24", fontSize: 10, marginBottom: 4 }}>actions</div>
              <div>0: do nothing</div>
              <div>1: fire main engine</div>
              <div>2: fire right-side thruster (rotate left)</div>
              <div>3: fire left-side thruster (rotate right)</div>
            </div>
          </div>
        </div>
        <footer style={{ marginTop: 18, color: "#475569", fontSize: 10, textAlign: "center" }}>
          simplified LunarLander · linear policy (4×7 = 28 params) · Cross-Entropy Method
        </footer>
      </div>
    </div>
  );
}
