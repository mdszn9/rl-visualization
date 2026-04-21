import { useState, useRef, useEffect, useCallback } from "react";

// === Environment: simplified LunarLander ===
const WORLD_W = 20;
const WORLD_H = 15;
const GRAVITY = 0.006;
const MAIN_THRUST = 0.016;
const SIDE_TORQUE = 0.0016;
const OMEGA_DAMP = 0.985;
const PAD_X = 10, PAD_HALF = 1.9;
const GROUND_Y = 0.55;
const MAX_STEPS = 400;

// Actions: 0 = main thrust, 1 = rotate CCW, 2 = rotate CW.
const N_OBS = 7, N_ACT = 3;

function rand(a, b) { return Math.random() * (b - a) + a; }
function randn() {
  const u = Math.random(), v = Math.random();
  return Math.sqrt(-2 * Math.log(u + 1e-9)) * Math.cos(2 * Math.PI * v);
}

function envReset() {
  return {
    x: WORLD_W / 2 + rand(-1.8, 1.8),
    y: WORLD_H - 2,
    vx: rand(-0.18, 0.18),
    vy: rand(-0.22, -0.1),
    theta: rand(-0.12, 0.12),
    omega: rand(-0.008, 0.008),
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
  if (action === 0) {
    tX = Math.sin(theta) * MAIN_THRUST;
    tY = Math.cos(theta) * MAIN_THRUST;
  } else if (action === 1) {
    torque = -SIDE_TORQUE;
  } else if (action === 2) {
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

  const heightFactor = Math.max(0, 1 - Math.max(0, y) / WORLD_H);
  let reward = 0;
  reward -= Math.abs(x - PAD_X) * (0.025 + 0.08 * heightFactor);
  reward -= Math.abs(theta) * 0.3;
  reward -= (Math.abs(vx) * 0.15 + Math.abs(vy) * 0.08);
  reward -= 0.01;
  if (Math.abs(x - PAD_X) < PAD_HALF && y > GROUND_Y + 0.5 && vy > -0.35 && vy < 0.1) {
    reward += 0.25;
  }

  const ns = { x, y, vx, vy, theta, omega, steps, landed: false, crashed: false, lastAction: action, done: false };

  let done = false;
  if (y <= GROUND_Y) {
    ns.y = GROUND_Y;
    done = true;
    const atPad = Math.abs(x - PAD_X) < PAD_HALF;
    const soft = Math.abs(vx) < 0.35 && Math.abs(vy) < 0.55;
    const upright = Math.abs(theta) < 0.4;
    if (atPad && soft && upright) { reward += 150; ns.landed = true; }
    else { reward -= 40; ns.crashed = true; }
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

// === PPO: linear softmax policy + linear value function ===
function mkWp(scale = 0.15) {
  return [...Array(N_ACT)].map(() => [...Array(N_OBS)].map(() => randn() * scale));
}
function mkWv() { return new Array(N_OBS).fill(0); }
function cloneWp(W) { return W.map((r) => r.slice()); }
function cloneWv(W) { return W.slice(); }

function policyForward(Wp, obs) {
  const logits = new Array(N_ACT);
  for (let a = 0; a < N_ACT; a++) {
    let v = 0;
    for (let i = 0; i < N_OBS; i++) v += Wp[a][i] * obs[i];
    logits[a] = v;
  }
  const maxL = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - maxL));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

function valueForward(Wv, obs) {
  let v = 0;
  for (let i = 0; i < N_OBS; i++) v += Wv[i] * obs[i];
  return v;
}

function sampleFromProbs(probs) {
  const r = Math.random();
  let c = 0;
  for (let a = 0; a < N_ACT; a++) {
    c += probs[a];
    if (r < c) return a;
  }
  return N_ACT - 1;
}

function collectRollout(Wp, Wv, rolloutSteps) {
  const transitions = [];
  const epReturns = [];
  let s = envReset();
  let epRet = 0;
  for (let t = 0; t < rolloutSteps; t++) {
    const o = obsVec(s);
    const probs = policyForward(Wp, o);
    const a = sampleFromProbs(probs);
    const logp = Math.log(probs[a] + 1e-10);
    const v = valueForward(Wv, o);
    const { state: s2, reward, done } = envStep(s, a);
    transitions.push({ obs: o, a, reward, done, v, logp });
    epRet += reward;
    if (done) {
      epReturns.push(epRet);
      epRet = 0;
      s = envReset();
    } else {
      s = s2;
    }
  }
  const lastObs = obsVec(s);
  const lastV = valueForward(Wv, lastObs);
  return { transitions, epReturns, lastV };
}

function computeAdvantages(transitions, lastV, gamma, lambda) {
  const n = transitions.length;
  const advantages = new Array(n);
  const returns = new Array(n);
  let gae = 0;
  let nextV = lastV;
  let nextNonTerm = 1;
  for (let t = n - 1; t >= 0; t--) {
    const tr = transitions[t];
    const delta = tr.reward + gamma * nextV * nextNonTerm - tr.v;
    gae = delta + gamma * lambda * nextNonTerm * gae;
    advantages[t] = gae;
    returns[t] = gae + tr.v;
    nextV = tr.v;
    nextNonTerm = tr.done ? 0 : 1;
  }
  return { advantages, returns };
}

function ppoUpdate(Wp, Wv, transitions, advantages, returns, hp) {
  const { clipEps, lr, valueCoef, entropyCoef } = hp;
  const n = transitions.length;

  // Normalize advantages (standard PPO practice)
  let mean = 0;
  for (let i = 0; i < n; i++) mean += advantages[i];
  mean /= n;
  let variance = 0;
  for (let i = 0; i < n; i++) variance += (advantages[i] - mean) ** 2;
  variance /= n;
  const std = Math.sqrt(variance) + 1e-8;
  const advNorm = new Array(n);
  for (let i = 0; i < n; i++) advNorm[i] = (advantages[i] - mean) / std;

  const gradWp = [...Array(N_ACT)].map(() => new Array(N_OBS).fill(0));
  const gradWv = new Array(N_OBS).fill(0);

  let policyLoss = 0;
  let valueLoss = 0;
  let approxKL = 0;
  let clipFrac = 0;

  for (let i = 0; i < n; i++) {
    const tr = transitions[i];
    const obs = tr.obs;
    const a = tr.a;
    const oldLogP = tr.logp;
    const adv = advNorm[i];
    const ret = returns[i];

    // Policy forward at CURRENT (possibly updated) weights
    const probs = policyForward(Wp, obs);
    const newLogP = Math.log(probs[a] + 1e-10);
    const ratio = Math.exp(newLogP - oldLogP);

    // Stats
    approxKL += (oldLogP - newLogP);
    if (ratio > 1 + clipEps || ratio < 1 - clipEps) clipFrac += 1;

    // Clipped surrogate objective gradient
    // J = min(ratio * adv, clip(ratio, 1±ε) * adv)
    // If clip bound is binding AND in the direction that would hurt J, grad is zero.
    let policyCoef = 0;
    let surrogate = 0;
    if (adv >= 0) {
      if (ratio < 1 + clipEps) { policyCoef = adv * ratio; surrogate = adv * ratio; }
      else { surrogate = adv * (1 + clipEps); }
    } else {
      if (ratio > 1 - clipEps) { policyCoef = adv * ratio; surrogate = adv * ratio; }
      else { surrogate = adv * (1 - clipEps); }
    }
    policyLoss -= surrogate; // we maximize surrogate => loss is negated

    // ∂log π(a|s)/∂Wp[act][k] = (1[act=a] - π(act|s)) * obs[k]
    for (let act = 0; act < N_ACT; act++) {
      const dLogPdL = (act === a ? 1 : 0) - probs[act];
      const coef = policyCoef * dLogPdL;
      for (let k = 0; k < N_OBS; k++) gradWp[act][k] += coef * obs[k];
    }

    // Entropy bonus: encourages exploration, prevents premature determinism
    if (entropyCoef > 0) {
      let H = 0;
      for (let act = 0; act < N_ACT; act++) H -= probs[act] * Math.log(probs[act] + 1e-10);
      for (let act = 0; act < N_ACT; act++) {
        const dHdL = -probs[act] * (Math.log(probs[act] + 1e-10) + H);
        for (let k = 0; k < N_OBS; k++) gradWp[act][k] += entropyCoef * dHdL * obs[k];
      }
    }

    // Value: L_v = 0.5 * (V(s) - ret)^2
    const v = valueForward(Wv, obs);
    const vErr = v - ret;
    valueLoss += 0.5 * vErr * vErr;
    for (let k = 0; k < N_OBS; k++) gradWv[k] += vErr * obs[k];
  }

  // Ascent on policy objective, descent on value loss
  const newWp = Wp.map((row, a) => row.map((w, i) => w + (lr / n) * gradWp[a][i]));
  const newWv = Wv.map((w, i) => w - (lr * valueCoef / n) * gradWv[i]);

  return {
    Wp: newWp,
    Wv: newWv,
    policyLoss: policyLoss / n,
    valueLoss: valueLoss / n,
    approxKL: approxKL / n,
    clipFrac: clipFrac / n,
  };
}

// === Rendering ===
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
  const mainOn = state.lastAction === 0 && !state.crashed && !state.landed;
  const leftOn = state.lastAction === 1 && !state.crashed && !state.landed;
  const rightOn = state.lastAction === 2 && !state.crashed && !state.landed;

  return (
    <svg viewBox={`0 0 ${pxW} ${pxH}`} style={{ width: "100%", display: "block", background: "#070914", borderRadius: 8 }}>
      {[...Array(22)].map((_, i) => {
        const sx = (i * 137.5) % pxW;
        const sy = (i * 47.3) % (pxH - 50);
        const r = (i % 3 === 0) ? 0.9 : 0.5;
        return <circle key={i} cx={sx} cy={sy} r={r} fill="#cbd5e1" opacity={0.4 + (i % 3) * 0.15} />;
      })}
      <rect x={0} y={groundY} width={pxW} height={pxH - groundY} fill="#1e293b" />
      <rect x={0} y={groundY} width={pxW} height={2} fill="#475569" />
      <rect x={padL} y={groundY - 6} width={padR - padL} height={6} fill="#fbbf24" />
      <line x1={padL} y1={groundY - 6} x2={padL} y2={groundY - 16} stroke="#fbbf24" strokeWidth="1.5" />
      <line x1={padR} y1={groundY - 6} x2={padR} y2={groundY - 16} stroke="#fbbf24" strokeWidth="1.5" />
      <g transform={`translate(${lx.toFixed(2)},${ly.toFixed(2)}) rotate(${thetaDeg})`}>
        {mainOn && (
          <polygon points="-4,10 4,10 0,22" fill="#f97316" opacity={0.85}>
            <animate attributeName="opacity" values="0.7;1;0.7" dur="0.15s" repeatCount="indefinite" />
          </polygon>
        )}
        {leftOn && <polygon points="6,-2 6,4 14,1" fill="#f97316" opacity={0.9} />}
        {rightOn && <polygon points="-6,-2 -6,4 -14,1" fill="#f97316" opacity={0.9} />}
        <polygon points="-9,8 9,8 6,-9 -6,-9" fill={bodyColor} stroke="#0b1225" strokeWidth="1" />
        <circle cx={0} cy={-3} r={2.5} fill="#0ea5e9" opacity={0.85} />
        <line x1={-9} y1={8} x2={-13} y2={12} stroke="#94a3b8" strokeWidth="1.5" />
        <line x1={9} y1={8} x2={13} y2={12} stroke="#94a3b8" strokeWidth="1.5" />
      </g>
      <rect x={8} y={8} width={150} height={20} fill="rgba(11, 18, 37, 0.8)" stroke={badgeColor} rx={3} />
      <text x={83} y={22} textAnchor="middle" fill={badgeColor} fontSize="10" fontFamily={font} fontWeight={600}>
        {subtitle}
      </text>
      <rect x={pxW - 120} y={8} width={112} height={38} fill="rgba(11, 18, 37, 0.8)" stroke="#334155" rx={3} />
      <text x={pxW - 114} y={22} fill="#64748b" fontSize="9" fontFamily={font}>this ep</text>
      <text x={pxW - 14} y={22} textAnchor="end" fill={epReward >= 0 ? "#34d399" : "#f87171"} fontSize="11" fontFamily={font} fontWeight={600}>
        {epReward == null ? "–" : epReward.toFixed(1)}
      </text>
      <text x={pxW - 114} y={38} fill="#64748b" fontSize="9" fontFamily={font}>last</text>
      <text x={pxW - 14} y={38} textAnchor="end" fill={lastEpReward == null ? "#64748b" : (lastEpReward >= 0 ? "#34d399" : "#f87171")} fontSize="11" fontFamily={font}>
        {lastEpReward == null ? "–" : lastEpReward.toFixed(1)}
      </text>
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
  const WpRef = useRef(mkWp(0.15));
  const WvRef = useRef(mkWv());

  const untrainedState = useRef(envReset());
  const trainedState = useRef(envReset());
  const untrainedEpRewardRef = useRef(0);
  const trainedEpRewardRef = useRef(0);
  const untrainedLastRewardRef = useRef(null);
  const trainedLastRewardRef = useRef(null);
  const untrainedHoldUntilRef = useRef(0);
  const trainedHoldUntilRef = useRef(0);
  const trainedActionBufRef = useRef([]);

  const [, forceRender] = useState(0);
  const [training, setTraining] = useState(false);
  const [iteration, setIteration] = useState(0);
  const [avgReturn, setAvgReturn] = useState(null);
  const [bestReturn, setBestReturn] = useState(null);
  const [bestEverReturn, setBestEverReturn] = useState(null);
  const bestEverRef = useRef(-Infinity);
  const [polLoss, setPolLoss] = useState(null);
  const [valLoss, setValLoss] = useState(null);
  const [kl, setKl] = useState(null);
  const [clipFrac, setClipFrac] = useState(null);
  const [history, setHistory] = useState([]);
  const [lr, setLr] = useState(0.025);
  const [ppoEpochs, setPpoEpochs] = useState(6);
  const [rolloutSteps, setRolloutSteps] = useState(1024);

  const trainingRef = useRef(false);
  const stopFlag = useRef(false);

  useEffect(() => {
    let running = true;
    let lastTick = 0;
    const tick = (ts) => {
      if (!running) return;
      if (ts - lastTick >= 33) {
        lastTick = ts;
        // Untrained: uniform random action
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
        // Trained: sample from softmax policy
        if (trainedHoldUntilRef.current > 0) {
          trainedHoldUntilRef.current -= 1;
          if (trainedHoldUntilRef.current === 0) {
            trainedState.current = envReset();
            trainedEpRewardRef.current = 0;
          }
        } else {
          const probs = policyForward(WpRef.current, obsVec(trainedState.current));
          const a = sampleFromProbs(probs);
          const { state, reward, done } = envStep(trainedState.current, a);
          trainedState.current = state;
          trainedEpRewardRef.current += reward;
          const buf = trainedActionBufRef.current;
          buf.push(a);
          if (buf.length > 200) buf.shift();
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
    WpRef.current = mkWp(0.15);
    WvRef.current = mkWv();
    bestEverRef.current = -Infinity;
    setIteration(0);
    setAvgReturn(null); setBestReturn(null); setBestEverReturn(null);
    setPolLoss(null); setValLoss(null); setKl(null); setClipFrac(null);
    setHistory([]);
    trainedActionBufRef.current = [];
  };

  const resetTrainedEp = () => {
    trainedState.current = envReset();
    trainedHoldUntilRef.current = 0;
    trainedEpRewardRef.current = 0;
  };
  const resetUntrainedEp = () => {
    untrainedState.current = envReset();
    untrainedHoldUntilRef.current = 0;
    untrainedEpRewardRef.current = 0;
  };

  const startTrain = useCallback(async () => {
    if (trainingRef.current) return;
    trainingRef.current = true;
    stopFlag.current = false;
    setTraining(true);
    const GAMMA = 0.99;
    const LAMBDA = 0.95;
    const CLIP_EPS = 0.2;
    const VALUE_COEF = 0.5;
    const ENTROPY_COEF = 0.01;
    while (trainingRef.current && !stopFlag.current) {
      // Collect rollout
      const { transitions, epReturns, lastV } = collectRollout(WpRef.current, WvRef.current, rolloutSteps);
      // Compute GAE
      const { advantages, returns } = computeAdvantages(transitions, lastV, GAMMA, LAMBDA);
      // Multiple PPO epochs over the same rollout
      let lastStats = null;
      for (let ep = 0; ep < ppoEpochs; ep++) {
        const upd = ppoUpdate(WpRef.current, WvRef.current, transitions, advantages, returns, {
          clipEps: CLIP_EPS, lr, valueCoef: VALUE_COEF, entropyCoef: ENTROPY_COEF,
        });
        WpRef.current = upd.Wp;
        WvRef.current = upd.Wv;
        lastStats = upd;
        // Early-stop if KL gets too big
        if (Math.abs(upd.approxKL) > 0.03) break;
      }
      setIteration((i) => i + 1);
      if (epReturns.length > 0) {
        const avg = epReturns.reduce((a, b) => a + b, 0) / epReturns.length;
        const best = Math.max(...epReturns);
        setAvgReturn(avg);
        setBestReturn(best);
        if (best > bestEverRef.current) {
          bestEverRef.current = best;
          setBestEverReturn(best);
        }
        setHistory((h) => {
          const next = [...h, { avg, best, bestEver: bestEverRef.current }];
          return next.length > 160 ? next.slice(-160) : next;
        });
      }
      if (lastStats) {
        setPolLoss(lastStats.policyLoss);
        setValLoss(lastStats.valueLoss);
        setKl(lastStats.approxKL);
        setClipFrac(lastStats.clipFrac);
      }
      await new Promise((r) => setTimeout(r, 8));
    }
    trainingRef.current = false;
    setTraining(false);
  }, [lr, ppoEpochs, rolloutSteps]);

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
          <h1 style={{ fontSize: 22, fontWeight: 500, letterSpacing: "0.5px" }}>Reinforcement Learning · LunarLander (PPO)</h1>
          <p style={{ color: "#94a3b8", fontSize: 12, marginTop: 4, lineHeight: 1.55 }}>
            Simplified LunarLander. Left panel: agent that acts uniformly at random. Right panel: stochastic softmax policy, sampled live from the policy being trained.
            Algorithm: <span style={{ color: "#34d399" }}>Proximal Policy Optimization (PPO)</span> with GAE advantages, a linear policy head and a linear value head.
          </p>
        </header>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 18 }}>
          <div style={panel}>
            <LanderView state={untrainedState.current} subtitle="UNTRAINED" badgeColor="#f87171" bodyColor="#94a3b8" epReward={untrainedEpRewardRef.current} lastEpReward={untrainedLastRewardRef.current} />
            <div style={{ display: "flex", gap: 8, marginTop: 10, alignItems: "center" }}>
              <button onClick={resetUntrainedEp} style={btn(false, "#64748b")}>New episode</button>
              <span style={{ color: "#64748b", fontSize: 10, marginLeft: "auto" }}>uniform random actions</span>
            </div>
          </div>
          <div style={panel}>
            <LanderView state={trainedState.current} subtitle={iteration > 0 ? `TRAINED (iter ${iteration})` : "TRAINED"} badgeColor="#34d399" bodyColor="#38bdf8" epReward={trainedEpRewardRef.current} lastEpReward={trainedLastRewardRef.current} />
            <div style={{ display: "flex", gap: 8, marginTop: 10, alignItems: "center" }}>
              <button onClick={resetTrainedEp} style={btn(false, "#64748b")}>New episode</button>
              <span style={{ color: "#64748b", fontSize: 10, marginLeft: "auto" }}>sampled from softmax policy</span>
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
                <div style={{ fontSize: 10, color: "#64748b", marginBottom: 4 }}>learning rate</div>
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <input type="range" min={0.005} max={0.05} step={0.005} value={lr}
                    onChange={(e) => setLr(parseFloat(e.target.value))}
                    disabled={training} style={{ flex: 1 }} />
                  <span style={{ color: "#67e8f9", fontSize: 11, width: 42, textAlign: "right" }}>{lr.toFixed(3)}</span>
                </div>
              </div>
              <div>
                <div style={{ fontSize: 10, color: "#64748b", marginBottom: 4 }}>rollout steps</div>
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <input type="range" min={256} max={2048} step={128} value={rolloutSteps}
                    onChange={(e) => setRolloutSteps(parseInt(e.target.value, 10))}
                    disabled={training} style={{ flex: 1 }} />
                  <span style={{ color: "#67e8f9", fontSize: 11, width: 38, textAlign: "right" }}>{rolloutSteps}</span>
                </div>
              </div>
              <div>
                <div style={{ fontSize: 10, color: "#64748b", marginBottom: 4 }}>PPO epochs</div>
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <input type="range" min={1} max={10} step={1} value={ppoEpochs}
                    onChange={(e) => setPpoEpochs(parseInt(e.target.value, 10))}
                    disabled={training} style={{ flex: 1 }} />
                  <span style={{ color: "#67e8f9", fontSize: 11, width: 20, textAlign: "right" }}>{ppoEpochs}</span>
                </div>
              </div>
            </div>
            <div style={{ marginTop: 14 }}>
              <div style={{ fontSize: 10, color: "#64748b", marginBottom: 4 }}>
                rollout return · <span style={{ color: "#34d399" }}>best-ever</span> / <span style={{ color: "#38bdf8" }}>iter best</span> / <span style={{ color: "#94a3b8" }}>iter mean</span>
              </div>
              {lossChart || <div style={{ height: 68, border: "1px dashed #1e293b", borderRadius: 4, display: "flex", alignItems: "center", justifyContent: "center", color: "#475569", fontSize: 11 }}>no data yet</div>}
            </div>
          </div>

          <div style={panel}>
            <div style={{ fontSize: 11, color: "#94a3b8", marginBottom: 8 }}>metrics</div>
            <div style={{ fontSize: 11, lineHeight: 1.75 }}>
              <div>iteration: <span style={{ color: "#e2e8f0" }}>{iteration}</span></div>
              <div>best-ever return: <span style={{ color: "#34d399" }}>{bestEverReturn == null ? "–" : bestEverReturn.toFixed(2)}</span></div>
              <div>iter best: <span style={{ color: "#38bdf8" }}>{bestReturn == null ? "–" : bestReturn.toFixed(2)}</span></div>
              <div>iter mean: <span style={{ color: "#94a3b8" }}>{avgReturn == null ? "–" : avgReturn.toFixed(2)}</span></div>
              <div style={{ marginTop: 6, paddingTop: 6, borderTop: "1px solid #1e293b" }}>policy loss: <span style={{ color: "#f472b6" }}>{polLoss == null ? "–" : polLoss.toFixed(4)}</span></div>
              <div>value loss:&nbsp; <span style={{ color: "#fbbf24" }}>{valLoss == null ? "–" : valLoss.toFixed(3)}</span></div>
              <div>approx KL:&nbsp; <span style={{ color: "#a78bfa" }}>{kl == null ? "–" : kl.toFixed(4)}</span></div>
              <div>clip frac:&nbsp; <span style={{ color: "#67e8f9" }}>{clipFrac == null ? "–" : (clipFrac * 100).toFixed(1) + "%"}</span></div>
            </div>
            <div style={{ marginTop: 10, paddingTop: 10, borderTop: "1px solid #1e293b", fontSize: 11, color: "#94a3b8", lineHeight: 1.6 }}>
              <div style={{ color: "#fbbf24", fontSize: 10, marginBottom: 6 }}>action usage (trained, last ~200 steps)</div>
              {(() => {
                const buf = trainedActionBufRef.current;
                const total = Math.max(1, buf.length);
                const counts = [0, 0, 0];
                for (const a of buf) counts[a] += 1;
                const pct = counts.map((c) => (c / total) * 100);
                const labels = ["main", "rot CCW", "rot CW"];
                const colors = ["#f97316", "#38bdf8", "#a78bfa"];
                return labels.map((lbl, i) => (
                  <div key={i} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 3 }}>
                    <div style={{ width: 56, fontSize: 10, color: colors[i] }}>{lbl}</div>
                    <div style={{ flex: 1, background: "#1e293b", height: 8, borderRadius: 2, overflow: "hidden" }}>
                      <div style={{ width: `${pct[i].toFixed(1)}%`, height: "100%", background: colors[i] }} />
                    </div>
                    <div style={{ width: 36, fontSize: 10, textAlign: "right", color: "#cbd5e1" }}>{pct[i].toFixed(0)}%</div>
                  </div>
                ));
              })()}
            </div>
          </div>
        </div>
        <footer style={{ marginTop: 18, color: "#475569", fontSize: 10, textAlign: "center" }}>
          simplified LunarLander · linear softmax policy · linear value · GAE(γ=0.99, λ=0.95) · clip ε=0.2
        </footer>
      </div>
    </div>
  );
}
