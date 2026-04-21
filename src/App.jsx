import { useState, useRef, useEffect } from "react";

const font = "'JetBrains Mono', 'Fira Code', 'SF Mono', monospace";

// LunarLander world: x in [0, 20], y in [0, 13.33]. Origin at bottom-left.
const WORLD_W = 20;
const WORLD_H = 13.333;

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

function LanderScene({ episode, frameIdx, title, badgeColor, bodyColor }) {
  const pxW = 460, pxH = 340;
  const marginTop = 20, marginBot = 8;

  const frame = episode?.frames[Math.min(frameIdx, episode.frames.length - 1)];
  const terrain = episode?.terrain ?? [];
  const helipad = episode?.helipad;

  // World→pixel
  const toPx = (wx, wy) => {
    const x = (wx / WORLD_W) * pxW;
    const y = pxH - marginBot - (wy / WORLD_H) * (pxH - marginTop - marginBot);
    return [x, y];
  };

  // Ground polygon: from terrain points, close it to bottom of viewport
  let groundPath = "";
  if (terrain.length > 1) {
    const pts = terrain.map(([x, y]) => toPx(x, y));
    groundPath =
      `M${pts[0][0].toFixed(1)},${(pxH).toFixed(1)} ` +
      pts.map((p) => `L${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ") +
      ` L${pts[pts.length - 1][0].toFixed(1)},${(pxH).toFixed(1)} Z`;
  } else {
    // fallback flat ground
    const [, gy] = toPx(0, 1);
    groundPath = `M0,${gy} L${pxW},${gy} L${pxW},${pxH} L0,${pxH} Z`;
  }

  // Helipad
  let padEls = null;
  if (helipad) {
    const [x1, y1] = toPx(helipad.x1, helipad.y);
    const [x2] = toPx(helipad.x2, helipad.y);
    padEls = (
      <g>
        <rect x={x1} y={y1 - 3} width={x2 - x1} height={4} fill="#fbbf24" />
        <line x1={x1} y1={y1 - 3} x2={x1} y2={y1 - 14} stroke="#fbbf24" strokeWidth="1.5" />
        <polygon points={`${x1},${y1 - 14} ${x1 + 8},${y1 - 12} ${x1},${y1 - 10}`} fill="#fbbf24" />
        <line x1={x2} y1={y1 - 3} x2={x2} y2={y1 - 14} stroke="#fbbf24" strokeWidth="1.5" />
        <polygon points={`${x2},${y1 - 14} ${x2 - 8},${y1 - 12} ${x2},${y1 - 10}`} fill="#fbbf24" />
      </g>
    );
  }

  // Lander
  let landerEl = null;
  if (frame) {
    const [lx, ly] = toPx(frame.x, frame.y);
    // LunarLander uses +CCW angle in radians; SVG rotates +CW in degrees
    const deg = (-frame.angle * 180) / Math.PI;
    const act = frame.action;
    const a = act;
    const mainOn = a === 2;
    const leftOn = a === 1; // fires LEFT orientation engine (pushes right-rotate)
    const rightOn = a === 3;
    const legs = frame.legs || [false, false];
    landerEl = (
      <g transform={`translate(${lx.toFixed(2)},${ly.toFixed(2)}) rotate(${deg.toFixed(1)})`}>
        {/* main engine flame */}
        {mainOn && (
          <polygon points="-5,10 5,10 0,26" fill="#f97316" opacity={0.92}>
            <animate attributeName="opacity" values="0.8;1;0.8" dur="0.15s" repeatCount="indefinite" />
          </polygon>
        )}
        {/* side flames */}
        {leftOn && <polygon points="-8,-2 -8,4 -16,1" fill="#f97316" opacity={0.9} />}
        {rightOn && <polygon points="8,-2 8,4 16,1" fill="#f97316" opacity={0.9} />}
        {/* body */}
        <polygon points="-8,10 -10,2 -6,-10 6,-10 10,2 8,10" fill={bodyColor} stroke="#0b1225" strokeWidth="1" />
        {/* window */}
        <circle cx={0} cy={-3} r={2.8} fill="#0ea5e9" opacity={0.9} />
        {/* legs */}
        <line x1={-8} y1={10} x2={-13} y2={15} stroke={legs[0] ? "#34d399" : "#94a3b8"} strokeWidth="1.8" />
        <line x1={8} y1={10} x2={13} y2={15} stroke={legs[1] ? "#34d399" : "#94a3b8"} strokeWidth="1.8" />
        <circle cx={-13} cy={15} r={1.4} fill={legs[0] ? "#34d399" : "#94a3b8"} />
        <circle cx={13} cy={15} r={1.4} fill={legs[1] ? "#34d399" : "#94a3b8"} />
      </g>
    );
  }

  // Stars (decorative)
  const stars = [...Array(28)].map((_, i) => {
    const sx = (i * 137.5) % pxW;
    const sy = (i * 47.3) % (pxH * 0.7);
    const r = i % 4 === 0 ? 0.9 : 0.5;
    return <circle key={i} cx={sx} cy={sy} r={r} fill="#cbd5e1" opacity={0.3 + (i % 3) * 0.15} />;
  });

  // Cumulative reward up to current frame
  let cumR = 0;
  if (episode) {
    for (let i = 0; i <= Math.min(frameIdx, episode.frames.length - 1); i++) {
      cumR += episode.frames[i].reward || 0;
    }
  }

  const atEnd = episode && frameIdx >= episode.frames.length - 1;
  const outcome = atEnd ? (episode.landed ? "LANDED" : "CRASHED") : null;
  const outcomeColor = episode?.landed ? "#34d399" : "#f87171";

  return (
    <svg viewBox={`0 0 ${pxW} ${pxH}`} style={{ width: "100%", display: "block", background: "#070914", borderRadius: 8 }}>
      {stars}
      <path d={groundPath} fill="#1e293b" />
      {padEls}
      {landerEl}
      {/* title badge */}
      <rect x={8} y={8} width={150} height={20} fill="rgba(11, 18, 37, 0.85)" stroke={badgeColor} rx={3} />
      <text x={83} y={22} textAnchor="middle" fill={badgeColor} fontSize="10" fontFamily={font} fontWeight={600}>
        {title}
      </text>
      {/* running reward */}
      <rect x={pxW - 128} y={8} width={120} height={38} fill="rgba(11, 18, 37, 0.85)" stroke="#334155" rx={3} />
      <text x={pxW - 122} y={22} fill="#64748b" fontSize="9" fontFamily={font}>reward (live)</text>
      <text x={pxW - 14} y={22} textAnchor="end" fill={cumR >= 0 ? "#34d399" : "#f87171"} fontSize="11" fontFamily={font} fontWeight={600}>
        {cumR.toFixed(1)}
      </text>
      <text x={pxW - 122} y={38} fill="#64748b" fontSize="9" fontFamily={font}>final</text>
      <text x={pxW - 14} y={38} textAnchor="end" fill={episode ? (episode.total_reward >= 0 ? "#34d399" : "#f87171") : "#64748b"} fontSize="11" fontFamily={font}>
        {episode ? episode.total_reward.toFixed(1) : "–"}
      </text>
      {/* outcome banner */}
      {outcome && (
        <g>
          <rect x={pxW / 2 - 70} y={pxH / 2 - 18} width={140} height={32} fill="rgba(11, 18, 37, 0.9)" stroke={outcomeColor} rx={4} />
          <text x={pxW / 2} y={pxH / 2 + 4} textAnchor="middle" fill={outcomeColor} fontSize="13" fontFamily={font} fontWeight={700}>
            {outcome}
          </text>
        </g>
      )}
    </svg>
  );
}

function useEpisodePlayer(episodes, { fps = 30, holdFrames = 25, enabled = true } = {}) {
  const [epIdx, setEpIdx] = useState(0);
  const [frameIdx, setFrameIdx] = useState(0);
  const holdRef = useRef(0);

  useEffect(() => {
    if (!enabled) return;
    const id = setInterval(() => {
      if (!episodes || episodes.length === 0) return;
      setFrameIdx((f) => {
        const ep = episodes[epIdx];
        if (!ep) return 0;
        if (f < ep.frames.length - 1) return f + 1;
        holdRef.current += 1;
        if (holdRef.current > holdFrames) {
          holdRef.current = 0;
          setEpIdx((i) => (i + 1) % episodes.length);
          return 0;
        }
        return f;
      });
    }, 1000 / fps);
    return () => clearInterval(id);
  }, [epIdx, episodes, fps, holdFrames, enabled]);

  const reset = () => {
    holdRef.current = 0;
    setFrameIdx(0);
  };
  const nextEpisode = () => {
    holdRef.current = 0;
    setFrameIdx(0);
    setEpIdx((i) => (i + 1) % (episodes?.length || 1));
  };
  const restartFromEp1 = () => {
    holdRef.current = 0;
    setEpIdx(0);
    setFrameIdx(0);
  };

  return { epIdx, frameIdx, reset, nextEpisode, restartFromEp1 };
}

export default function App() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [showTrained, setShowTrained] = useState(false);

  useEffect(() => {
    const base = import.meta.env.BASE_URL || "/";
    fetch(`${base}trajectories.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(setData)
      .catch((e) => setError(String(e)));
  }, []);

  const trained = data?.trained ?? [];
  const untrained = data?.untrained ?? [];

  const trainedPlayer = useEpisodePlayer(trained, { enabled: showTrained });
  const untrainedPlayer = useEpisodePlayer(untrained);

  const loadPPO = () => {
    trainedPlayer.restartFromEp1();
    setShowTrained(true);
  };
  const hidePPO = () => setShowTrained(false);

  const trainedEp = trained[trainedPlayer.epIdx];
  const untrainedEp = untrained[untrainedPlayer.epIdx];

  const landedCount = trained.filter((e) => e.landed).length;
  const avgReward = trained.length
    ? trained.reduce((s, e) => s + e.total_reward, 0) / trained.length
    : 0;

  const panel = { background: "#11162a", borderRadius: 8, padding: 14, border: "1px solid #1e293b" };

  return (
    <div style={{ minHeight: "100vh", background: "#0c0f1a", color: "#e2e8f0", fontFamily: font, padding: 18 }}>
      <div style={{ maxWidth: 1280, margin: "0 auto" }}>
        <header style={{ marginBottom: 16 }}>
          <h1 style={{ fontSize: 22, fontWeight: 500, letterSpacing: "0.5px" }}>
            Reinforcement Learning · LunarLander-v3 (PPO)
          </h1>
          <p style={{ color: "#94a3b8", fontSize: 12, marginTop: 4, lineHeight: 1.55 }}>
            A baseline agent that picks uniformly-random actions is always shown on the left.
            Click <strong style={{ color: "#34d399" }}>Load PPO agent</strong> to reveal a
            second panel replaying an agent trained offline with PPO (PyTorch, Gymnasium's
            real Box2D LunarLander) alongside for comparison.
          </p>
        </header>

        {error && (
          <div style={{ ...panel, color: "#f87171", fontSize: 12 }}>
            Failed to load trajectories.json: {error}
          </div>
        )}
        {!data && !error && (
          <div style={{ ...panel, color: "#94a3b8", fontSize: 12 }}>Loading trajectories…</div>
        )}

        {data && (
          <>
            <div style={{ display: "grid", gridTemplateColumns: showTrained ? "1fr 1fr" : "1fr", gap: 18, maxWidth: showTrained ? "none" : 620, margin: showTrained ? "0" : "0 auto" }}>
              <div style={panel}>
                <LanderScene
                  episode={untrainedEp}
                  frameIdx={untrainedPlayer.frameIdx}
                  title={`UNTRAINED · ep ${untrainedPlayer.epIdx + 1}/${untrained.length}`}
                  badgeColor="#f87171"
                  bodyColor="#94a3b8"
                />
                <div style={{ display: "flex", gap: 8, marginTop: 10, alignItems: "center", flexWrap: "wrap" }}>
                  <button onClick={untrainedPlayer.nextEpisode} style={btn(false, "#64748b")}>
                    Next episode
                  </button>
                  <button onClick={untrainedPlayer.reset} style={btn(false, "#64748b")}>
                    Restart
                  </button>
                  {!showTrained && (
                    <button onClick={loadPPO} style={btn(true, "#34d399")}>
                      Load PPO agent →
                    </button>
                  )}
                  <span style={{ color: "#64748b", fontSize: 10, marginLeft: "auto" }}>
                    uniform random actions
                  </span>
                </div>
              </div>

              {showTrained && (
                <div style={panel}>
                  <LanderScene
                    episode={trainedEp}
                    frameIdx={trainedPlayer.frameIdx}
                    title={`PPO TRAINED · ep ${trainedPlayer.epIdx + 1}/${trained.length}`}
                    badgeColor="#34d399"
                    bodyColor="#38bdf8"
                  />
                  <div style={{ display: "flex", gap: 8, marginTop: 10, alignItems: "center", flexWrap: "wrap" }}>
                    <button onClick={trainedPlayer.nextEpisode} style={btn(false, "#64748b")}>
                      Next episode
                    </button>
                    <button onClick={trainedPlayer.reset} style={btn(false, "#64748b")}>
                      Restart
                    </button>
                    <button onClick={hidePPO} style={btn(false, "#64748b")}>
                      Hide
                    </button>
                    <span style={{ color: "#64748b", fontSize: 10, marginLeft: "auto" }}>
                      PPO policy · deterministic argmax
                    </span>
                  </div>
                </div>
              )}
            </div>

            <div style={{ display: "grid", gridTemplateColumns: showTrained ? "1fr 1fr 1fr" : "1fr", gap: 18, marginTop: 18, maxWidth: showTrained ? "none" : 620, margin: showTrained ? "18px 0 0" : "18px auto 0" }}>
              {showTrained && (
                <div style={panel}>
                  <div style={{ fontSize: 11, color: "#94a3b8", marginBottom: 8 }}>trained evaluation</div>
                  <div style={{ fontSize: 12, lineHeight: 1.8 }}>
                    <div>episodes recorded: <span style={{ color: "#e2e8f0" }}>{trained.length}</span></div>
                    <div>successful landings: <span style={{ color: "#34d399" }}>{landedCount} / {trained.length}</span></div>
                    <div>avg reward: <span style={{ color: "#67e8f9" }}>{avgReward.toFixed(1)}</span></div>
                    <div style={{ color: "#64748b", fontSize: 10, marginTop: 4 }}>
                      LunarLander is considered "solved" at ≥200 mean reward.
                    </div>
                  </div>
                </div>
              )}

              {showTrained && (
                <div style={panel}>
                  <div style={{ fontSize: 11, color: "#94a3b8", marginBottom: 8 }}>episode reward table</div>
                  <div style={{ fontSize: 11, lineHeight: 1.6 }}>
                    <div style={{ display: "grid", gridTemplateColumns: "40px 70px 70px 1fr", color: "#64748b", fontSize: 10, borderBottom: "1px solid #1e293b", paddingBottom: 4, marginBottom: 4 }}>
                      <div>#</div><div>trained</div><div>random</div><div></div>
                    </div>
                    {[...Array(Math.max(trained.length, untrained.length))].map((_, i) => (
                      <div key={i} style={{ display: "grid", gridTemplateColumns: "40px 70px 70px 1fr", padding: "2px 0" }}>
                        <div style={{ color: "#64748b" }}>{i + 1}</div>
                        <div style={{ color: trained[i] ? (trained[i].landed ? "#34d399" : "#f87171") : "#475569" }}>
                          {trained[i] ? trained[i].total_reward.toFixed(0) : "–"}
                        </div>
                        <div style={{ color: untrained[i] ? (untrained[i].landed ? "#34d399" : "#f87171") : "#475569" }}>
                          {untrained[i] ? untrained[i].total_reward.toFixed(0) : "–"}
                        </div>
                        <div style={{ color: trained[i]?.landed ? "#34d399" : "#64748b", fontSize: 10 }}>
                          {trained[i]?.landed ? "landed" : ""}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div style={panel}>
                <div style={{ fontSize: 11, color: "#94a3b8", marginBottom: 8 }}>legend · actions</div>
                <div style={{ fontSize: 11, lineHeight: 1.6 }}>
                  <div>0 — do nothing</div>
                  <div>1 — fire left orientation engine</div>
                  <div>2 — fire main engine</div>
                  <div>3 — fire right orientation engine</div>
                </div>
                <div style={{ marginTop: 10, paddingTop: 10, borderTop: "1px solid #1e293b", fontSize: 11, color: "#94a3b8", lineHeight: 1.6 }}>
                  <div style={{ color: "#fbbf24", fontSize: 10, marginBottom: 4 }}>visual cues</div>
                  <div>orange flame below → main engine</div>
                  <div>small side flame → orientation engine</div>
                  <div>green leg → ground contact</div>
                  <div>yellow bar with flags → landing pad</div>
                </div>
              </div>
            </div>
          </>
        )}

        <footer style={{ marginTop: 18, color: "#475569", fontSize: 10, textAlign: "center" }}>
          training: PyTorch PPO (2-layer MLP 64, GAE γ=0.99 λ=0.95, clip ε=0.2) on Gymnasium
          LunarLander-v3 · replay: state-by-state playback of the trained agent
        </footer>
      </div>
    </div>
  );
}
