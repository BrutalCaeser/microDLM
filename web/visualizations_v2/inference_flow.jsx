import { useState, useMemo, useCallback, useRef, useEffect } from "react";

/* ==========================================================================
   DEMO DATA — realistic synthetic traces so it works without a backend
   ========================================================================== */

const SHAKESPEARE = [
  "First Citizen:", "\n", "Before we proceed any further, hear me speak.", "\n", "\n",
  "All:", "\n", "Speak, speak.", "\n", "\n",
  "First Citizen:", "\n", "You are all resolved rather to die than to famish?", "\n", "\n",
  "All:", "\n", "Resolved. resolved.", "\n", "\n",
  "First Citizen:", "\n", "First, you know Caius Marcius is chief enemy to the people.",
];

function makeCharTokens(textFragments) {
  return textFragments.join("").split("");
}

const ALL_CHARS = makeCharTokens(SHAKESPEARE);
const COMMON = " etaoinsrhldcumfpgwybvkxjqzETAOINSRHLDCUMFPGWYBVKXJQZ";

function fakePredictions(trueChar, position) {
  const preds = [];
  const trueConf = trueChar === " " || trueChar === "\n"
    ? 0.4 + Math.random() * 0.5
    : trueChar === "." || trueChar === "," || trueChar === ":"
      ? 0.25 + Math.random() * 0.45
      : 0.05 + Math.random() * 0.55;

  preds.push({ token: trueChar, prob: trueConf });
  let remaining = 1 - trueConf;
  for (let i = 0; i < 7; i++) {
    const ch = COMMON[Math.floor(Math.random() * COMMON.length)];
    if (preds.find(p => p.token === ch)) continue;
    const p = remaining * (0.3 + Math.random() * 0.5);
    remaining -= p;
    if (remaining < 0) break;
    preds.push({ token: ch, prob: Math.max(0.001, p) });
  }
  preds.sort((a, b) => b.prob - a.prob);
  return preds;
}

function generateGPTTrace() {
  const promptLen = 16;
  const genLen = 60;
  const chars = ALL_CHARS.slice(0, promptLen + genLen);
  const steps = [];

  for (let i = 0; i < genLen; i++) {
    const contextEnd = promptLen + i;
    const nextChar = chars[contextEnd] || " ";
    const predictions = fakePredictions(nextChar, contextEnd);
    steps.push({
      step: i,
      context: chars.slice(Math.max(0, contextEnd - 40), contextEnd),
      context_start: Math.max(0, contextEnd - 40),
      next_position: contextEnd,
      predictions,
      chosen: { token: nextChar, prob: predictions[0].prob },
    });
  }

  return {
    model: "gpt",
    prompt: chars.slice(0, promptLen).join(""),
    prompt_length: promptLen,
    total_tokens: chars.length,
    steps,
    chars,
  };
}

function generateDiffusionTrace() {
  const promptLen = 16;
  const totalLen = 76;
  const numSteps = 16;
  const chars = ALL_CHARS.slice(0, totalLen);
  const genPositions = [];
  for (let i = promptLen; i < totalLen; i++) genPositions.push(i);

  // Assign reveal steps — confident chars reveal early
  const reveals = {};
  const shuffled = [...genPositions].sort(() => Math.random() - 0.5);
  const perStep = Math.ceil(shuffled.length / numSteps);
  shuffled.forEach((pos, idx) => {
    reveals[pos] = Math.min(numSteps, Math.floor(idx / perStep) + 1);
  });

  const steps = [];
  for (let s = 0; s <= numSteps; s++) {
    const tokenStates = [];
    const unmaskThisStep = [];

    for (let i = 0; i < totalLen; i++) {
      const isPrompt = i < promptLen;
      const revealStep = reveals[i] ?? -1;
      const isMasked = !isPrompt && revealStep > s;
      const justUnmasked = !isPrompt && revealStep === s;
      const conf = isPrompt ? 1.0 : (isMasked ? 0.02 + Math.random() * 0.15 : 0.3 + Math.random() * 0.65);

      if (justUnmasked) unmaskThisStep.push(i);

      tokenStates.push({
        position: i,
        token: isMasked ? "_" : chars[i],
        true_token: chars[i],
        is_masked: isMasked,
        is_prompt: isPrompt,
        just_unmasked: justUnmasked,
        confidence: conf,
        predictions: (justUnmasked || (isMasked && s > 0))
          ? fakePredictions(chars[i], i) : null,
      });
    }
    steps.push({ step: s, n_masked: tokenStates.filter(t => t.is_masked).length, unmasked_this_step: unmaskThisStep, tokens: tokenStates });
  }
  return { model: "diffusion", prompt: chars.slice(0, promptLen).join(""), prompt_length: promptLen, total_tokens: totalLen, num_steps: numSteps, steps, chars };
}

/* ==========================================================================
   SVG: ATTENTION PATTERN GRID
   ========================================================================== */

function AttentionGrid({ isCausal, size = 60, cells = 8, accentColor }) {
  const cellSize = size / cells;
  const rects = [];
  for (let r = 0; r < cells; r++) {
    for (let c = 0; c < cells; c++) {
      const active = isCausal ? c <= r : true;
      rects.push(
        <rect key={`${r}-${c}`} x={c * cellSize + 0.5} y={r * cellSize + 0.5} width={cellSize - 1} height={cellSize - 1} rx={1}
          fill={active ? accentColor : "rgba(255,255,255,0.02)"}
          opacity={active ? (0.25 + Math.random() * 0.45) : 0.5} />
      );
    }
  }
  return (
    <svg width={size} height={size} style={{ borderRadius: 4, overflow: "hidden" }}>
      <rect width={size} height={size} fill="rgba(0,0,0,0.3)" rx={4} />
      {rects}
    </svg>
  );
}

/* ==========================================================================
   SVG: MODEL BLOCK — the stylized transformer
   ========================================================================== */

function ModelBlock({ model, isProcessing }) {
  const isGPT = model === "gpt";
  const accent = isGPT ? "#d4915a" : "#5a9abf";
  const accentDim = isGPT ? "rgba(212,145,90,0.12)" : "rgba(90,154,191,0.12)";
  const label = isGPT ? "CAUSAL" : "BIDIRECTIONAL";
  const sublabel = isGPT ? "sees only past tokens" : "sees all positions";

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 10, padding: "12px 16px",
      background: accentDim, border: `1px solid ${isGPT ? "rgba(212,145,90,0.15)" : "rgba(90,154,191,0.15)"}`,
      borderRadius: 10, position: "relative", minWidth: 130 }}>

      {/* Pulse ring when processing */}
      {isProcessing && (
        <div style={{ position: "absolute", inset: -3, borderRadius: 13,
          border: `2px solid ${accent}`, opacity: 0.3,
          animation: "pulseRing 1.2s ease-out infinite" }} />
      )}

      <div style={{ fontSize: 9, color: accent, letterSpacing: "0.18em", textTransform: "uppercase", fontWeight: 600 }}>
        {isGPT ? "GPT" : "DIFFUSION"}
      </div>

      <AttentionGrid isCausal={isGPT} size={64} cells={8} accentColor={accent} />

      <div style={{ fontSize: 9, color: accent, letterSpacing: "0.1em", fontWeight: 500, textTransform: "uppercase" }}>{label}</div>
      <div style={{ fontSize: 9, color: "rgba(255,255,255,0.25)", letterSpacing: "0.04em", textAlign: "center" }}>{sublabel}</div>

      {/* Layer indicators */}
      <div style={{ display: "flex", gap: 3 }}>
        {Array.from({ length: 6 }, (_, i) => (
          <div key={i} style={{ width: 6, height: 6, borderRadius: 2,
            background: isProcessing ? accent : "rgba(255,255,255,0.08)",
            opacity: isProcessing ? 0.4 + (i * 0.1) : 1,
            transition: "all 0.3s ease",
            animationDelay: isProcessing ? `${i * 0.1}s` : undefined }} />
        ))}
      </div>

      <style>{`
        @keyframes pulseRing {
          0% { transform: scale(1); opacity: 0.3; }
          100% { transform: scale(1.06); opacity: 0; }
        }
      `}</style>
    </div>
  );
}

/* ==========================================================================
   PROBABILITY BARS
   ========================================================================== */

function ProbBars({ predictions, chosenToken, accentColor, maxShow = 6, label }) {
  if (!predictions || predictions.length === 0) {
    return (
      <div style={{ padding: 20, color: "rgba(255,255,255,0.15)", fontSize: 11, fontStyle: "italic", textAlign: "center" }}>
        waiting for prediction...
      </div>
    );
  }

  const top = predictions.slice(0, maxShow);
  const maxProb = Math.max(...top.map(p => p.prob), 0.01);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
      {label && (
        <div style={{ fontSize: 9, color: "rgba(255,255,255,0.2)", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 8 }}>
          {label}
        </div>
      )}
      {top.map((pred, i) => {
        const isChosen = chosenToken && pred.token === chosenToken;
        const barWidth = (pred.prob / maxProb) * 100;
        const displayTok = pred.token === " " ? "␣" : pred.token === "\n" ? "↵" : pred.token === "_" ? "▒" : pred.token;

        return (
          <div key={i} style={{
            display: "flex", alignItems: "center", gap: 8, padding: "3px 0",
            opacity: i === 0 ? 1 : 0.5 + 0.5 * (1 - i / maxShow),
          }}>
            <span style={{
              minWidth: 22, textAlign: "center", fontSize: 14,
              fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
              color: isChosen ? accentColor : "#bbb", fontWeight: isChosen ? 700 : 400,
              background: isChosen ? `${accentColor}15` : "transparent",
              borderRadius: 3, padding: "1px 4px",
            }}>
              {displayTok}
            </span>
            <div style={{ flex: 1, height: 14, background: "rgba(255,255,255,0.03)", borderRadius: 3, overflow: "hidden", position: "relative" }}>
              <div style={{
                height: "100%", borderRadius: 3, transition: "width 0.4s ease",
                width: `${barWidth}%`,
                background: isChosen
                  ? `linear-gradient(90deg, ${accentColor}88, ${accentColor}cc)`
                  : `linear-gradient(90deg, rgba(255,255,255,0.08), rgba(255,255,255,0.18))`,
              }} />
            </div>
            <span style={{
              minWidth: 42, textAlign: "right", fontSize: 10,
              fontFamily: "'JetBrains Mono', monospace",
              color: isChosen ? accentColor : "rgba(255,255,255,0.3)",
            }}>
              {(pred.prob * 100).toFixed(1)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}

/* ==========================================================================
   TOKEN CHIPS — for input context and output sequence
   ========================================================================== */

function TokenChip({ char, role, accent, animation }) {
  const isNewline = char === "\n";
  const isSpace = char === " ";
  const isMask = char === "_";
  const display = isNewline ? "↵" : isSpace ? "·" : isMask ? "▒" : char;

  const bgMap = {
    prompt: "rgba(120,180,255,0.1)",
    context: "rgba(255,255,255,0.04)",
    masked: "rgba(255,255,255,0.02)",
    predicted: `${accent}20`,
    justRevealed: `${accent}30`,
  };

  const colorMap = {
    prompt: "rgba(120,180,255,0.7)",
    context: "rgba(255,255,255,0.45)",
    masked: "rgba(255,255,255,0.12)",
    predicted: accent,
    justRevealed: accent,
  };

  const borderMap = {
    prompt: "rgba(120,180,255,0.15)",
    context: "rgba(255,255,255,0.06)",
    masked: "rgba(255,255,255,0.04)",
    predicted: `${accent}50`,
    justRevealed: `${accent}60`,
  };

  return (
    <span style={{
      display: "inline-flex", alignItems: "center", justifyContent: "center",
      padding: "1px 4px", margin: "1.5px", borderRadius: 3,
      background: bgMap[role] || bgMap.context,
      border: `1px solid ${borderMap[role] || borderMap.context}`,
      color: colorMap[role] || colorMap.context,
      fontSize: 12, fontFamily: "'JetBrains Mono','Fira Code',monospace",
      lineHeight: 1.4, minWidth: isMask || isSpace ? 10 : "auto",
      fontWeight: role === "predicted" || role === "justRevealed" ? 600 : 400,
      transition: "all 0.25s ease",
      animation: animation || "none",
    }}>
      {display}
    </span>
  );
}

/* ==========================================================================
   FLOW ARROW — animated connecting line
   ========================================================================== */

function FlowArrow({ active, accentColor }) {
  return (
    <div style={{ display: "flex", alignItems: "center", padding: "0 4px", minWidth: 40 }}>
      <svg width="40" height="20" viewBox="0 0 40 20">
        <defs>
          <linearGradient id={`arrowGrad-${accentColor.replace('#','')}`} x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor={active ? accentColor : "rgba(255,255,255,0.1)"} stopOpacity={0.2} />
            <stop offset="100%" stopColor={active ? accentColor : "rgba(255,255,255,0.1)"} stopOpacity={0.7} />
          </linearGradient>
        </defs>
        <line x1="0" y1="10" x2="32" y2="10" stroke={`url(#arrowGrad-${accentColor.replace('#','')})`} strokeWidth="2" />
        <polygon points="30,5 38,10 30,15" fill={active ? accentColor : "rgba(255,255,255,0.1)"} opacity={active ? 0.6 : 0.3} />
        {active && (
          <circle r="2.5" fill={accentColor} opacity="0.8">
            <animateMotion dur="0.8s" repeatCount="indefinite" path="M0,10 L32,10" />
          </circle>
        )}
      </svg>
    </div>
  );
}

/* ==========================================================================
   PLAYBACK CONTROLS
   ========================================================================== */

function Controls({ step, maxStep, onStep, isPlaying, onTogglePlay, onReset, accentColor }) {
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 6, padding: "6px 10px",
      background: "rgba(255,255,255,0.02)", borderRadius: 6,
      border: "1px solid rgba(255,255,255,0.05)",
    }}>
      <button onClick={onReset} style={btnStyle}>⏮</button>
      <button onClick={() => onStep(Math.max(0, step - 1))} style={btnStyle}>◀</button>
      <button onClick={onTogglePlay} style={{ ...btnStyle, color: isPlaying ? accentColor : "#666", borderColor: isPlaying ? `${accentColor}40` : "rgba(255,255,255,0.08)" }}>
        {isPlaying ? "⏸" : "▶"}
      </button>
      <button onClick={() => onStep(Math.min(maxStep, step + 1))} style={btnStyle}>▶</button>
      <button onClick={() => onStep(maxStep)} style={btnStyle}>⏭</button>

      <div style={{ marginLeft: 8, fontSize: 11, fontFamily: "'JetBrains Mono', monospace", color: "#555", minWidth: 90 }}>
        step <span style={{ color: accentColor }}>{step}</span>
        <span style={{ color: "#333" }}> / {maxStep}</span>
      </div>

      {/* Progress bar */}
      <div style={{ flex: 1, height: 4, background: "rgba(255,255,255,0.04)", borderRadius: 2, overflow: "hidden", minWidth: 80, cursor: "pointer" }}
        onClick={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const pct = (e.clientX - rect.left) / rect.width;
          onStep(Math.round(pct * maxStep));
        }}>
        <div style={{
          height: "100%", borderRadius: 2, transition: "width 0.15s ease",
          width: `${(step / Math.max(1, maxStep)) * 100}%`,
          background: `linear-gradient(90deg, ${accentColor}66, ${accentColor})`,
        }} />
      </div>
    </div>
  );
}

const btnStyle = {
  width: 28, height: 28, display: "flex", alignItems: "center", justifyContent: "center",
  background: "transparent", border: "1px solid rgba(255,255,255,0.08)",
  borderRadius: 4, color: "#666", fontSize: 11, cursor: "pointer",
  fontFamily: "inherit",
};

/* ==========================================================================
   GPT INFERENCE VIEW
   ========================================================================== */

function GPTView({ trace, step }) {
  const accent = "#d4915a";
  const s = trace.steps[step];
  if (!s) return null;

  const outputChars = trace.chars.slice(0, trace.prompt_length + step + 1);
  const isProcessing = step < trace.steps.length - 1;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {/* Main flow */}
      <div style={{ display: "flex", alignItems: "center", gap: 0, flexWrap: "nowrap", justifyContent: "center" }}>
        {/* Input context */}
        <div style={{ flex: "0 1 280px", minWidth: 180 }}>
          <div style={{ fontSize: 9, color: "rgba(255,255,255,0.2)", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 8 }}>
            context window <span style={{ color: "#444" }}>({s.context.length} chars)</span>
          </div>
          <div style={{ padding: "10px 12px", background: "rgba(255,255,255,0.015)", border: "1px solid rgba(255,255,255,0.04)", borderRadius: 6, lineHeight: 1.9 }}>
            {s.context.map((ch, i) => {
              const globalPos = s.context_start + i;
              const isPrompt = globalPos < trace.prompt_length;
              return <TokenChip key={i} char={ch} role={isPrompt ? "prompt" : "context"} accent={accent} />;
            })}
            <span style={{ display: "inline-flex", alignItems: "center", justifyContent: "center",
              padding: "1px 6px", margin: "1.5px", borderRadius: 3,
              background: `${accent}15`, border: `1px dashed ${accent}40`,
              color: accent, fontSize: 12, fontFamily: "'JetBrains Mono', monospace",
              animation: "blink 1s infinite",
            }}>?</span>
          </div>
        </div>

        <FlowArrow active={isProcessing} accentColor={accent} />
        <ModelBlock model="gpt" isProcessing={isProcessing} />
        <FlowArrow active={isProcessing} accentColor={accent} />

        {/* Predictions */}
        <div style={{ flex: "0 1 240px", minWidth: 180 }}>
          <div style={{ fontSize: 9, color: "rgba(255,255,255,0.2)", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 8 }}>
            next token prediction
          </div>
          <div style={{ padding: "10px 14px", background: "rgba(255,255,255,0.015)", border: "1px solid rgba(255,255,255,0.04)", borderRadius: 6 }}>
            <ProbBars predictions={s.predictions} chosenToken={s.chosen.token} accentColor={accent} />
            <div style={{ marginTop: 8, paddingTop: 8, borderTop: "1px solid rgba(255,255,255,0.04)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: 9, color: "#444", letterSpacing: "0.08em", textTransform: "uppercase" }}>sampled</span>
              <span style={{ fontSize: 16, fontFamily: "'JetBrains Mono', monospace", color: accent, fontWeight: 700,
                background: `${accent}15`, padding: "2px 8px", borderRadius: 4 }}>
                {s.chosen.token === " " ? "␣" : s.chosen.token === "\n" ? "↵" : s.chosen.token}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Output sequence */}
      <div>
        <div style={{ fontSize: 9, color: "rgba(255,255,255,0.2)", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 6 }}>
          generated sequence <span style={{ color: "#444" }}>({outputChars.length} chars)</span>
        </div>
        <div style={{ padding: "10px 12px", background: "rgba(255,255,255,0.015)", border: "1px solid rgba(255,255,255,0.04)", borderRadius: 6, lineHeight: 1.9 }}>
          {outputChars.map((ch, i) => {
            const isPrompt = i < trace.prompt_length;
            const isLatest = i === trace.prompt_length + step;
            return <TokenChip key={i} char={ch} role={isPrompt ? "prompt" : isLatest ? "predicted" : "context"} accent={accent} />;
          })}
        </div>
      </div>

      <style>{`@keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }`}</style>
    </div>
  );
}

/* ==========================================================================
   DIFFUSION INFERENCE VIEW
   ========================================================================== */

function DiffusionView({ trace, step }) {
  const accent = "#5a9abf";
  const s = trace.steps[step];
  if (!s) return null;

  const isProcessing = step > 0 && step < trace.steps.length - 1;
  const justUnmasked = s.unmasked_this_step || [];

  // Find the most confident newly-unmasked position for prediction display
  const predPosition = justUnmasked.length > 0
    ? s.tokens.filter(t => t.just_unmasked).sort((a, b) => b.confidence - a.confidence)[0]
    : s.tokens.find(t => t.is_masked && t.predictions);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {/* Main flow */}
      <div style={{ display: "flex", alignItems: "center", gap: 0, flexWrap: "nowrap", justifyContent: "center" }}>
        {/* Masked sequence */}
        <div style={{ flex: "0 1 280px", minWidth: 180 }}>
          <div style={{ fontSize: 9, color: "rgba(255,255,255,0.2)", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 8 }}>
            current state <span style={{ color: "#444" }}>({s.n_masked} masked)</span>
          </div>
          <div style={{ padding: "10px 12px", background: "rgba(255,255,255,0.015)", border: "1px solid rgba(255,255,255,0.04)", borderRadius: 6, lineHeight: 1.9, maxHeight: 200, overflowY: "auto" }}>
            {s.tokens.map((t, i) => (
              <TokenChip
                key={i} char={t.token}
                role={t.is_prompt ? "prompt" : t.just_unmasked ? "justRevealed" : t.is_masked ? "masked" : "context"}
                accent={accent}
              />
            ))}
          </div>
        </div>

        <FlowArrow active={isProcessing} accentColor={accent} />
        <ModelBlock model="diffusion" isProcessing={isProcessing} />
        <FlowArrow active={isProcessing} accentColor={accent} />

        {/* Predictions panel */}
        <div style={{ flex: "0 1 240px", minWidth: 180 }}>
          <div style={{ fontSize: 9, color: "rgba(255,255,255,0.2)", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 8 }}>
            {justUnmasked.length > 0
              ? `unmasked ${justUnmasked.length} tokens this step`
              : step === 0 ? "all tokens masked" : "predictions at masked positions"
            }
          </div>
          <div style={{ padding: "10px 14px", background: "rgba(255,255,255,0.015)", border: "1px solid rgba(255,255,255,0.04)", borderRadius: 6 }}>
            {predPosition?.predictions ? (
              <>
                <div style={{ fontSize: 9, color: "#444", marginBottom: 6, letterSpacing: "0.06em" }}>
                  position {predPosition.position} — confidence {(predPosition.confidence * 100).toFixed(0)}%
                </div>
                <ProbBars
                  predictions={predPosition.predictions}
                  chosenToken={predPosition.true_token || predPosition.token}
                  accentColor={accent}
                />
              </>
            ) : (
              <div style={{ color: "rgba(255,255,255,0.12)", fontSize: 11, padding: "16px 0", textAlign: "center", fontStyle: "italic" }}>
                {step === 0 ? "begin denoising →" : "all tokens revealed"}
              </div>
            )}

            {/* Confidence heatmap for unmasked positions */}
            {justUnmasked.length > 0 && (
              <div style={{ marginTop: 10, paddingTop: 8, borderTop: "1px solid rgba(255,255,255,0.04)" }}>
                <div style={{ fontSize: 9, color: "#444", marginBottom: 6, letterSpacing: "0.06em", textTransform: "uppercase" }}>
                  revealed this step
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 3 }}>
                  {s.tokens.filter(t => t.just_unmasked).sort((a, b) => b.confidence - a.confidence).map((t, i) => (
                    <span key={i} style={{
                      fontSize: 13, fontFamily: "'JetBrains Mono',monospace", fontWeight: 600,
                      padding: "2px 5px", borderRadius: 3,
                      background: `rgba(90,154,191,${0.1 + t.confidence * 0.3})`,
                      color: `rgba(90,154,191,${0.4 + t.confidence * 0.6})`,
                      border: `1px solid rgba(90,154,191,${0.1 + t.confidence * 0.2})`,
                    }}>
                      {t.true_token === " " ? "␣" : t.true_token === "\n" ? "↵" : t.true_token}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Confidence map — all positions */}
      <div>
        <div style={{ fontSize: 9, color: "rgba(255,255,255,0.2)", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 6 }}>
          position confidence map
          <span style={{ color: "#444", marginLeft: 8 }}>bright = confident, dim = uncertain</span>
        </div>
        <div style={{
          display: "flex", gap: 1, padding: "8px 12px",
          background: "rgba(255,255,255,0.015)", border: "1px solid rgba(255,255,255,0.04)",
          borderRadius: 6, overflowX: "auto",
        }}>
          {s.tokens.filter(t => !t.is_prompt).map((t, i) => {
            const v = t.is_masked ? 0.03 : Math.pow(t.confidence, 0.6);
            return (
              <div key={i} title={`pos ${t.position}: ${t.token} (${(t.confidence * 100).toFixed(0)}%)`}
                style={{
                  width: 5, minHeight: 24, borderRadius: 1.5, transition: "all 0.3s ease",
                  background: t.just_unmasked
                    ? accent
                    : t.is_masked
                      ? "rgba(255,255,255,0.03)"
                      : `rgba(90,154,191,${0.15 + v * 0.7})`,
                  opacity: t.is_masked ? 0.4 : 0.6 + v * 0.4,
                }} />
            );
          })}
        </div>
      </div>
    </div>
  );
}

/* ==========================================================================
   MAIN APP
   ========================================================================== */

export default function InferenceFlow() {
  const [model, setModel] = useState("gpt");
  const [step, setStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(300);
  const [showLoadPanel, setShowLoadPanel] = useState(false);
  const timerRef = useRef(null);
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);

  const [gptTrace, setGptTrace] = useState(() => generateGPTTrace());
  const [diffTrace, setDiffTrace] = useState(() => generateDiffusionTrace());

  const currentTrace = model === "gpt" ? gptTrace : diffTrace;
  const maxStep = currentTrace.steps.length - 1;
  const accent = model === "gpt" ? "#d4915a" : "#5a9abf";

  // Reset step on model switch
  useEffect(() => { setStep(0); setIsPlaying(false); }, [model]);

  // Playback timer
  useEffect(() => {
    if (isPlaying) {
      timerRef.current = setInterval(() => {
        setStep(prev => {
          if (prev >= maxStep) { setIsPlaying(false); return prev; }
          return prev + 1;
        });
      }, speed);
    }
    return () => clearInterval(timerRef.current);
  }, [isPlaying, maxStep, speed]);

  const handleLoadJSON = useCallback((jsonStr) => {
    try {
      const parsed = JSON.parse(jsonStr);
      // Support generate_traces.py format: { traces: [...] }
      if (parsed.traces) {
        const gptT = parsed.traces.find(t => t.model === "gpt");
        const diffT = parsed.traces.find(t => t.model === "diffusion");
        if (gptT) setGptTrace(gptT);
        if (diffT) setDiffTrace(diffT);
        setStep(0);
        setShowLoadPanel(false);
      }
    } catch (e) { alert("Invalid JSON: " + e.message); }
  }, []);

  const handleFileUpload = useCallback((e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => handleLoadJSON(ev.target.result);
    reader.readAsText(file);
  }, [handleLoadJSON]);

  return (
    <div style={{
      minHeight: "100vh", background: "#0a0a0d", color: "#ccc",
      fontFamily: "'JetBrains Mono','Fira Code','SF Mono',monospace",
    }}>
      {/* Header */}
      <div style={{ padding: "24px 28px 16px", borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 12 }}>
          <div>
            <h1 style={{ fontSize: 14, fontWeight: 600, color: "#e0e0e0", margin: 0, letterSpacing: "0.12em", textTransform: "uppercase" }}>
              Inference Flow
            </h1>
            <p style={{ fontSize: 11, color: "#3a3a42", margin: "4px 0 0", letterSpacing: "0.04em" }}>
              Step through generation — see what the model sees, predicts, and chooses
            </p>
          </div>
          <div style={{ display: "flex", gap: 6 }}>
            <button onClick={() => setShowLoadPanel(!showLoadPanel)}
              style={{ padding: "4px 12px", fontSize: 9, border: "1px solid rgba(255,255,255,0.08)",
                background: "transparent", color: "#555", borderRadius: 4, cursor: "pointer",
                fontFamily: "inherit", textTransform: "uppercase", letterSpacing: "0.08em" }}>
              load traces
            </button>
          </div>
        </div>

        {showLoadPanel && (
          <div style={{ marginTop: 12, padding: 14, borderRadius: 6, background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)", display: "flex", gap: 10 }}>
            <textarea ref={textareaRef} placeholder="Paste trace JSON..." style={{
              flex: 1, height: 60, background: "#0d0d10", border: "1px solid rgba(255,255,255,0.06)",
              borderRadius: 4, padding: 8, color: "#777", fontSize: 10, fontFamily: "inherit", resize: "vertical" }} />
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <button onClick={() => textareaRef.current && handleLoadJSON(textareaRef.current.value)}
                style={{ ...btnStyle, width: "auto", padding: "6px 14px", fontSize: 9, letterSpacing: "0.08em" }}>Load</button>
              <button onClick={() => fileInputRef.current?.click()}
                style={{ ...btnStyle, width: "auto", padding: "6px 14px", fontSize: 9, letterSpacing: "0.08em" }}>File</button>
              <input ref={fileInputRef} type="file" accept=".json" onChange={handleFileUpload} style={{ display: "none" }} />
            </div>
          </div>
        )}

        {/* Model toggle + controls */}
        <div style={{ display: "flex", gap: 12, alignItems: "center", marginTop: 16, flexWrap: "wrap" }}>
          <div style={{ display: "flex", gap: 2, background: "rgba(255,255,255,0.03)", borderRadius: 6, padding: 3 }}>
            {["gpt", "diffusion"].map(m => (
              <button key={m} onClick={() => setModel(m)}
                style={{
                  padding: "7px 20px", fontSize: 11, border: "none", cursor: "pointer",
                  background: model === m ? "rgba(255,255,255,0.07)" : "transparent",
                  color: model === m ? (m === "gpt" ? "#d4915a" : "#5a9abf") : "#444",
                  borderRadius: 4, fontFamily: "inherit", fontWeight: 600,
                  textTransform: "uppercase", letterSpacing: "0.12em",
                  transition: "all 0.15s ease",
                }}>
                {m}
              </button>
            ))}
          </div>

          <Controls
            step={step} maxStep={maxStep}
            onStep={setStep} isPlaying={isPlaying}
            onTogglePlay={() => setIsPlaying(p => !p)}
            onReset={() => { setStep(0); setIsPlaying(false); }}
            accentColor={accent}
          />

          {/* Speed control */}
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ fontSize: 9, color: "#444", textTransform: "uppercase", letterSpacing: "0.08em" }}>speed</span>
            {[500, 300, 150, 50].map(s => (
              <button key={s} onClick={() => setSpeed(s)}
                style={{
                  ...btnStyle, width: "auto", padding: "4px 8px", fontSize: 9,
                  color: speed === s ? accent : "#444",
                  borderColor: speed === s ? `${accent}40` : "rgba(255,255,255,0.06)",
                }}>
                {s <= 50 ? "4×" : s <= 150 ? "2×" : s <= 300 ? "1×" : "½×"}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main content */}
      <div style={{ padding: "24px 28px" }}>
        {/* Info bar */}
        <div style={{
          display: "flex", gap: 20, marginBottom: 20, fontSize: 10, color: "#333", letterSpacing: "0.06em",
          padding: "8px 14px", background: "rgba(255,255,255,0.015)", borderRadius: 6,
          border: "1px solid rgba(255,255,255,0.03)",
        }}>
          <span>model: <span style={{ color: accent }}>{model}</span></span>
          <span>prompt: <span style={{ color: "#555" }}>{currentTrace.prompt_length} chars</span></span>
          {model === "gpt" && <span>generated: <span style={{ color: "#555" }}>{Math.min(step + 1, currentTrace.steps.length)} / {currentTrace.steps.length} tokens</span></span>}
          {model === "diffusion" && <span>step: <span style={{ color: "#555" }}>{step} / {currentTrace.num_steps || maxStep}</span></span>}
          <span style={{ marginLeft: "auto", color: "#444" }}>
            {model === "gpt"
              ? "sequential: one token at a time, left → right"
              : "parallel: multiple tokens per step, any position"
            }
          </span>
        </div>

        {model === "gpt"
          ? <GPTView trace={currentTrace} step={step} />
          : <DiffusionView trace={currentTrace} step={step} />
        }
      </div>
    </div>
  );
}
