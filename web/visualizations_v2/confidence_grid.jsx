import { useState, useMemo, useCallback, useRef, useEffect } from "react";

// ============================================================================
// Demo Data Generator — creates realistic synthetic traces
// ============================================================================

function generateDemoData() {
  // Shakespeare-like fragments used to build demo text
  const fragments = [
    "First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\n",
    "First Citizen:\nYou are all resolved rather to die than to famish?\n\nAll:\nResolved. resolved.\n\n",
    "First Citizen:\nFirst, you know Caius Marcius is chief enemy to the people.\n\nAll:\nWe know't, we know't.\n\n",
    "KING RICHARD II:\nOld John of Gaunt, time-honour'd Lancaster,\nHast thou, according to thy oath and band,\n",
    "ROMEO:\nBut, soft! what light through yonder window breaks?\nIt is the east, and Juliet is the sun.\n",
    "HAMLET:\nTo be, or not to be, that is the question:\nWhether 'tis nobler in the mind to suffer\n",
    "PROSPERO:\nOur revels now are ended. These our actors,\nAs I foretold you, were all spirits and\n",
    "MACBETH:\nIs this a dagger which I see before me,\nThe handle toward my hand? Come, let me clutch thee.\n",
  ];

  const highConfChars = new Set([" ", "\n", ".", ",", ":", ";", "!", "?", "'", "e", "t", "a", "o", "i", "n", "s", "h", "r"]);
  const medConfChars = new Set(["d", "l", "u", "c", "m", "f", "w", "g", "y", "p", "b"]);

  function charConfidence(ch, pos, totalLen) {
    // Simulate realistic confidence patterns
    let base;
    if (ch === " " || ch === "\n") base = 0.7 + Math.random() * 0.25;
    else if (ch === "." || ch === "," || ch === ":" || ch === ";") base = 0.5 + Math.random() * 0.35;
    else if (highConfChars.has(ch.toLowerCase())) base = 0.3 + Math.random() * 0.5;
    else if (medConfChars.has(ch.toLowerCase())) base = 0.15 + Math.random() * 0.45;
    else base = 0.05 + Math.random() * 0.35;

    // Uppercase chars at start of names get moderate confidence
    if (ch === ch.toUpperCase() && ch !== ch.toLowerCase()) {
      base = Math.max(base, 0.2 + Math.random() * 0.4);
    }
    return Math.min(1.0, Math.max(0.01, base));
  }

  const traces = [];

  // Generate GPT traces
  for (let f = 0; f < 3; f++) {
    const fullText = fragments[f % fragments.length] + fragments[(f + 3) % fragments.length];
    const promptLen = Math.min(32, Math.floor(fullText.length * 0.15));
    const tokens = [];

    for (let i = 0; i < fullText.length; i++) {
      tokens.push({
        token: fullText[i],
        confidence: i < promptLen ? 1.0 : charConfidence(fullText[i], i, fullText.length),
        is_prompt: i < promptLen,
        position: i,
      });
    }

    traces.push({
      model: "gpt",
      name: `sample_${f}`,
      prompt: fullText.slice(0, promptLen),
      prompt_length: promptLen,
      full_text: fullText,
      generation_time: 0.5 + Math.random() * 2,
      tokens,
    });
  }

  // Generate Diffusion traces
  for (let f = 0; f < 3; f++) {
    const fullText = fragments[(f + 1) % fragments.length] + fragments[(f + 4) % fragments.length];
    const promptLen = Math.min(32, Math.floor(fullText.length * 0.15));
    const numSteps = 40;
    const tokens = [];

    for (let i = 0; i < fullText.length; i++) {
      const isPrompt = i < promptLen;
      // Diffusion: confident tokens get revealed early (low step number)
      const conf = isPrompt ? 1.0 : charConfidence(fullText[i], i, fullText.length);
      // More confident tokens are revealed in earlier steps
      const revealStep = isPrompt ? -1 : Math.max(1, Math.round((1 - conf) * numSteps * 0.8 + Math.random() * numSteps * 0.2));

      tokens.push({
        token: fullText[i],
        confidence: conf,
        is_prompt: isPrompt,
        position: i,
        reveal_step: revealStep,
        reveal_step_normalized: isPrompt ? 0 : revealStep / numSteps,
      });
    }

    traces.push({
      model: "diffusion",
      name: `sample_${f}`,
      prompt: fullText.slice(0, promptLen),
      prompt_length: promptLen,
      full_text: fullText,
      generation_time: 0.8 + Math.random() * 1.5,
      num_steps: numSteps,
      tokens,
    });
  }

  return { vocab_size: 66, traces };
}

// ============================================================================
// Color Utilities
// ============================================================================

function confidenceToColor(confidence, isPrompt) {
  if (isPrompt) return { bg: "rgba(120, 180, 255, 0.15)", text: "rgba(140, 190, 255, 0.9)", border: "rgba(120, 180, 255, 0.25)" };
  // Map confidence to brightness: 0 = very dark, 1 = bright white
  const v = Math.pow(confidence, 0.6); // gamma curve to spread out low values
  const gray = Math.round(v * 235 + 20);
  const bgAlpha = 0.06 + v * 0.22;
  return {
    bg: `rgba(${gray}, ${gray}, ${gray}, ${bgAlpha})`,
    text: `rgba(${gray}, ${gray}, ${gray}, ${0.4 + v * 0.6})`,
    border: `rgba(${gray}, ${gray}, ${gray}, ${0.08 + v * 0.18})`,
  };
}

function confidenceToHeat(confidence, isPrompt) {
  if (isPrompt) return { bg: "rgba(60, 130, 200, 0.18)", text: "rgba(120, 180, 240, 0.95)", border: "rgba(60, 130, 200, 0.3)" };
  // Cool (dark blue) → Warm (bright amber/white)
  const v = Math.pow(confidence, 0.55);
  const r = Math.round(40 + v * 215);
  const g = Math.round(30 + v * 210);
  const b = Math.round(60 + v * 160);
  const bgAlpha = 0.08 + v * 0.2;
  return {
    bg: `rgba(${r}, ${g}, ${b}, ${bgAlpha})`,
    text: `rgba(${r}, ${g}, ${b}, ${0.45 + v * 0.55})`,
    border: `rgba(${r}, ${g}, ${b}, ${0.1 + v * 0.2})`,
  };
}

const COLOR_SCHEMES = {
  grayscale: { fn: confidenceToColor, label: "Grayscale" },
  heat: { fn: confidenceToHeat, label: "Heat" },
};

// ============================================================================
// Components
// ============================================================================

function TokenBox({ token, confidence, isPrompt, colorFn, isRevealed = true, showTooltip, onHover }) {
  const colors = colorFn(isRevealed ? confidence : 0, isPrompt);
  const displayToken = isRevealed ? (token === "\n" ? "↵" : token === " " ? "·" : token) : "█";
  const isWhitespace = token === " " || token === "\n";

  return (
    <span
      onMouseEnter={onHover}
      style={{
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        padding: isWhitespace ? "2px 3px" : "2px 5px",
        margin: "1.5px",
        borderRadius: "3px",
        background: colors.bg,
        border: `1px solid ${colors.border}`,
        color: colors.text,
        fontSize: "13px",
        fontFamily: "'JetBrains Mono', 'Fira Code', 'SF Mono', 'Cascadia Code', monospace",
        fontWeight: isPrompt ? 500 : 400,
        lineHeight: "1.6",
        cursor: "default",
        transition: "all 0.15s ease",
        minWidth: isWhitespace ? "10px" : "auto",
        letterSpacing: "0.02em",
        opacity: isRevealed ? 1 : 0.3,
      }}
    >
      {displayToken}
    </span>
  );
}

function Legend({ colorFn }) {
  const steps = 20;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "10px", marginTop: "4px" }}>
      <span style={{ fontSize: "11px", color: "#555", fontFamily: "monospace", letterSpacing: "0.05em", textTransform: "uppercase" }}>uncertain</span>
      <div style={{ display: "flex", height: "10px", borderRadius: "5px", overflow: "hidden", flex: "0 0 200px" }}>
        {Array.from({ length: steps }, (_, i) => {
          const conf = i / (steps - 1);
          const c = colorFn(conf, false);
          return <div key={i} style={{ flex: 1, background: c.text }} />;
        })}
      </div>
      <span style={{ fontSize: "11px", color: "#888", fontFamily: "monospace", letterSpacing: "0.05em", textTransform: "uppercase" }}>confident</span>
    </div>
  );
}

function StatsBar({ tokens }) {
  const generated = tokens.filter(t => !t.is_prompt);
  if (generated.length === 0) return null;
  const avgConf = generated.reduce((s, t) => s + t.confidence, 0) / generated.length;
  const minConf = Math.min(...generated.map(t => t.confidence));
  const maxConf = Math.max(...generated.map(t => t.confidence));
  const highConf = generated.filter(t => t.confidence > 0.7).length;
  const lowConf = generated.filter(t => t.confidence < 0.2).length;

  const stats = [
    { label: "tokens", value: generated.length },
    { label: "avg conf", value: avgConf.toFixed(3) },
    { label: "min", value: minConf.toFixed(3) },
    { label: "max", value: maxConf.toFixed(3) },
    { label: ">0.7", value: `${highConf} (${(100 * highConf / generated.length).toFixed(0)}%)` },
    { label: "<0.2", value: `${lowConf} (${(100 * lowConf / generated.length).toFixed(0)}%)` },
  ];

  return (
    <div style={{
      display: "flex", gap: "20px", flexWrap: "wrap",
      padding: "10px 16px", borderRadius: "6px",
      background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)",
      marginBottom: "16px",
    }}>
      {stats.map(s => (
        <div key={s.label} style={{ display: "flex", gap: "6px", alignItems: "baseline" }}>
          <span style={{ fontSize: "10px", color: "#555", fontFamily: "monospace", textTransform: "uppercase", letterSpacing: "0.08em" }}>{s.label}</span>
          <span style={{ fontSize: "13px", color: "#aaa", fontFamily: "'JetBrains Mono', monospace" }}>{s.value}</span>
        </div>
      ))}
    </div>
  );
}

function DiffusionSlider({ numSteps, currentStep, onChange }) {
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: "14px",
      padding: "10px 16px", borderRadius: "6px",
      background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)",
      marginBottom: "16px",
    }}>
      <span style={{ fontSize: "10px", color: "#555", fontFamily: "monospace", textTransform: "uppercase", letterSpacing: "0.08em", whiteSpace: "nowrap" }}>
        diffusion step
      </span>
      <input
        type="range"
        min={0}
        max={numSteps}
        value={currentStep}
        onChange={e => onChange(parseInt(e.target.value))}
        style={{ flex: 1, accentColor: "#6a8cbe" }}
      />
      <span style={{ fontSize: "13px", color: "#aaa", fontFamily: "'JetBrains Mono', monospace", minWidth: "52px", textAlign: "right" }}>
        {currentStep} / {numSteps}
      </span>
    </div>
  );
}

function TokenTooltip({ token, position }) {
  if (!token) return null;
  const topK = token.top_k || [];
  return (
    <div style={{
      position: "fixed", left: position.x + 12, top: position.y - 8,
      background: "#1a1a1e", border: "1px solid rgba(255,255,255,0.12)",
      borderRadius: "8px", padding: "12px 16px", zIndex: 1000,
      boxShadow: "0 8px 30px rgba(0,0,0,0.5)", minWidth: "180px",
      fontFamily: "'JetBrains Mono', monospace",
      pointerEvents: "none",
    }}>
      <div style={{ fontSize: "16px", color: "#eee", marginBottom: "6px" }}>
        "{token.token === " " ? "·" : token.token === "\n" ? "↵" : token.token}"
        <span style={{ fontSize: "11px", color: "#666", marginLeft: "8px" }}>pos {token.position}</span>
      </div>
      <div style={{ fontSize: "11px", color: "#888", marginBottom: "4px" }}>
        confidence: <span style={{ color: "#ccc" }}>{(token.confidence * 100).toFixed(1)}%</span>
      </div>
      {token.reveal_step !== undefined && token.reveal_step > 0 && (
        <div style={{ fontSize: "11px", color: "#888", marginBottom: "4px" }}>
          revealed at step: <span style={{ color: "#ccc" }}>{token.reveal_step}</span>
        </div>
      )}
      {topK.length > 0 && (
        <div style={{ marginTop: "8px", borderTop: "1px solid rgba(255,255,255,0.06)", paddingTop: "8px" }}>
          <div style={{ fontSize: "10px", color: "#555", marginBottom: "4px", textTransform: "uppercase", letterSpacing: "0.1em" }}>top predictions</div>
          {topK.slice(0, 5).map((k, i) => (
            <div key={i} style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", padding: "1px 0" }}>
              <span style={{ color: i === 0 ? "#bbb" : "#666" }}>"{k.token === " " ? "·" : k.token}"</span>
              <span style={{ color: "#555" }}>{(k.prob * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Main App
// ============================================================================

export default function ConfidenceGrid() {
  const [data, setData] = useState(() => generateDemoData());
  const [selectedModel, setSelectedModel] = useState("gpt");
  const [selectedSample, setSelectedSample] = useState(0);
  const [colorScheme, setColorScheme] = useState("grayscale");
  const [diffusionStep, setDiffusionStep] = useState(40);
  const [hoveredToken, setHoveredToken] = useState(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
  const [showLoadPanel, setShowLoadPanel] = useState(false);
  const [isAnimating, setIsAnimating] = useState(false);
  const animRef = useRef(null);
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);

  // Filter traces by selected model
  const modelTraces = useMemo(() =>
    data.traces.filter(t => t.model === selectedModel),
    [data, selectedModel]
  );

  const currentTrace = modelTraces[selectedSample] || modelTraces[0];
  const colorFn = COLOR_SCHEMES[colorScheme].fn;

  // For diffusion: determine which tokens are revealed at current step
  const visibleTokens = useMemo(() => {
    if (!currentTrace) return [];
    if (currentTrace.model !== "diffusion") return currentTrace.tokens;

    const numSteps = currentTrace.num_steps || 40;
    return currentTrace.tokens.map(t => ({
      ...t,
      _revealed: t.is_prompt || t.reveal_step <= diffusionStep,
    }));
  }, [currentTrace, diffusionStep]);

  // Animate diffusion unmasking
  const animateDiffusion = useCallback(() => {
    if (!currentTrace || currentTrace.model !== "diffusion") return;
    const numSteps = currentTrace.num_steps || 40;
    setDiffusionStep(0);
    setIsAnimating(true);

    let step = 0;
    const tick = () => {
      step++;
      setDiffusionStep(step);
      if (step < numSteps) {
        animRef.current = requestAnimationFrame(() => setTimeout(tick, 60));
      } else {
        setIsAnimating(false);
      }
    };
    animRef.current = requestAnimationFrame(() => setTimeout(tick, 200));
  }, [currentTrace]);

  useEffect(() => {
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current); };
  }, []);

  // Reset sample index when switching models
  useEffect(() => { setSelectedSample(0); }, [selectedModel]);

  // Reset diffusion step when switching samples
  useEffect(() => {
    if (currentTrace?.model === "diffusion") {
      setDiffusionStep(currentTrace.num_steps || 40);
    }
  }, [currentTrace]);

  const handleLoadJSON = useCallback((jsonStr) => {
    try {
      const parsed = JSON.parse(jsonStr);
      if (parsed.traces && Array.isArray(parsed.traces)) {
        setData(parsed);
        setSelectedSample(0);
        setShowLoadPanel(false);
        // Auto-select first available model
        const models = [...new Set(parsed.traces.map(t => t.model))];
        if (models.length > 0) setSelectedModel(models[0]);
      } else {
        alert("Invalid format: expected { traces: [...] }");
      }
    } catch (e) {
      alert("Invalid JSON: " + e.message);
    }
  }, []);

  const handleFileUpload = useCallback((e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => handleLoadJSON(ev.target.result);
    reader.readAsText(file);
  }, [handleLoadJSON]);

  const handleTokenHover = useCallback((token, e) => {
    setHoveredToken(token);
    setTooltipPos({ x: e.clientX, y: e.clientY });
  }, []);

  const availableModels = useMemo(() =>
    [...new Set(data.traces.map(t => t.model))],
    [data]
  );

  return (
    <div style={{
      minHeight: "100vh",
      background: "#0c0c0f",
      color: "#ccc",
      fontFamily: "'JetBrains Mono', 'Fira Code', 'SF Mono', monospace",
      padding: "0",
    }}>
      {/* Header */}
      <div style={{
        padding: "28px 32px 20px",
        borderBottom: "1px solid rgba(255,255,255,0.05)",
      }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "16px" }}>
          <div>
            <h1 style={{
              fontSize: "15px", fontWeight: 600, color: "#e0e0e0",
              margin: 0, letterSpacing: "0.12em", textTransform: "uppercase",
            }}>
              Token Confidence
            </h1>
            <p style={{ fontSize: "12px", color: "#4a4a52", margin: "6px 0 0", letterSpacing: "0.04em" }}>
              Per-token prediction probability — lighter = more confident, darker = more uncertain
            </p>
          </div>

          <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
            {/* Color scheme toggle */}
            <div style={{ display: "flex", gap: "2px", background: "rgba(255,255,255,0.03)", borderRadius: "5px", padding: "2px" }}>
              {Object.entries(COLOR_SCHEMES).map(([key, { label }]) => (
                <button
                  key={key}
                  onClick={() => setColorScheme(key)}
                  style={{
                    padding: "4px 10px", fontSize: "10px", border: "none", cursor: "pointer",
                    background: colorScheme === key ? "rgba(255,255,255,0.08)" : "transparent",
                    color: colorScheme === key ? "#bbb" : "#555",
                    borderRadius: "3px", fontFamily: "inherit",
                    textTransform: "uppercase", letterSpacing: "0.08em",
                  }}
                >
                  {label}
                </button>
              ))}
            </div>

            {/* Load data button */}
            <button
              onClick={() => setShowLoadPanel(!showLoadPanel)}
              style={{
                padding: "4px 12px", fontSize: "10px", border: "1px solid rgba(255,255,255,0.08)",
                background: showLoadPanel ? "rgba(255,255,255,0.06)" : "transparent",
                color: "#666", borderRadius: "4px", cursor: "pointer",
                fontFamily: "inherit", textTransform: "uppercase", letterSpacing: "0.08em",
              }}
            >
              load data
            </button>
          </div>
        </div>

        {/* Load panel */}
        {showLoadPanel && (
          <div style={{
            marginTop: "16px", padding: "16px", borderRadius: "6px",
            background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)",
          }}>
            <div style={{ fontSize: "11px", color: "#666", marginBottom: "10px" }}>
              Load <span style={{ color: "#888" }}>combined_traces.json</span> from generate_traces.py
            </div>
            <div style={{ display: "flex", gap: "10px", alignItems: "flex-start" }}>
              <textarea
                ref={textareaRef}
                placeholder='Paste JSON here... { "traces": [...] }'
                style={{
                  flex: 1, height: "80px", background: "#111114", border: "1px solid rgba(255,255,255,0.06)",
                  borderRadius: "4px", padding: "10px", color: "#888", fontSize: "11px",
                  fontFamily: "inherit", resize: "vertical",
                }}
              />
              <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                <button
                  onClick={() => textareaRef.current && handleLoadJSON(textareaRef.current.value)}
                  style={{
                    padding: "8px 16px", fontSize: "10px", border: "1px solid rgba(255,255,255,0.1)",
                    background: "rgba(255,255,255,0.04)", color: "#888", borderRadius: "4px",
                    cursor: "pointer", fontFamily: "inherit", textTransform: "uppercase",
                    letterSpacing: "0.08em",
                  }}
                >
                  Load
                </button>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  style={{
                    padding: "8px 16px", fontSize: "10px", border: "1px solid rgba(255,255,255,0.1)",
                    background: "rgba(255,255,255,0.04)", color: "#888", borderRadius: "4px",
                    cursor: "pointer", fontFamily: "inherit", textTransform: "uppercase",
                    letterSpacing: "0.08em",
                  }}
                >
                  File
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".json"
                  onChange={handleFileUpload}
                  style={{ display: "none" }}
                />
              </div>
            </div>
          </div>
        )}

        {/* Model & Sample selector */}
        <div style={{ display: "flex", gap: "16px", alignItems: "center", marginTop: "20px", flexWrap: "wrap" }}>
          {/* Model tabs */}
          <div style={{ display: "flex", gap: "2px", background: "rgba(255,255,255,0.03)", borderRadius: "6px", padding: "3px" }}>
            {availableModels.map(model => (
              <button
                key={model}
                onClick={() => setSelectedModel(model)}
                style={{
                  padding: "7px 18px", fontSize: "12px", border: "none", cursor: "pointer",
                  background: selectedModel === model ? "rgba(255,255,255,0.08)" : "transparent",
                  color: selectedModel === model ? "#ddd" : "#555",
                  borderRadius: "4px", fontFamily: "inherit", fontWeight: 500,
                  textTransform: "uppercase", letterSpacing: "0.1em",
                  transition: "all 0.15s ease",
                }}
              >
                {model}
              </button>
            ))}
          </div>

          {/* Sample selector */}
          {modelTraces.length > 1 && (
            <div style={{ display: "flex", gap: "2px", background: "rgba(255,255,255,0.03)", borderRadius: "5px", padding: "2px" }}>
              {modelTraces.map((trace, i) => (
                <button
                  key={i}
                  onClick={() => setSelectedSample(i)}
                  style={{
                    padding: "5px 12px", fontSize: "10px", border: "none", cursor: "pointer",
                    background: selectedSample === i ? "rgba(255,255,255,0.08)" : "transparent",
                    color: selectedSample === i ? "#bbb" : "#555",
                    borderRadius: "3px", fontFamily: "inherit",
                    letterSpacing: "0.04em",
                  }}
                >
                  {trace.name || `sample ${i}`}
                </button>
              ))}
            </div>
          )}

          {/* Diffusion animate button */}
          {currentTrace?.model === "diffusion" && (
            <button
              onClick={isAnimating ? () => { cancelAnimationFrame(animRef.current); setIsAnimating(false); } : animateDiffusion}
              style={{
                padding: "5px 14px", fontSize: "10px",
                border: "1px solid rgba(100, 160, 255, 0.2)",
                background: isAnimating ? "rgba(100, 160, 255, 0.1)" : "rgba(100, 160, 255, 0.05)",
                color: isAnimating ? "#8ab4f0" : "#6a8cbe",
                borderRadius: "4px", cursor: "pointer", fontFamily: "inherit",
                textTransform: "uppercase", letterSpacing: "0.08em",
              }}
            >
              {isAnimating ? "■ stop" : "▶ animate"}
            </button>
          )}

          <Legend colorFn={colorFn} />
        </div>
      </div>

      {/* Content */}
      {currentTrace && (
        <div style={{ padding: "24px 32px" }}>
          {/* Prompt indicator */}
          <div style={{
            fontSize: "11px", color: "#444", marginBottom: "12px",
            letterSpacing: "0.04em",
          }}>
            <span style={{ color: "#555", textTransform: "uppercase", letterSpacing: "0.1em", fontSize: "10px" }}>prompt </span>
            <span style={{ color: "#5a7a9e" }}>
              {currentTrace.prompt.length > 80 ? `"${currentTrace.prompt.slice(0, 80)}..."` : `"${currentTrace.prompt}"`}
            </span>
            <span style={{ color: "#333", marginLeft: "12px" }}>
              {currentTrace.prompt_length} chars → {(currentTrace.tokens?.length || 0) - (currentTrace.prompt_length || 0)} generated
            </span>
            {currentTrace.generation_time && (
              <span style={{ color: "#333", marginLeft: "12px" }}>
                {currentTrace.generation_time.toFixed(2)}s
              </span>
            )}
          </div>

          <StatsBar tokens={currentTrace.tokens || []} />

          {/* Diffusion step slider */}
          {currentTrace.model === "diffusion" && (
            <DiffusionSlider
              numSteps={currentTrace.num_steps || 40}
              currentStep={diffusionStep}
              onChange={setDiffusionStep}
            />
          )}

          {/* Token grid */}
          <div
            style={{
              lineHeight: "2.1",
              padding: "20px",
              borderRadius: "8px",
              background: "rgba(255,255,255,0.01)",
              border: "1px solid rgba(255,255,255,0.04)",
              position: "relative",
            }}
            onMouseLeave={() => setHoveredToken(null)}
          >
            {(currentTrace.model === "diffusion" ? visibleTokens : currentTrace.tokens)?.map((t, i) => {
              const isRevealed = currentTrace.model === "diffusion" ? t._revealed !== false : true;
              return (
                <TokenBox
                  key={i}
                  token={t.token}
                  confidence={t.confidence}
                  isPrompt={t.is_prompt}
                  colorFn={colorFn}
                  isRevealed={isRevealed}
                  onHover={(e) => handleTokenHover(t, e)}
                />
              );
            })}
          </div>

          {/* Model info footer */}
          <div style={{
            marginTop: "16px", display: "flex", gap: "24px", fontSize: "10px",
            color: "#333", letterSpacing: "0.06em",
          }}>
            <span>model: {currentTrace.model}</span>
            {currentTrace.model === "diffusion" && <span>steps: {currentTrace.num_steps}</span>}
            <span>
              {currentTrace.model === "gpt"
                ? "sequential left-to-right generation"
                : "parallel iterative unmasking"
              }
            </span>
          </div>
        </div>
      )}

      {/* Tooltip */}
      {hoveredToken && (
        <TokenTooltip token={hoveredToken} position={tooltipPos} />
      )}
    </div>
  );
}
