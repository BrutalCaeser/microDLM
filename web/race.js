/* ============================================================================
   race.js — MicroDiffusion LM Race Animation Engine
   ============================================================================
   Loads pre-computed frames from frames.json and animates the side-by-side
   diffusion vs GPT text generation race in the browser.
   ============================================================================ */

(function () {
  "use strict";

  // ─── State ───
  let framesData = null;
  let itos = null;
  let meta = null;
  let animationId = null;
  let isRunning = false;

  // ─── DOM refs ───
  const btnGenerate     = document.getElementById("btn-generate");
  const btnText         = btnGenerate.querySelector(".btn-text");
  const btnIcon         = btnGenerate.querySelector(".btn-icon");
  const speedSlider     = document.getElementById("speed");
  const speedLabel      = document.getElementById("speed-label");
  const diffOutput      = document.getElementById("diff-output");
  const gptOutput       = document.getElementById("gpt-output");
  const diffProgress    = document.getElementById("diff-progress");
  const gptProgress     = document.getElementById("gpt-progress");
  const diffProgressTxt = document.getElementById("diff-progress-text");
  const gptProgressTxt  = document.getElementById("gpt-progress-text");
  const diffStatus      = document.getElementById("diff-status");
  const gptStatus       = document.getElementById("gpt-status");
  const speedupBanner   = document.getElementById("speedup-banner");
  const speedupText     = document.getElementById("speedup-text");
  const metaInfo        = document.getElementById("meta-info");
  const diffStepCount   = document.getElementById("diff-step-count");
  const gptStepCount    = document.getElementById("gpt-step-count");

  // ─── Load frames.json ───
  fetch("frames.json")
    .then((r) => {
      if (!r.ok) throw new Error("frames.json not found");
      return r.json();
    })
    .then((data) => {
      framesData = data;
      itos = data.itos;
      meta = data.meta;

      btnGenerate.disabled = false;
      btnText.textContent = "Generate";
      btnIcon.textContent = "▶";

      metaInfo.textContent =
        `${meta.architecture} · ${meta.params} params · vocab ${meta.vocabSize} · seed ${meta.seed}`;
      diffStepCount.textContent = `${data.diffusion.totalFrames} steps`;
      gptStepCount.textContent = `${data.gpt.totalFrames} steps`;
    })
    .catch((err) => {
      btnText.textContent = "Error loading data";
      console.error(err);
    });

  // ─── Speed slider ───
  speedSlider.addEventListener("input", () => {
    speedLabel.textContent = `${speedSlider.value}ms`;
  });

  // ─── Button click ───
  btnGenerate.addEventListener("click", () => {
    if (isRunning) {
      stopRace();
    } else {
      startRace();
    }
  });

  // ─── Decode token array to string ───
  function decodeChar(tokenId) {
    return itos[String(tokenId)] || "?";
  }

  // ─── Render diffusion frame ───
  function renderDiffusionFrame(frame, promptLen, totalGen, maskTokenId) {
    const tokens = frame.tokens;
    const newly = new Set(frame.newly || []);
    const remaining = frame.remaining;
    const revealed = totalGen - remaining;
    const pct = totalGen > 0 ? revealed / totalGen : 1;

    // Progress
    diffProgress.style.width = `${(pct * 100).toFixed(1)}%`;
    diffProgressTxt.textContent = `${revealed} / ${totalGen}`;

    // Status
    if (remaining === 0) {
      diffStatus.textContent = `✓ Done · ${frame.step} steps`;
      diffStatus.classList.add("done");
    } else {
      diffStatus.textContent = `Step ${frame.step}`;
      diffStatus.classList.remove("done");
    }

    // Build HTML
    const parts = [];
    for (let i = 0; i < tokens.length; i++) {
      const ch = decodeChar(tokens[i]);
      const escaped = escapeHtml(ch);
      if (i < promptLen) {
        parts.push(`<span class="tok-prompt">${escaped}</span>`);
      } else if (tokens[i] === maskTokenId) {
        parts.push(`<span class="tok-mask">·</span>`);
      } else if (newly.has(i)) {
        parts.push(`<span class="tok-new">${escaped}</span>`);
      } else {
        parts.push(`<span class="tok-revealed">${escaped}</span>`);
      }
    }
    diffOutput.innerHTML = parts.join("");
  }

  // ─── Render GPT frame ───
  function renderGptFrame(frame, promptLen, totalGen) {
    const tokens = frame.tokens;
    const newPos = frame.newPos;
    const nGen = tokens.length - promptLen;
    const pct = totalGen > 0 ? nGen / totalGen : 0;

    // Progress
    gptProgress.style.width = `${(pct * 100).toFixed(1)}%`;
    gptProgressTxt.textContent = `${nGen} / ${totalGen}`;

    // Status
    if (nGen >= totalGen) {
      gptStatus.textContent = `✓ Done · ${tokens.length - promptLen} tokens`;
      gptStatus.classList.add("done");
    } else {
      gptStatus.textContent = `Token ${frame.step}`;
      gptStatus.classList.remove("done");
    }

    // Build HTML
    const parts = [];
    for (let i = 0; i < tokens.length; i++) {
      const ch = decodeChar(tokens[i]);
      const escaped = escapeHtml(ch);
      if (i < promptLen) {
        parts.push(`<span class="tok-prompt">${escaped}</span>`);
      } else if (i === newPos) {
        parts.push(`<span class="tok-gpt-new">${escaped}</span>`);
      } else {
        parts.push(`<span class="tok-gpt-gen">${escaped}</span>`);
      }
    }
    gptOutput.innerHTML = parts.join("");
  }

  // ─── Race animation ───
  function startRace() {
    if (!framesData) return;

    isRunning = true;
    btnText.textContent = "Stop";
    btnIcon.textContent = "■";
    btnGenerate.classList.add("running");
    speedupBanner.hidden = true;

    const diffFrames = framesData.diffusion.frames;
    const gptFrames = framesData.gpt.frames;
    const promptLen = meta.promptLen;
    const totalGen = meta.totalGen;
    const maskTokenId = meta.maskTokenId;
    const maxFrames = Math.max(diffFrames.length, gptFrames.length);

    let frameIdx = 0;
    let diffDoneShown = false;

    function tick() {
      if (!isRunning || frameIdx >= maxFrames) {
        finishRace(diffFrames.length, gptFrames.length);
        return;
      }

      // Diffusion
      if (frameIdx < diffFrames.length) {
        renderDiffusionFrame(diffFrames[frameIdx], promptLen, totalGen, maskTokenId);
      } else if (!diffDoneShown) {
        // Diffusion finished — show banner
        diffDoneShown = true;
        const speedup = (gptFrames.length / diffFrames.length).toFixed(0);
        speedupBanner.hidden = false;
        speedupText.textContent =
          `Diffusion finished! ~${speedup}× fewer steps than GPT. GPT still generating…`;
      }

      // GPT
      if (frameIdx < gptFrames.length) {
        renderGptFrame(gptFrames[frameIdx], promptLen, totalGen);
      }

      frameIdx++;
      animationId = setTimeout(tick, parseInt(speedSlider.value));
    }

    // Reset UI
    diffOutput.innerHTML = "";
    gptOutput.innerHTML = "";
    diffProgress.style.width = "0%";
    gptProgress.style.width = "0%";
    diffStatus.textContent = "Running";
    diffStatus.classList.remove("done");
    gptStatus.textContent = "Running";
    gptStatus.classList.remove("done");

    // Start
    tick();
  }

  function stopRace() {
    isRunning = false;
    if (animationId) {
      clearTimeout(animationId);
      animationId = null;
    }
    btnText.textContent = "Generate";
    btnIcon.textContent = "▶";
    btnGenerate.classList.remove("running");
  }

  function finishRace(diffTotal, gptTotal) {
    isRunning = false;
    if (animationId) {
      clearTimeout(animationId);
      animationId = null;
    }
    btnText.textContent = "Generate";
    btnIcon.textContent = "▶";
    btnGenerate.classList.remove("running");

    // Final speedup banner
    const speedup = (gptTotal / diffTotal).toFixed(1);
    speedupBanner.hidden = false;
    speedupText.textContent =
      `Race complete — Diffusion: ${diffTotal} steps vs GPT: ${gptTotal} steps (${speedup}× fewer forward passes)`;
  }

  // ─── Util ───
  function escapeHtml(str) {
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }
})();
