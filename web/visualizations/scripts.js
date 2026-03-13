/* ============================================================================
   MicroDiffusion LM Visualizations — Anime.js Implementation
   Interactive animations showcasing core concepts
   ============================================================================ */

(function () {
  "use strict";

  // ─── DOM Elements ───
  const navButtons = document.querySelectorAll('.nav-btn');
  const visualizations = document.querySelectorAll('.visualization');

  // ─── Navigation ───
  navButtons.forEach(button => {
    button.addEventListener('click', () => {
      const visId = button.dataset.vis;

      // Update active nav button
      navButtons.forEach(btn => btn.classList.remove('active'));
      button.classList.add('active');

      // Show corresponding visualization
      visualizations.forEach(vis => {
        vis.classList.remove('active');
        if (vis.id === `${visId}-vis`) {
          vis.classList.add('active');
          // Initialize visualization if needed
          if (visId === 'masking' && !window.maskingInitialized) {
            initMaskingVisualization();
            window.maskingInitialized = true;
          } else if (visId === 'attention' && !window.attentionInitialized) {
            initAttentionVisualization();
            window.attentionInitialized = true;
          }
        }
      });
    });
  });

  // ─── Token Masking Visualization ───
  function initMaskingVisualization() {
    const grid = document.getElementById('masking-grid');
    const playBtn = document.getElementById('masking-play');
    const resetBtn = document.getElementById('masking-reset');
    const timeSlider = document.getElementById('masking-time');
    const timeValue = document.getElementById('masking-time-value');

    // Create token grid (20x10 = 200 tokens)
    const rows = 10;
    const cols = 20;
    const tokens = [];

    // Clear grid
    grid.innerHTML = '';

    // Create token cells
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const cell = document.createElement('div');
        cell.className = 'token-cell token-normal';
        cell.textContent = String.fromCharCode(97 + (i * cols + j) % 26); // a-z
        cell.dataset.index = i * cols + j;
        grid.appendChild(cell);
        tokens.push(cell);
      }
    }

    // Cosine schedule function: α(t) = cos(πt/2)²
    function cosineSchedule(t) {
      return Math.pow(Math.cos(Math.PI * t / 2), 2);
    }

    // Apply masking based on time (0-1)
    function applyMasking(time) {
      const maskProb = 1 - cosineSchedule(time); // Probability of being masked

      tokens.forEach(token => {
        const rand = Math.random();
        if (rand < maskProb) {
          token.className = 'token-cell token-masked';
          token.textContent = '·';
        } else {
          token.className = 'token-cell token-normal';
          // Restore original character (simplified)
          const index = parseInt(token.dataset.index);
          token.textContent = String.fromCharCode(97 + index % 26);
        }
      });

      timeValue.textContent = `${Math.round(time * 100)}%`;
      timeSlider.value = Math.round(time * 100);
    }

    // Event listeners
    playBtn.addEventListener('click', () => {
      const icon = playBtn.querySelector('.btn-icon');
      const text = playBtn.querySelector('.btn-text');

      if (icon.textContent === '▶') {
        // Play animation with anime.js
        icon.textContent = '⏸';
        text.textContent = 'Pause';

        // Create anime.js timeline for masking animation
        const maskingTimeline = anime.timeline({
          duration: 5000,
          easing: 'linear',
          update: function(anim) {
            const progress = anim.progress / 100;
            applyMasking(progress);
          },
          complete: function() {
            icon.textContent = '▶';
            text.textContent = 'Play';
          }
        });

        // Add keyframes for the cosine schedule
        maskingTimeline
          .add({
            targets: '#masking-grid',
            // We'll update the masking through the update callback
          });
      } else {
        // Pause (simplified - just change button)
        icon.textContent = '▶';
        text.textContent = 'Play';
      }
    });

    resetBtn.addEventListener('click', () => {
      applyMasking(0);
      const icon = document.getElementById('masking-play').querySelector('.btn-icon');
      const text = document.getElementById('masking-play').querySelector('.btn-text');
      icon.textContent = '▶';
      text.textContent = 'Play';
    });

    timeSlider.addEventListener('input', () => {
      const time = parseInt(timeSlider.value) / 100;
      applyMasking(time);
    });

    // Initialize
    applyMasking(0);
  }

  // ─── Attention Mechanism Visualization ───
  function initAttentionVisualization() {
    const diffContainer = document.getElementById('diffusion-attention');
    const gptContainer = document.getElementById('gpt-attention');

    // Clear containers
    diffContainer.innerHTML = '';
    gptContainer.innerHTML = '';

    // Create token nodes (5 tokens for simplicity)
    const tokens = ['The', 'cat', 'sat', 'on', 'mat'];
    const centerX = 150;
    const centerY = 100;
    const radius = 80;
    const nodes = [];

    // Create nodes in a circle
    tokens.forEach((token, i) => {
      const angle = (i * 2 * Math.PI / tokens.length) - Math.PI / 2;
      const x = centerX + radius * Math.cos(angle);
      const y = centerY + radius * Math.sin(angle);

      const node = document.createElement('div');
      node.className = 'token-node';
      node.textContent = token[0]; // First letter
      node.style.left = `${x - 15}px`;
      node.style.top = `${y - 15}px`;
      node.dataset.index = i;

      diffContainer.appendChild(node);
      gptContainer.appendChild(node.cloneNode(true));

      nodes.push({ element: node, x, y });
    });

    // Create attention connections for diffusion (bidirectional)
    function createDiffusionConnections() {
      const connections = [];

      for (let i = 0; i < nodes.length; i++) {
        for (let j = 0; j < nodes.length; j++) {
          if (i !== j) {
            const conn = document.createElement('div');
            conn.className = 'attention-connection';

            const dx = nodes[j].x - nodes[i].x;
            const dy = nodes[j].y - nodes[i].y;
            const length = Math.sqrt(dx * dx + dy * dy);
            const angle = Math.atan2(dy, dx) * 180 / Math.PI;

            conn.style.width = `${length}px`;
            conn.style.left = `${nodes[i].x}px`;
            conn.style.top = `${nodes[i].y}px`;
            conn.style.transform = `rotate(${angle}deg)`;

            diffContainer.appendChild(conn);
            connections.push(conn);
          }
        }
      }

      return connections;
    }

    // Create attention connections for GPT (causal)
    function createGPTConnections() {
      const connections = [];

      for (let i = 0; i < nodes.length; i++) {
        for (let j = 0; j < i; j++) { // Only connect to previous tokens
          const conn = document.createElement('div');
          conn.className = 'attention-connection';

          const dx = nodes[j].x - nodes[i].x;
          const dy = nodes[j].y - nodes[i].y;
          const length = Math.sqrt(dx * dx + dy * dy);
          const angle = Math.atan2(dy, dx) * 180 / Math.PI;

          conn.style.width = `${length}px`;
          conn.style.left = `${nodes[i].x}px`;
          conn.style.top = `${nodes[i].y}px`;
          conn.style.transform = `rotate(${angle}deg)`;

          gptContainer.appendChild(conn);
          connections.push(conn);
        }
      }

      return connections;
    }

    const diffConnections = createDiffusionConnections();
    const gptConnections = createGPTConnections();

    // Animation controls
    const playBtn = document.getElementById('attention-play');
    const resetBtn = document.getElementById('attention-reset');

    playBtn.addEventListener('click', () => {
      const icon = playBtn.querySelector('.btn-icon');
      const text = playBtn.querySelector('.btn-text');

      if (icon.textContent === '▶') {
        icon.textContent = '⏹';
        text.textContent = 'Stop';

        // Animate connections with anime.js
        anime({
          targets: diffConnections,
          opacity: [0, 1],
          backgroundColor: ['#161b22', '#58a6ff'],
          delay: anime.stagger(100, {start: 100}),
          duration: 800,
          easing: 'easeInOutQuad'
        });

        setTimeout(() => {
          anime({
            targets: gptConnections,
            opacity: [0, 1],
            backgroundColor: ['#161b22', '#d2a8ff'],
            delay: anime.stagger(150, {start: 100}),
            duration: 800,
            easing: 'easeInOutQuad'
          });
        }, 1000);

      } else {
        icon.textContent = '▶';
        text.textContent = 'Animate';

        // Reset connections
        anime({
          targets: [...diffConnections, ...gptConnections],
          opacity: 1,
          backgroundColor: '#161b22',
          duration: 300,
          easing: 'easeOutQuad'
        });
      }
    });

    resetBtn.addEventListener('click', () => {
      diffConnections.forEach(conn => conn.style.opacity = '1');
      gptConnections.forEach(conn => conn.style.opacity = '1');

      const icon = document.getElementById('attention-play').querySelector('.btn-icon');
      const text = document.getElementById('attention-play').querySelector('.btn-text');
      icon.textContent = '▶';
      text.textContent = 'Animate';
    });
  }

  // ─── Training Objective Visualization ───
  function initTrainingVisualization() {
    const playBtn = document.getElementById('training-play');
    const resetBtn = document.getElementById('training-reset');
    const diffTargets = document.querySelectorAll('#diffusion-training .token.target');
    const gptPrediction = document.querySelector('#gpt-training .token.prediction');

    playBtn.addEventListener('click', () => {
      const icon = playBtn.querySelector('.btn-icon');
      const text = playBtn.querySelector('.btn-text');

      if (icon.textContent === '▶') {
        icon.textContent = '⏹';
        text.textContent = 'Stop';

        // Animate diffusion targets
        anime({
          targets: diffTargets,
          scale: [0, 1],
          opacity: [0, 1],
          delay: anime.stagger(200, {start: 300}),
          duration: 800,
          easing: 'spring(1, 80, 10, 0)'
        });

        // Animate GPT prediction
        setTimeout(() => {
          anime({
            targets: gptPrediction,
            scale: [0, 1],
            opacity: [0, 1],
            duration: 800,
            easing: 'spring(1, 80, 10, 0)'
          });
        }, 1000);

      } else {
        icon.textContent = '▶';
        text.textContent = 'Demonstrate';

        // Reset animations
        anime({
          targets: [...diffTargets, gptPrediction],
          scale: 1,
          opacity: 1,
          duration: 300
        });
      }
    });

    resetBtn.addEventListener('click', () => {
      const icon = document.getElementById('training-play').querySelector('.btn-icon');
      const text = document.getElementById('training-play').querySelector('.btn-text');
      icon.textContent = '▶';
      text.textContent = 'Demonstrate';

      // Reset to initial state
      anime({
        targets: [...diffTargets, gptPrediction],
        scale: 1,
        opacity: 1,
        duration: 300
      });
    });
  }

  // ─── Loss Scope Visualization ───
  function initLossVisualization() {
    const playBtn = document.getElementById('loss-play');
    const resetBtn = document.getElementById('loss-reset');
    const diffActiveTokens = document.querySelectorAll('#diffusion-loss .token.loss-active');
    const gptActiveTokens = document.querySelectorAll('#gpt-loss .token.loss-active');
    const diffInactiveTokens = document.querySelectorAll('#diffusion-loss .token.loss-inactive');

    playBtn.addEventListener('click', () => {
      const icon = playBtn.querySelector('.btn-icon');
      const text = playBtn.querySelector('.btn-text');

      if (icon.textContent === '▶') {
        icon.textContent = '⏹';
        text.textContent = 'Stop';

        // Highlight diffusion active tokens
        anime({
          targets: diffActiveTokens,
          scale: [1, 1.1, 1],
          boxShadow: [
            '0 0 0 0px var(--green)',
            '0 0 0 3px var(--green), 0 0 12px var(--green-bright)',
            '0 0 0 0px var(--green)'
          ],
          duration: 1000,
          delay: anime.stagger(200),
          easing: 'easeInOutQuad'
        });

        // Dim inactive tokens
        anime({
          targets: diffInactiveTokens,
          opacity: [1, 0.3],
          duration: 800,
          easing: 'easeInOutQuad'
        });

        // Highlight GPT active tokens
        setTimeout(() => {
          anime({
            targets: gptActiveTokens,
            scale: [1, 1.1, 1],
            boxShadow: [
              '0 0 0 0px var(--green)',
              '0 0 0 3px var(--green), 0 0 12px var(--green-bright)',
              '0 0 0 0px var(--green)'
            ],
            duration: 1000,
            delay: anime.stagger(100),
            easing: 'easeInOutQuad'
          });
        }, 800);

      } else {
        icon.textContent = '▶';
        text.textContent = 'Animate';

        // Reset all tokens
        anime({
          targets: [...diffActiveTokens, ...diffInactiveTokens, ...gptActiveTokens],
          scale: 1,
          opacity: 1,
          boxShadow: '0 0 0 0px var(--green)',
          duration: 300
        });
      }
    });

    resetBtn.addEventListener('click', () => {
      const icon = document.getElementById('loss-play').querySelector('.btn-icon');
      const text = document.getElementById('loss-play').querySelector('.btn-text');
      icon.textContent = '▶';
      text.textContent = 'Animate';

      // Reset all tokens
      anime({
        targets: [...diffActiveTokens, ...diffInactiveTokens, ...gptActiveTokens],
        scale: 1,
        opacity: 1,
        boxShadow: '0 0 0 0px var(--green)',
        duration: 300
      });
    });
  }

  // ─── Generation Process Visualization ───
  function initGenerationVisualization() {
    const playBtn = document.getElementById('generation-play');
    const resetBtn = document.getElementById('generation-reset');
    const speedSlider = document.getElementById('generation-speed');
    const speedValue = document.getElementById('generation-speed-value');
    const diffTrack = document.getElementById('diffusion-track');
    const gptTrack = document.getElementById('gpt-track');
    const diffSteps = document.getElementById('diffusion-steps');
    const diffTokens = document.getElementById('diffusion-tokens');
    const gptTokens = document.getElementById('gpt-tokens');
    const diffFill = document.querySelector('.diffusion-fill');
    const gptFill = document.querySelector('.gpt-fill');

    let diffAnimation = null;
    let gptAnimation = null;

    // Speed control
    speedSlider.addEventListener('input', () => {
      const speed = parseInt(speedSlider.value);
      speedValue.textContent = `${speed}x`;
    });

    playBtn.addEventListener('click', () => {
      const icon = playBtn.querySelector('.btn-icon');
      const text = playBtn.querySelector('.btn-text');

      if (icon.textContent === '▶') {
        icon.textContent = '⏹';
        text.textContent = 'Stop';

        // Reset counters
        diffSteps.textContent = '0';
        diffTokens.textContent = '0';
        gptTokens.textContent = '0';
        diffFill.style.width = '0%';
        gptFill.style.width = '0%';

        // Get masked tokens for diffusion
        const diffMaskedTokens = diffTrack.querySelectorAll('.token.masked');
        const gptEmptyTokens = gptTrack.querySelectorAll('.token.empty');

        // Diffusion animation - parallel unmasking
        let diffStep = 0;
        const diffTotalSteps = 5;
        const diffTokensPerStep = Math.ceil(diffMaskedTokens.length / diffTotalSteps);

        function animateDiffusionStep() {
          if (diffStep >= diffTotalSteps) {
            return;
          }

          // Unmask a batch of tokens
          const startIndex = diffStep * diffTokensPerStep;
          const endIndex = Math.min(startIndex + diffTokensPerStep, diffMaskedTokens.length);

          for (let i = startIndex; i < endIndex; i++) {
            if (diffMaskedTokens[i]) {
              diffMaskedTokens[i].className = 'token normal';
              diffMaskedTokens[i].textContent = String.fromCharCode(97 + i);
            }
          }

          diffStep++;
          diffSteps.textContent = diffStep;
          diffTokens.textContent = Math.min(diffStep * diffTokensPerStep, diffMaskedTokens.length);
          diffFill.style.width = `${(diffStep / diffTotalSteps) * 100}%`;

          if (diffStep < diffTotalSteps) {
            const speed = parseInt(speedSlider.value);
            setTimeout(animateDiffusionStep, 1000 / speed);
          }
        }

        // GPT animation - sequential generation
        let gptIndex = 0;

        function animateGPTStep() {
          if (gptIndex >= gptEmptyTokens.length) {
            return;
          }

          // Generate next token
          gptEmptyTokens[gptIndex].className = 'token normal';
          gptEmptyTokens[gptIndex].textContent = String.fromCharCode(97 + gptIndex + 5);

          gptIndex++;
          gptTokens.textContent = gptIndex;
          gptFill.style.width = `${(gptIndex / gptEmptyTokens.length) * 100}%`;

          if (gptIndex < gptEmptyTokens.length) {
            const speed = parseInt(speedSlider.value);
            setTimeout(animateGPTStep, 800 / speed);
          }
        }

        // Start both animations
        animateDiffusionStep();
        setTimeout(animateGPTStep, 500);

      } else {
        icon.textContent = '▶';
        text.textContent = 'Start Race';
      }
    });

    resetBtn.addEventListener('click', () => {
      const icon = document.getElementById('generation-play').querySelector('.btn-icon');
      const text = document.getElementById('generation-play').querySelector('.btn-text');
      icon.textContent = '▶';
      text.textContent = 'Start Race';

      // Reset tracks
      const diffMaskedTokens = diffTrack.querySelectorAll('.token.masked, .token.normal');
      const gptEmptyTokens = gptTrack.querySelectorAll('.token.empty, .token.normal');

      diffMaskedTokens.forEach((token, i) => {
        if (i >= 5) { // First 5 are prompt
          token.className = 'token masked';
          token.textContent = '·';
        }
      });

      gptEmptyTokens.forEach((token, i) => {
        if (i >= 5) { // First 5 are prompt
          token.className = 'token empty';
          token.textContent = '';
        }
      });

      // Reset counters
      diffSteps.textContent = '0';
      diffTokens.textContent = '0';
      gptTokens.textContent = '0';
      diffFill.style.width = '0%';
      gptFill.style.width = '0%';
    });
  }

  // ─── Cosine Schedule Visualization ───
  function initCosineVisualization() {
    const canvas = document.getElementById('cosine-graph');
    const ctx = canvas.getContext('2d');
    const timeSlider = document.getElementById('cosine-time');
    const timeValue = document.getElementById('cosine-time-value');
    const alphaValue = document.getElementById('alpha-value');
    const maskProbValue = document.getElementById('mask-prob-value');
    const unmaskRateValue = document.getElementById('unmask-rate-value');
    const playBtn = document.getElementById('cosine-play');
    const resetBtn = document.getElementById('cosine-reset');

    // Cosine schedule function: α(t) = cos(πt/2)²
    function cosineSchedule(t) {
      return Math.pow(Math.cos(Math.PI * t / 2), 2);
    }

    // Derivative for unmasking rate
    function cosineDerivative(t) {
      if (t <= 0 || t >= 1) return 0;
      return -Math.PI * Math.cos(Math.PI * t / 2) * Math.sin(Math.PI * t / 2);
    }

    // Draw the cosine graph
    function drawGraph(highlightTime = 0) {
      const width = canvas.width;
      const height = canvas.height;
      const padding = 40;

      // Clear canvas
      ctx.clearRect(0, 0, width, height);

      // Draw grid
      ctx.strokeStyle = '#30363d';
      ctx.lineWidth = 1;

      // Vertical grid lines
      for (let i = 0; i <= 10; i++) {
        const x = padding + (i / 10) * (width - 2 * padding);
        ctx.beginPath();
        ctx.moveTo(x, padding);
        ctx.lineTo(x, height - padding);
        ctx.stroke();
      }

      // Horizontal grid lines
      for (let i = 0; i <= 10; i++) {
        const y = padding + (i / 10) * (height - 2 * padding);
        ctx.beginPath();
        ctx.moveTo(padding, y);
        ctx.lineTo(width - padding, y);
        ctx.stroke();
      }

      // Draw axes
      ctx.strokeStyle = '#8b949e';
      ctx.lineWidth = 2;

      // X-axis
      ctx.beginPath();
      ctx.moveTo(padding, height - padding);
      ctx.lineTo(width - padding, height - padding);
      ctx.stroke();

      // Y-axis
      ctx.beginPath();
      ctx.moveTo(padding, padding);
      ctx.lineTo(padding, height - padding);
      ctx.stroke();

      // Draw axis labels
      ctx.fillStyle = '#8b949e';
      ctx.font = '12px monospace';
      ctx.textAlign = 'center';

      // X-axis labels
      for (let i = 0; i <= 10; i++) {
        const x = padding + (i / 10) * (width - 2 * padding);
        ctx.fillText((i / 10).toFixed(1), x, height - padding + 20);
      }

      // Y-axis labels
      ctx.textAlign = 'right';
      for (let i = 0; i <= 10; i++) {
        const y = height - padding - (i / 10) * (height - 2 * padding);
        ctx.fillText((i / 10).toFixed(1), padding - 10, y + 4);
      }

      // Axis titles
      ctx.textAlign = 'center';
      ctx.fillText('Time (t)', width / 2, height - 10);
      ctx.save();
      ctx.translate(15, height / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText('α(t)', 0, 0);
      ctx.restore();

      // Draw cosine curve
      ctx.beginPath();
      ctx.strokeStyle = '#58a6ff';
      ctx.lineWidth = 3;

      const steps = 200;
      for (let i = 0; i <= steps; i++) {
        const t = i / steps;
        const x = padding + t * (width - 2 * padding);
        const alpha = cosineSchedule(t);
        const y = height - padding - alpha * (height - 2 * padding);

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();

      // Highlight current point
      if (highlightTime >= 0 && highlightTime <= 1) {
        const x = padding + highlightTime * (width - 2 * padding);
        const alpha = cosineSchedule(highlightTime);
        const y = height - padding - alpha * (height - 2 * padding);

        // Draw point
        ctx.beginPath();
        ctx.fillStyle = '#3fb950';
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fill();

        // Draw vertical line to x-axis
        ctx.beginPath();
        ctx.strokeStyle = '#3fb950';
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        ctx.moveTo(x, y);
        ctx.lineTo(x, height - padding);
        ctx.stroke();
        ctx.setLineDash([]);

        // Draw horizontal line to y-axis
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(padding, y);
        ctx.stroke();
      }
    }

    // Update values based on time
    function updateValues(time) {
      const alpha = cosineSchedule(time);
      const maskProb = 1 - alpha;
      const unmaskRate = Math.abs(cosineDerivative(time));

      timeValue.textContent = `${Math.round(time * 100)}%`;
      alphaValue.textContent = alpha.toFixed(2);
      maskProbValue.textContent = maskProb.toFixed(2);
      unmaskRateValue.textContent = unmaskRate.toFixed(2);

      drawGraph(time);
    }

    // Initialize
    drawGraph();
    updateValues(0);

    // Event listeners
    timeSlider.addEventListener('input', () => {
      const time = parseInt(timeSlider.value) / 100;
      updateValues(time);
    });

    playBtn.addEventListener('click', () => {
      const icon = playBtn.querySelector('.btn-icon');
      const text = playBtn.querySelector('.btn-text');

      if (icon.textContent === '▶') {
        icon.textContent = '⏹';
        text.textContent = 'Stop';

        let startTime = null;
        const duration = 5000; // 5 seconds

        function animate(timestamp) {
          if (!startTime) startTime = timestamp;
          const elapsed = timestamp - startTime;
          const progress = Math.min(elapsed / duration, 1);

          timeSlider.value = Math.round(progress * 100);
          updateValues(progress);

          if (progress < 1) {
            requestAnimationFrame(animate);
          } else {
            icon.textContent = '▶';
            text.textContent = 'Animate';
          }
        }

        requestAnimationFrame(animate);
      } else {
        icon.textContent = '▶';
        text.textContent = 'Animate';
      }
    });

    resetBtn.addEventListener('click', () => {
      timeSlider.value = 0;
      updateValues(0);

      const icon = document.getElementById('cosine-play').querySelector('.btn-icon');
      const text = document.getElementById('cosine-play').querySelector('.btn-text');
      icon.textContent = '▶';
      text.textContent = 'Animate';
    });
  }

  // ─── Data Flow Comparison Visualization ───
  function initDataFlowVisualization() {
    const playBtn = document.getElementById('dataflow-play');
    const resetBtn = document.getElementById('dataflow-reset');

    playBtn.addEventListener('click', () => {
      const icon = playBtn.querySelector('.btn-icon');
      const text = playBtn.querySelector('.btn-text');

      if (icon.textContent === '▶') {
        icon.textContent = '⏹';
        text.textContent = 'Stop';

        // Animate diffusion data packets
        const diffPackets = document.querySelectorAll('#diffusion-dataflow .data-packet');
        anime({
          targets: diffPackets,
          scale: [0, 1.2, 1],
          opacity: [0, 1],
          delay: anime.stagger(100, {start: 300}),
          duration: 800,
          easing: 'easeInOutQuad',
          complete: function() {
            // Animate GPT data packets after diffusion
            setTimeout(() => {
              const gptPackets = document.querySelectorAll('#gpt-dataflow .data-packet');
              anime({
                targets: gptPackets,
                translateX: [0, 20, 0],
                opacity: [0, 1],
                delay: anime.stagger(200, {start: 100}),
                duration: 600,
                easing: 'easeInOutQuad'
              });

              // Animate arrows
              const gptArrows = document.querySelectorAll('#gpt-dataflow .data-packet-arrow');
              anime({
                targets: gptArrows,
                opacity: [0, 1],
                delay: anime.stagger(200, {start: 200}),
                duration: 400,
                easing: 'easeInOutQuad'
              });
            }, 500);
          }
        });

      } else {
        icon.textContent = '▶';
        text.textContent = 'Animate';

        // Reset animations
        anime({
          targets: [...document.querySelectorAll('#diffusion-dataflow .data-packet'),
                   ...document.querySelectorAll('#gpt-dataflow .data-packet'),
                   ...document.querySelectorAll('#gpt-dataflow .data-packet-arrow')],
          scale: 1,
          translateX: 0,
          opacity: 1,
          duration: 300
        });
      }
    });

    resetBtn.addEventListener('click', () => {
      const icon = document.getElementById('dataflow-play').querySelector('.btn-icon');
      const text = document.getElementById('dataflow-play').querySelector('.btn-text');
      icon.textContent = '▶';
      text.textContent = 'Animate';

      // Reset all elements to initial state
      anime({
        targets: [...document.querySelectorAll('#diffusion-dataflow .data-packet'),
                 ...document.querySelectorAll('#gpt-dataflow .data-packet'),
                 ...document.querySelectorAll('#gpt-dataflow .data-packet-arrow')],
        scale: 1,
        translateX: 0,
        opacity: 1,
        duration: 300
      });
    });
  }

  // ─── Transformer Block Visualization ───
  function initTransformerVisualization() {
    const playBtn = document.getElementById('transformer-play');
    const resetBtn = document.getElementById('transformer-reset');
    const diffDiagram = document.getElementById('diffusion-transformer');
    const gptDiagram = document.getElementById('gpt-transformer');

    // Create attention connections for diffusion (bidirectional)
    function createDiffusionConnections() {
      const container = diffDiagram.querySelector('.attention-connections');
      container.innerHTML = '';

      // Get token nodes
      const tokenNodes = diffDiagram.querySelectorAll('.token-node');
      const positions = [
        { x: 30, y: 30 },
        { x: 80, y: 30 },
        { x: 130, y: 30 },
        { x: 180, y: 30 },
        { x: 230, y: 30 }
      ];

      // Create bidirectional connections
      for (let i = 0; i < tokenNodes.length; i++) {
        for (let j = 0; j < tokenNodes.length; j++) {
          if (i !== j) {
            const conn = document.createElement('div');
            conn.className = 'attention-connection';

            const dx = positions[j].x - positions[i].x;
            const dy = positions[j].y - positions[i].y;
            const length = Math.sqrt(dx * dx + dy * dy);
            const angle = Math.atan2(dy, dx) * 180 / Math.PI;

            conn.style.width = `${length}px`;
            conn.style.left = `${positions[i].x + 15}px`; // Center of node
            conn.style.top = `${positions[i].y + 15}px`;
            conn.style.transform = `rotate(${angle}deg)`;

            container.appendChild(conn);
          }
        }
      }

      return container.querySelectorAll('.attention-connection');
    }

    // Create attention connections for GPT (causal)
    function createGPTConnections() {
      const container = gptDiagram.querySelector('.attention-connections');
      container.innerHTML = '';

      // Get token nodes
      const tokenNodes = gptDiagram.querySelectorAll('.token-node');
      const positions = [
        { x: 30, y: 30 },
        { x: 80, y: 30 },
        { x: 130, y: 30 },
        { x: 180, y: 30 },
        { x: 230, y: 30 }
      ];

      // Create causal connections (only to previous tokens)
      for (let i = 0; i < tokenNodes.length; i++) {
        for (let j = 0; j < i; j++) { // Only connect to previous tokens
          const conn = document.createElement('div');
          conn.className = 'attention-connection causal';

          const dx = positions[j].x - positions[i].x;
          const dy = positions[j].y - positions[i].y;
          const length = Math.sqrt(dx * dx + dy * dy);
          const angle = Math.atan2(dy, dx) * 180 / Math.PI;

          conn.style.width = `${length}px`;
          conn.style.left = `${positions[i].x + 15}px`; // Center of node
          conn.style.top = `${positions[i].y + 15}px`;
          conn.style.transform = `rotate(${angle}deg)`;

          container.appendChild(conn);
        }
      }

      return container.querySelectorAll('.attention-connection');
    }

    // Initialize connections
    let diffConnections = createDiffusionConnections();
    let gptConnections = createGPTConnections();

    // Animation controls
    playBtn.addEventListener('click', () => {
      const icon = playBtn.querySelector('.btn-icon');
      const text = playBtn.querySelector('.btn-text');

      if (icon.textContent === '▶') {
        icon.textContent = '⏹';
        text.textContent = 'Stop';

        // Animate diffusion connections
        anime({
          targets: diffConnections,
          opacity: [0, 1],
          backgroundColor: ['#161b22', '#58a6ff'],
          delay: anime.stagger(50, {start: 100}),
          duration: 800,
          easing: 'easeInOutQuad',
          complete: function() {
            // Animate GPT connections after diffusion
            setTimeout(() => {
              anime({
                targets: gptConnections,
                opacity: [0, 1],
                backgroundColor: ['#161b22', '#d2a8ff'],
                delay: anime.stagger(100, {start: 100}),
                duration: 800,
                easing: 'easeInOutQuad'
              });
            }, 500);
          }
        });

      } else {
        icon.textContent = '▶';
        text.textContent = 'Animate';

        // Reset connections
        anime({
          targets: [...diffConnections, ...gptConnections],
          opacity: 1,
          backgroundColor: '#161b22',
          duration: 300,
          easing: 'easeOutQuad'
        });
      }
    });

    resetBtn.addEventListener('click', () => {
      // Recreate connections
      diffConnections = createDiffusionConnections();
      gptConnections = createGPTConnections();

      const icon = document.getElementById('transformer-play').querySelector('.btn-icon');
      const text = document.getElementById('transformer-play').querySelector('.btn-text');
      icon.textContent = '▶';
      text.textContent = 'Animate';
    });
  }

  // ─── Confidence-Based Unmasking Visualization ───
  function initConfidenceVisualization() {
    const grid = document.getElementById('confidence-grid');
    const playBtn = document.getElementById('confidence-play');
    const resetBtn = document.getElementById('confidence-reset');
    const thresholdSlider = document.getElementById('confidence-threshold');
    const thresholdValue = document.getElementById('confidence-threshold-value');

    // Create token grid (5x10 = 50 tokens)
    const rows = 5;
    const cols = 10;
    const tokens = [];

    // Clear grid
    grid.innerHTML = '';

    // Create token cells with random confidence values
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const cell = document.createElement('div');
        cell.className = 'token-cell token-masked';
        cell.textContent = '·';
        cell.dataset.index = i * cols + j;

        // Assign random confidence level (0-100)
        const confidence = Math.floor(Math.random() * 101);
        cell.dataset.confidence = confidence;

        grid.appendChild(cell);
        tokens.push(cell);
      }
    }

    // Apply confidence-based styling
    function applyConfidenceStyling(threshold = 50) {
      tokens.forEach(token => {
        const confidence = parseInt(token.dataset.confidence);

        // Remove previous confidence classes
        token.classList.remove('high-confidence', 'medium-confidence', 'low-confidence');

        if (confidence >= threshold) {
          // Determine confidence level
          if (confidence >= 80) {
            token.classList.add('high-confidence');
            token.textContent = String.fromCharCode(65 + (parseInt(token.dataset.index) % 26)); // A-Z
          } else if (confidence >= 60) {
            token.classList.add('medium-confidence');
            token.textContent = String.fromCharCode(97 + (parseInt(token.dataset.index) % 26)); // a-z
          } else {
            token.classList.add('low-confidence');
            token.textContent = String.fromCharCode(48 + (parseInt(token.dataset.index) % 10)); // 0-9
          }
        } else {
          // Keep masked
          token.className = 'token-cell token-masked';
          token.textContent = '·';
        }
      });

      thresholdValue.textContent = `${threshold}%`;
    }

    // Event listeners
    playBtn.addEventListener('click', () => {
      const icon = playBtn.querySelector('.btn-icon');
      const text = playBtn.querySelector('.btn-text');

      if (icon.textContent === '▶') {
        icon.textContent = '⏹';
        text.textContent = 'Stop';

        // Animate confidence-based unmasking with anime.js
        // First reveal high confidence tokens
        const highConfTokens = tokens.filter(token => parseInt(token.dataset.confidence) >= 80);
        anime({
          targets: highConfTokens,
          scale: [0, 1.1, 1],
          opacity: [0, 1],
          delay: anime.stagger(50),
          duration: 800,
          easing: 'spring(1, 80, 10, 0)',
          begin: function() {
            highConfTokens.forEach(token => {
              token.textContent = String.fromCharCode(65 + (parseInt(token.dataset.index) % 26));
              token.classList.add('high-confidence');
            });
          }
        });

        // Then reveal medium confidence tokens
        setTimeout(() => {
          const medConfTokens = tokens.filter(token =>
            parseInt(token.dataset.confidence) >= 60 && parseInt(token.dataset.confidence) < 80);
          anime({
            targets: medConfTokens,
            scale: [0, 1.05, 1],
            opacity: [0, 1],
            delay: anime.stagger(100),
            duration: 800,
            easing: 'spring(1, 80, 10, 0)',
            begin: function() {
              medConfTokens.forEach(token => {
                token.textContent = String.fromCharCode(97 + (parseInt(token.dataset.index) % 26));
                token.classList.add('medium-confidence');
              });
            }
          });
        }, 1000);

        // Finally reveal low confidence tokens
        setTimeout(() => {
          const lowConfTokens = tokens.filter(token =>
            parseInt(token.dataset.confidence) >= 50 && parseInt(token.dataset.confidence) < 60);
          anime({
            targets: lowConfTokens,
            scale: [0, 1, 1],
            opacity: [0, 1],
            delay: anime.stagger(150),
            duration: 800,
            easing: 'spring(1, 80, 10, 0)',
            begin: function() {
              lowConfTokens.forEach(token => {
                token.textContent = String.fromCharCode(48 + (parseInt(token.dataset.index) % 10));
                token.classList.add('low-confidence');
              });
            },
            complete: function() {
              icon.textContent = '▶';
              text.textContent = 'Animate';
            }
          });
        }, 2000);

      } else {
        icon.textContent = '▶';
        text.textContent = 'Animate';
      }
    });

    resetBtn.addEventListener('click', () => {
      // Reset all tokens to masked state
      tokens.forEach(token => {
        token.className = 'token-cell token-masked';
        token.textContent = '·';
      });

      const icon = document.getElementById('confidence-play').querySelector('.btn-icon');
      const text = document.getElementById('confidence-play').querySelector('.btn-text');
      icon.textContent = '▶';
      text.textContent = 'Animate';
    });

    thresholdSlider.addEventListener('input', () => {
      const threshold = parseInt(thresholdSlider.value);
      applyConfidenceStyling(threshold);
    });

    // Initialize
    applyConfidenceStyling(50);
  }

  // ─── Initialize first visualization ───
  document.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('masking-vis')) {
      initMaskingVisualization();
      window.maskingInitialized = true;
    }

    // Add event listeners for other visualizations
    const trainingBtn = document.querySelector('[data-vis="training"]');
    if (trainingBtn) {
      trainingBtn.addEventListener('click', () => {
        if (!window.trainingInitialized) {
          initTrainingVisualization();
          window.trainingInitialized = true;
        }
      });
    }

    const lossBtn = document.querySelector('[data-vis="loss"]');
    if (lossBtn) {
      lossBtn.addEventListener('click', () => {
        if (!window.lossInitialized) {
          initLossVisualization();
          window.lossInitialized = true;
        }
      });
    }

    const generationBtn = document.querySelector('[data-vis="generation"]');
    if (generationBtn) {
      generationBtn.addEventListener('click', () => {
        if (!window.generationInitialized) {
          initGenerationVisualization();
          window.generationInitialized = true;
        }
      });
    }

    const cosineBtn = document.querySelector('[data-vis="cosine"]');
    if (cosineBtn) {
      cosineBtn.addEventListener('click', () => {
        if (!window.cosineInitialized) {
          initCosineVisualization();
          window.cosineInitialized = true;
        }
      });
    }

    const confidenceBtn = document.querySelector('[data-vis="confidence"]');
    if (confidenceBtn) {
      confidenceBtn.addEventListener('click', () => {
        if (!window.confidenceInitialized) {
          initConfidenceVisualization();
          window.confidenceInitialized = true;
        }
      });
    }

    const transformerBtn = document.querySelector('[data-vis="transformer"]');
    if (transformerBtn) {
      transformerBtn.addEventListener('click', () => {
        if (!window.transformerInitialized) {
          initTransformerVisualization();
          window.transformerInitialized = true;
        }
      });
    }

    const dataflowBtn = document.querySelector('[data-vis="dataflow"]');
    if (dataflowBtn) {
      dataflowBtn.addEventListener('click', () => {
        if (!window.dataflowInitialized) {
          initDataFlowVisualization();
          window.dataflowInitialized = true;
        }
      });
    }

    const overfittingBtn = document.querySelector('[data-vis="overfitting"]');
    if (overfittingBtn) {
      overfittingBtn.addEventListener('click', () => {
        if (!window.overfittingInitialized) {
          initOverfittingVisualization();
          window.overfittingInitialized = true;
        }
      });
    }
  });

})();