(() => {
  "use strict";

  const state = {
    currentSplit: null,
    thresholdMin: 0.0,
    thresholdMax: 1.0,
    thresholdNorm: 0.5,
    auroc: 0.0,
    auprc: 0.0,
    prevalence: 0.0,
    grid: null,
    rocCurve: null,
    prCurve: null,
    samplesHistory: [], // Array of sample objects
    activeSample: null,
    labelRevealed: false,
    shapCache: {}, // key: `${split}:${row_index}` -> response
  };

  // DOM Elements
  const el = {
    splitSelect: document.getElementById("split-select"),
    metaRows: document.getElementById("meta-rows"),
    metaPrevalence: document.getElementById("meta-prevalence"),
    metaFeatures: document.getElementById("meta-features"),
    errorBanner: document.getElementById("error-banner"),
    errorMessage: document.getElementById("error-message"),

    rawThresholdPill: document.getElementById("raw-threshold-pill"),
    ppvBigText: document.getElementById("ppv-big-text"),
    ppvSmallText: document.getElementById("ppv-small-text"),
    ppvCountsText: document.getElementById("ppv-counts-text"),
    sensBigText: document.getElementById("sens-big-text"),
    sensSmallText: document.getElementById("sens-small-text"),
    sensCountsText: document.getElementById("sens-counts-text"),

    thresholdSlider: document.getElementById("threshold-slider"),
    thresholdNumber: document.getElementById("threshold-number"),
    stepUpBtn: document.getElementById("step-up-btn"),
    stepDownBtn: document.getElementById("step-down-btn"),
    scaleMinLabel: document.getElementById("scale-min-label"),
    scaleMaxLabel: document.getElementById("scale-max-label"),
    sampleScoreMarker: document.getElementById("sample-score-marker"),
    sampleMarkerLabel: document.getElementById("sample-marker-label"),

    freshSampleBtn: document.getElementById("fresh-sample-btn"),
    freshPositiveBtn: document.getElementById("fresh-positive-btn"),
    freshNegativeBtn: document.getElementById("fresh-negative-btn"),
    sampleHistorySelect: document.getElementById("sample-history-select"),
    rowIndexInput: document.getElementById("row-index-input"),
    loadRowBtn: document.getElementById("load-row-btn"),

    predictionBox: document.getElementById("prediction-box"),
    predictionStatusText: document.getElementById("prediction-status-text"),
    predictionScoresText: document.getElementById("prediction-scores-text"),

    revealLabelBox: document.getElementById("reveal-label-box"),
    revealLabelText: document.getElementById("reveal-label-text"),
    revealLabelHint: document.getElementById("reveal-label-hint"),

    aurocBadge: document.getElementById("auroc-badge"),
    rocCoordBadge: document.getElementById("roc-coord-badge"),
    rocSvg: document.getElementById("roc-svg"),

    auprcBadge: document.getElementById("auprc-badge"),
    prCoordBadge: document.getElementById("pr-coord-badge"),
    prSvg: document.getElementById("pr-svg"),

    tableSampleSubtitle: document.getElementById("table-sample-subtitle"),
    featureScopeSelect: document.getElementById("feature-scope-select"),
    featureSearchInput: document.getElementById("feature-search-input"),
    pivotedTableBody: document.getElementById("pivoted-table-body"),

    generateShapBtn: document.getElementById("generate-shap-btn"),
    shapLoadingContainer: document.getElementById("shap-loading-container"),
    shapLoadingText: document.getElementById("shap-loading-text"),
    shapLoadingPercent: document.getElementById("shap-loading-percent"),
    shapProgressBar: document.getElementById("shap-progress-bar"),
    shapPlaceholder: document.getElementById("shap-placeholder"),
    shapResult: document.getElementById("shap-result"),
    shapWaterfallImg: document.getElementById("shap-waterfall-img"),
  };

  // SVG Chart Geometry Constants
  const CHART = {
    width: 460,
    height: 340,
    padLeft: 56,
    padRight: 24,
    padTop: 22,
    padBottom: 46,
  };
  CHART.plotW = CHART.width - CHART.padLeft - CHART.padRight;
  CHART.plotH = CHART.height - CHART.padTop - CHART.padBottom;

  function toSvgX(val) {
    const clamped = Math.max(0, Math.min(1, val));
    return CHART.padLeft + clamped * CHART.plotW;
  }

  function toSvgY(val) {
    const clamped = Math.max(0, Math.min(1, val));
    return CHART.padTop + (1 - clamped) * CHART.plotH;
  }

  function syncSliderCssVariables() {
    el.thresholdSlider.style.setProperty("--attr-min", el.thresholdSlider.min || "0");
    el.thresholdSlider.style.setProperty("--attr-max", el.thresholdSlider.max || "1");
    el.thresholdSlider.style.setProperty("--control-value", el.thresholdSlider.value);
  }

  function buildCurveSvg(svgEl, pointsX, pointsY, xLabel, yLabel, baselineType, baselineY, colorHex, dotPrefix) {
    const ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0];
    let gridHtml = "";

    for (const t of ticks) {
      const x = toSvgX(t);
      const y = toSvgY(t);
      gridHtml += `
        <line x1="${x}" y1="${CHART.padTop}" x2="${x}" y2="${CHART.padTop + CHART.plotH}" stroke="#e2e8f0" stroke-width="1" />
        <line x1="${CHART.padLeft}" y1="${y}" x2="${CHART.padLeft + CHART.plotW}" y2="${y}" stroke="#e2e8f0" stroke-width="1" />
        <text x="${x}" y="${CHART.padTop + CHART.plotH + 18}" text-anchor="middle" font-size="11" fill="#64748b" font-family="monospace">${t.toFixed(1)}</text>
        <text x="${CHART.padLeft - 8}" y="${y + 4}" text-anchor="end" font-size="11" fill="#64748b" font-family="monospace">${t.toFixed(1)}</text>
      `;
    }

    // Build path string
    const pathCoords = [];
    for (let i = 0; i < pointsX.length; i++) {
      const px = toSvgX(pointsX[i]).toFixed(2);
      const py = toSvgY(pointsY[i]).toFixed(2);
      pathCoords.push(`${i === 0 ? "M" : "L"} ${px} ${py}`);
    }
    const linePath = pathCoords.join(" ");

    // Area under curve path
    let areaPath = "";
    if (pointsX.length > 1) {
      const firstX = toSvgX(pointsX[0]).toFixed(2);
      const lastX = toSvgX(pointsX[pointsX.length - 1]).toFixed(2);
      const bottomY = toSvgY(0).toFixed(2);
      areaPath = `${linePath} L ${lastX} ${bottomY} L ${firstX} ${bottomY} Z`;
    }

    // Baseline (diagonal chance line for ROC, horizontal prevalence line for PR)
    let baselineHtml = "";
    if (baselineType === "diagonal") {
      baselineHtml = `<line x1="${toSvgX(0)}" y1="${toSvgY(0)}" x2="${toSvgX(1)}" y2="${toSvgY(1)}" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="5,5" />`;
    } else if (baselineType === "horizontal") {
      const by = toSvgY(baselineY);
      baselineHtml = `<line x1="${toSvgX(0)}" y1="${by}" x2="${toSvgX(1)}" y2="${by}" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="5,5" />`;
    }

    svgEl.innerHTML = `
      ${gridHtml}
      <rect x="${CHART.padLeft}" y="${CHART.padTop}" width="${CHART.plotW}" height="${CHART.plotH}" fill="none" stroke="#cbd5e1" stroke-width="1.2" />
      ${ areaPath ? `<path d="${areaPath}" fill="${colorHex}" fill-opacity="0.10" />` : "" }
      ${baselineHtml}
      ${ linePath ? `<path d="${linePath}" fill="none" stroke="${colorHex}" stroke-width="2.6" stroke-linejoin="round" stroke-linecap="round" />` : "" }

      <!-- Live Threshold Crosshairs -->
      <line id="${dotPrefix}-vline" x1="${toSvgX(0)}" y1="${CHART.padTop}" x2="${toSvgX(0)}" y2="${CHART.padTop + CHART.plotH}" stroke="#dc2626" stroke-width="1.2" stroke-dasharray="3,3" />
      <line id="${dotPrefix}-hline" x1="${CHART.padLeft}" y1="${toSvgY(0)}" x2="${CHART.padLeft + CHART.plotW}" y2="${toSvgY(0)}" stroke="#dc2626" stroke-width="1.2" stroke-dasharray="3,3" />

      <!-- Live Threshold Circle / Dot -->
      <circle id="${dotPrefix}-halo" cx="${toSvgX(0)}" cy="${toSvgY(0)}" r="10" fill="#dc2626" fill-opacity="0.22" />
      <circle id="${dotPrefix}-dot" cx="${toSvgX(0)}" cy="${toSvgY(0)}" r="5.5" fill="#dc2626" stroke="#ffffff" stroke-width="2" />

      <!-- Axis Titles -->
      <text x="${CHART.padLeft + CHART.plotW / 2}" y="${CHART.height - 8}" text-anchor="middle" font-size="12" font-weight="600" fill="#334155">${xLabel}</text>
      <text x="16" y="${CHART.padTop + CHART.plotH / 2}" text-anchor="middle" font-size="12" font-weight="600" fill="#334155" transform="rotate(-90 16 ${CHART.padTop + CHART.plotH / 2})">${yLabel}</text>
    `;
  }

  function updateDotOnSvg(dotPrefix, xVal, yVal) {
    const cx = toSvgX(xVal);
    const cy = toSvgY(yVal);
    const dot = document.getElementById(`${dotPrefix}-dot`);
    const halo = document.getElementById(`${dotPrefix}-halo`);
    const vline = document.getElementById(`${dotPrefix}-vline`);
    const hline = document.getElementById(`${dotPrefix}-hline`);

    if (dot && halo && vline && hline) {
      dot.setAttribute("cx", cx);
      dot.setAttribute("cy", cy);
      halo.setAttribute("cx", cx);
      halo.setAttribute("cy", cy);
      vline.setAttribute("x1", cx);
      vline.setAttribute("x2", cx);
      hline.setAttribute("y1", cy);
      hline.setAttribute("y2", cy);
    }
  }

  function updateThresholdUI(normValue, source) {
    let clamped = parseFloat(normValue);
    if (Number.isNaN(clamped)) {
      clamped = state.thresholdNorm;
    }
    clamped = Math.max(0.0, Math.min(1.0, clamped));
    state.thresholdNorm = clamped;

    if (source !== "slider") {
      el.thresholdSlider.value = clamped.toFixed(3);
    }
    syncSliderCssVariables();

    if (source !== "number") {
      const formatted = Math.round(clamped * 1000) % 10 === 0
        ? clamped.toFixed(2)
        : clamped.toFixed(3);
      el.thresholdNumber.value = formatted;
    }

    if (!state.grid) return;

    const maxIdx = state.grid.t_norm.length - 1;
    const gridIdx = Math.min(maxIdx, Math.max(0, Math.round(clamped * maxIdx)));

    const ppv = state.grid.ppv[gridIdx];
    const sens = state.grid.sensitivity[gridIdx];
    const fpr = state.grid.fpr[gridIdx];
    const tp = state.grid.tp[gridIdx];
    const fp = state.grid.fp[gridIdx];
    const fn = state.grid.fn[gridIdx];
    const tRaw = state.grid.t_raw[gridIdx];

    el.rawThresholdPill.textContent = `Normalised: ${clamped.toFixed(3)} (Raw Prob: ${tRaw.toFixed(4)})`;
    el.thresholdSlider.setAttribute(
      "aria-valuetext",
      `Normalised threshold ${clamped.toFixed(3)} (raw probability ${tRaw.toFixed(4)}), PPV ${ppv.toFixed(4)}, Sensitivity ${sens.toFixed(4)}`
    );

    function formatOneInX(rate) {
      const raw = 1.0 / rate;
      if (raw >= 1.0 && raw <= 2.0) {
        const rounded = Math.round(raw * 10) / 10;
        return Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1);
      }
      return String(Math.max(1, Math.round(raw)));
    }

    // Left big text: "1 in X positives will be true positives" + "PPV: 0.xxxx"
    if (ppv > 0) {
      el.ppvBigText.textContent = `1 in ${formatOneInX(ppv)} positives will be true positives`;
    } else {
      el.ppvBigText.textContent = `0 positives will be true positives`;
    }
    el.ppvSmallText.textContent = `PPV: ${ppv.toFixed(4)}`;
    el.ppvCountsText.textContent = `TP: ${tp.toLocaleString()} / Predicted Positives: ${(tp + fp).toLocaleString()}`;

    // Right big text: "1 in X true cases will be detected" + "Sensitivity: 0.xxxx"
    if (sens > 0) {
      el.sensBigText.textContent = `1 in ${formatOneInX(sens)} true cases will be detected`;
    } else {
      el.sensBigText.textContent = `0 true cases will be detected`;
    }
    el.sensSmallText.textContent = `Sensitivity: ${sens.toFixed(4)}`;
    el.sensCountsText.textContent = `TP: ${tp.toLocaleString()} / Total True Cases: ${(tp + fn).toLocaleString()}`;

    // Update live circles/dots on AUROC and AUPRC curves
    updateDotOnSvg("roc", fpr, sens);
    updateDotOnSvg("pr", sens, ppv);

    el.rocCoordBadge.textContent = `FPR: ${fpr.toFixed(3)}, Sens: ${sens.toFixed(3)}`;
    el.prCoordBadge.textContent = `Sens: ${sens.toFixed(3)}, PPV: ${ppv.toFixed(3)}`;

    // Update patient prediction box based on threshold
    updatePatientPredictionBox();
  }

  function updatePatientPredictionBox() {
    const sample = state.activeSample;
    if (!sample) {
      el.predictionBox.className = "prediction-box prediction-neutral";
      el.predictionStatusText.textContent = "No sample selected";
      el.predictionScoresText.textContent = "Normalised score: — | Raw probability: —";
      el.sampleScoreMarker.classList.add("hidden");
      return;
    }

    // Show patient's score marker on the slider track
    const pct = Math.max(0, Math.min(100, sample.prob_norm * 100));
    el.sampleScoreMarker.style.insetInlineStart = `${pct}%`;
    el.sampleMarkerLabel.textContent = `Row #${sample.row_index} (${sample.prob_norm.toFixed(2)})`;
    el.sampleScoreMarker.classList.remove("hidden");

    const isResistantPredicted = sample.prob_norm >= state.thresholdNorm;

    if (isResistantPredicted) {
      el.predictionBox.className = "prediction-box prediction-resistant";
      el.predictionStatusText.textContent = "Resistance predicted";
    } else {
      el.predictionBox.className = "prediction-box prediction-susceptible";
      el.predictionStatusText.textContent = "No resistance predicted";
    }

    el.predictionScoresText.textContent =
      `Sample Norm Score: ${sample.prob_norm.toFixed(4)} | Raw Prob: ${sample.prob_raw.toFixed(4)}`;
  }

  function resetRevealBox() {
    state.labelRevealed = false;
    el.revealLabelBox.className = "reveal-box unrevealed";
    el.revealLabelBox.setAttribute("aria-pressed", "false");
    el.revealLabelText.textContent = "Reveal label";
    el.revealLabelHint.textContent = state.activeSample
      ? "Click to reveal whether this sample is resistant or non-resistant"
      : "Select a sample first";
  }

  function toggleRevealLabel() {
    const sample = state.activeSample;
    if (!sample) return;

    state.labelRevealed = !state.labelRevealed;
    if (!state.labelRevealed) {
      resetRevealBox();
      return;
    }

    el.revealLabelBox.setAttribute("aria-pressed", "true");
    if (sample.true_label === 1) {
      el.revealLabelBox.className = "reveal-box revealed-resistant";
      el.revealLabelText.textContent = "Resistant organism";
    } else {
      el.revealLabelBox.className = "reveal-box revealed-nonresistant";
      el.revealLabelText.textContent = "Non-resistant organism";
    }
    el.revealLabelHint.textContent = "Click again to hide ground truth label";
  }

  function renderEmptyTableRow(message) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 3;
    td.className = "empty-table-msg";
    td.textContent = message;
    tr.appendChild(td);
    el.pivotedTableBody.replaceChildren(tr);
  }

  function renderPivotedTable() {
    const sample = state.activeSample;
    if (!sample || !sample.pivoted_rows) {
      renderEmptyTableRow('Click "Get a fresh sample" to inspect a patient\'s feature values.');
      return;
    }

    const scope = el.featureScopeSelect.value;
    const query = (el.featureSearchInput.value || "").trim().toLowerCase();

    const filtered = sample.pivoted_rows.filter((row) => {
      if (scope === "model" && !row.is_model_feature) return false;
      if (query && !row.feature.toLowerCase().includes(query) && !String(row.value).toLowerCase().includes(query)) {
        return false;
      }
      return true;
    });

    el.tableSampleSubtitle.textContent =
      `Row #${sample.row_index} (Sample ID: ${sample.sample_id}) — Showing ${filtered.length} of ${sample.pivoted_rows.length} columns pivoted as rows`;

    if (filtered.length === 0) {
      renderEmptyTableRow("No features match your filter criteria.");
      return;
    }

    const fragment = document.createDocumentFragment();
    for (const item of filtered) {
      const tr = document.createElement("tr");

      const tdRank = document.createElement("td");
      tdRank.className = "col-rank";
      const rankSpan = document.createElement("span");
      if (item.rank) {
        rankSpan.className = "rank-pill";
        rankSpan.textContent = `#${item.rank}`;
      } else {
        rankSpan.style.color = "#64748b";
        rankSpan.style.fontSize = "0.75rem";
        rankSpan.textContent = "—";
      }
      tdRank.appendChild(rankSpan);

      const thFeature = document.createElement("th");
      thFeature.scope = "row";
      thFeature.className = "col-feature";
      thFeature.textContent = item.feature;

      const tdValue = document.createElement("td");
      tdValue.className = "col-value";
      tdValue.textContent = String(item.value);

      tr.append(tdRank, thFeature, tdValue);
      fragment.appendChild(tr);
    }

    el.pivotedTableBody.replaceChildren(fragment);
  }

  function updateSampleHistoryDropdown() {
    if (state.samplesHistory.length === 0) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = 'Click "Get a fresh sample"…';
      el.sampleHistorySelect.replaceChildren(opt);
      return;
    }

    const fragment = document.createDocumentFragment();
    state.samplesHistory.forEach((s, idx) => {
      const opt = document.createElement("option");
      opt.value = String(s.row_index);
      opt.selected = Boolean(state.activeSample && state.activeSample.row_index === s.row_index);
      opt.textContent = `Sample #${idx + 1} — Row ${s.row_index} (Score: ${s.prob_norm.toFixed(3)})`;
      fragment.appendChild(opt);
    });
    el.sampleHistorySelect.replaceChildren(fragment);
  }

  function setActiveSample(sample) {
    state.activeSample = sample;
    el.rowIndexInput.value = sample.row_index;
    el.generateShapBtn.disabled = false;

    resetRevealBox();
    updateSampleHistoryDropdown();
    updatePatientPredictionBox();
    renderPivotedTable();

    // Check if we already generated a SHAP waterfall for this sample
    const cacheKey = `${sample.split}:${sample.row_index}`;
    if (state.shapCache[cacheKey]) {
      displayShapResult(state.shapCache[cacheKey]);
    } else {
      el.shapLoadingContainer.classList.add("hidden");
      el.shapResult.classList.add("hidden");
      el.shapPlaceholder.classList.remove("hidden");
    }
  }

  async function fetchSample(rowIndex = null, targetLabel = null) {
    if (!state.currentSplit) return;
    el.freshSampleBtn.disabled = true;
    el.freshPositiveBtn.disabled = true;
    el.freshNegativeBtn.disabled = true;

    try {
      const params = new URLSearchParams({ split: state.currentSplit });
      if (rowIndex !== null && rowIndex !== "") {
        params.set("row_index", String(rowIndex));
      } else {
        if (targetLabel !== null && targetLabel !== undefined) {
          params.set("label", String(targetLabel));
        }
        const existingIndices = state.samplesHistory.map((s) => s.row_index).join(",");
        if (existingIndices) {
          params.set("exclude", existingIndices);
        }
      }

      const res = await fetch(`/api/sample?${params.toString()}`);
      const data = await res.json();

      if (!res.ok || data.status !== "ok") {
        throw new Error(data.message || "Failed to fetch sample");
      }

      const existingIdx = state.samplesHistory.findIndex((s) => s.row_index === data.row_index);
      if (existingIdx === -1) {
        state.samplesHistory.unshift(data);
      } else {
        state.samplesHistory[existingIdx] = data;
      }

      setActiveSample(data);
    } catch (err) {
      el.errorBanner.classList.remove("hidden");
      el.errorMessage.textContent = err.message;
    } finally {
      el.freshSampleBtn.disabled = false;
      el.freshPositiveBtn.disabled = false;
      el.freshNegativeBtn.disabled = false;
    }
  }

  function displayShapResult(shapData) {
    el.shapLoadingContainer.classList.add("hidden");
    el.shapPlaceholder.classList.add("hidden");
    el.shapWaterfallImg.src = shapData.image_data_url;
    el.shapResult.classList.remove("hidden");
  }

  function setShapProgress(pct) {
    el.shapProgressBar.value = pct;
    el.shapProgressBar.textContent = `${pct}%`;
    el.shapLoadingPercent.textContent = `${pct}%`;
  }

  async function generateShapWaterfall() {
    const sample = state.activeSample;
    if (!sample) return;

    const cacheKey = `${sample.split}:${sample.row_index}`;
    if (state.shapCache[cacheKey]) {
      displayShapResult(state.shapCache[cacheKey]);
      return;
    }

    el.generateShapBtn.disabled = true;
    el.shapPlaceholder.classList.add("hidden");
    el.shapResult.classList.add("hidden");
    el.shapLoadingContainer.classList.remove("hidden");

    let progress = 8;
    setShapProgress(progress);
    el.shapLoadingText.textContent = `Computing TreeSHAP contributions & waterfall plot for Row #${sample.row_index}…`;

    const progressTimer = setInterval(() => {
      if (progress < 90) {
        progress += Math.max(2, Math.round((90 - progress) * 0.18));
        setShapProgress(progress);
      }
    }, 140);

    try {
      const params = new URLSearchParams({
        split: sample.split,
        row_index: String(sample.row_index),
      });
      const res = await fetch(`/api/shap?${params.toString()}`, { method: "POST" });
      const data = await res.json();

      clearInterval(progressTimer);
      if (!res.ok || data.status !== "ok") {
        throw new Error(data.message || "Failed to generate Shapley waterfall plot");
      }

      setShapProgress(100);

      state.shapCache[cacheKey] = data;
      setTimeout(() => {
        displayShapResult(data);
      }, 160);
    } catch (err) {
      clearInterval(progressTimer);
      el.shapLoadingContainer.classList.add("hidden");
      el.shapPlaceholder.classList.remove("hidden");
      el.shapPlaceholder.textContent = `Error generating Shapley waterfall: ${err.message}`;
    } finally {
      el.generateShapBtn.disabled = false;
    }
  }

  async function loadSplitState(splitKey = null) {
    el.errorBanner.classList.add("hidden");

    try {
      const query = splitKey ? `?split=${encodeURIComponent(splitKey)}` : "";
      const res = await fetch(`/api/state${query}`);
      const data = await res.json();

      if (!res.ok || data.status !== "ok") {
        throw new Error(data.message || "Unable to load model and dataset state");
      }

      state.currentSplit = data.current_split;
      state.thresholdMin = data.threshold_min;
      state.thresholdMax = data.threshold_max;
      state.auroc = data.auroc;
      state.auprc = data.auprc;
      state.prevalence = data.prevalence;
      state.grid = data.grid;
      state.rocCurve = data.roc_curve;
      state.prCurve = data.pr_curve;
      state.samplesHistory = [];
      state.activeSample = null;

      // Populate split dropdown
      if (data.available_splits && data.available_splits.length > 0) {
        const splitFrag = document.createDocumentFragment();
        for (const sp of data.available_splits) {
          const opt = document.createElement("option");
          opt.value = sp.key;
          opt.selected = sp.key === data.current_split;
          opt.textContent = sp.label;
          splitFrag.appendChild(opt);
        }
        el.splitSelect.replaceChildren(splitFrag);
      }

      // Header pills
      el.metaRows.textContent = `${data.n_samples.toLocaleString()} samples (${data.x_file})`;
      el.metaPrevalence.textContent = `Resistant: ${data.n_positives.toLocaleString()} (${(data.prevalence * 100).toFixed(1)}%)`;
      el.metaFeatures.textContent = `${data.n_features_model} model features / ${data.n_features_total} total cols`;
      el.rowIndexInput.max = String(Math.max(0, data.n_samples - 1));

      // Scale min/max labels
      el.scaleMinLabel.textContent = `0.00 (min: ${data.threshold_min.toFixed(4)})`;
      el.scaleMaxLabel.textContent = `1.00 (max: ${data.threshold_max.toFixed(4)})`;

      // AUROC & AUPRC badges and SVG curves
      el.aurocBadge.textContent = `AUROC: ${data.auroc.toFixed(4)}`;
      el.auprcBadge.textContent = `AUPRC: ${data.auprc.toFixed(4)}`;

      buildCurveSvg(
        el.rocSvg,
        data.roc_curve.fpr,
        data.roc_curve.tpr,
        "False Positive Rate (1 - Specificity)",
        "Sensitivity (True Positive Rate)",
        "diagonal",
        0,
        "#2563eb",
        "roc"
      );

      buildCurveSvg(
        el.prSvg,
        data.pr_curve.recall,
        data.pr_curve.precision,
        "Sensitivity (Recall)",
        "Positive Predictive Value (PPV)",
        "horizontal",
        data.prevalence,
        "#0d9488",
        "pr"
      );

      updateThresholdUI(state.thresholdNorm, "init");

      // Automatically draw an initial sample so the dashboard is immediately populated
      await fetchSample();
    } catch (err) {
      el.errorBanner.classList.remove("hidden");
      el.errorMessage.textContent = err.message;
    }
  }

  // Event Listeners
  el.thresholdSlider.addEventListener("input", (e) => {
    updateThresholdUI(e.target.value, "slider");
  });

  el.thresholdNumber.addEventListener("input", (e) => {
    if (e.target.value === "") return;
    updateThresholdUI(e.target.value, "number");
  });

  el.thresholdNumber.addEventListener("change", (e) => {
    updateThresholdUI(e.target.value, "init");
  });

  el.stepUpBtn.addEventListener("click", () => {
    const nextVal = Math.min(1.0, Math.round((state.thresholdNorm + 0.01) * 1000) / 1000);
    updateThresholdUI(nextVal, "init");
  });

  el.stepDownBtn.addEventListener("click", () => {
    const nextVal = Math.max(0.0, Math.round((state.thresholdNorm - 0.01) * 1000) / 1000);
    updateThresholdUI(nextVal, "init");
  });

  el.freshSampleBtn.addEventListener("click", () => {
    fetchSample();
  });

  el.freshPositiveBtn.addEventListener("click", () => {
    fetchSample(null, 1);
  });

  el.freshNegativeBtn.addEventListener("click", () => {
    fetchSample(null, 0);
  });

  el.sampleHistorySelect.addEventListener("change", (e) => {
    const rowIdx = parseInt(e.target.value, 10);
    if (Number.isNaN(rowIdx)) return;
    const found = state.samplesHistory.find((s) => s.row_index === rowIdx);
    if (found) {
      setActiveSample(found);
    } else {
      fetchSample(rowIdx);
    }
  });

  el.loadRowBtn.addEventListener("click", () => {
    const val = el.rowIndexInput.value.trim();
    if (val !== "") {
      fetchSample(parseInt(val, 10));
    }
  });

  el.rowIndexInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.isComposing) {
      const val = el.rowIndexInput.value.trim();
      if (val !== "") {
        fetchSample(parseInt(val, 10));
      }
    }
  });

  el.revealLabelBox.addEventListener("click", () => {
    toggleRevealLabel();
  });

  el.featureScopeSelect.addEventListener("change", () => {
    renderPivotedTable();
  });

  el.featureSearchInput.addEventListener("input", () => {
    renderPivotedTable();
  });

  el.generateShapBtn.addEventListener("click", () => {
    generateShapWaterfall();
  });

  el.splitSelect.addEventListener("change", (e) => {
    loadSplitState(e.target.value);
  });

  // Initialise dashboard
  syncSliderCssVariables();
  loadSplitState();
})();
