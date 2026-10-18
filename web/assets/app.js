"use strict";

/* Pit Wall: a small client for the F1 pit strategy API.
 * Every value that comes from the user or the server is written with textContent or setAttribute,
 * never parsed as HTML, so nothing here can inject markup. */

const SVG_NS = "http://www.w3.org/2000/svg";

// ---------------------------------------------------------------- DOM helpers
function h(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === false || value === null || value === undefined) continue;
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  append(node, children);
  return node;
}

function s(tag, attrs, ...children) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs || {})) node.setAttribute(key, String(value));
  append(node, children);
  return node;
}

function append(node, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

const fmt = (value, digits = 2) => (Number.isFinite(value) ? value.toFixed(digits) : "n/a");
const pct = (value, digits = 0) => (Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "n/a");

// ---------------------------------------------------------------- API
class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

function describeDetail(detail) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        const where = Array.isArray(item.loc) ? item.loc.filter((p) => p !== "body").join(".") : "";
        return where ? `${where}: ${item.msg}` : String(item.msg);
      })
      .join("\n");
  }
  return "Unexpected error";
}

async function api(path, options) {
  let response;
  try {
    response = await fetch(path, options);
  } catch (error) {
    throw new ApiError(0, "Cannot reach the API. Is the server running?");
  }
  let body = null;
  try {
    body = await response.json();
  } catch (error) {
    body = null;
  }
  if (!response.ok) {
    throw new ApiError(response.status, describeDetail(body && body.detail) || `HTTP ${response.status}`);
  }
  return body;
}

function postJson(path, payload) {
  return api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

function errorNotice(error) {
  const lines = String(error.message).split("\n");
  return h("div", { class: "notice error", role: "alert" }, lines.map((line, i) => [i ? h("br") : null, line]));
}

// ---------------------------------------------------------------- charts (SVG, no dependencies)
function niceTicks(min, max, count) {
  const span = max - min || 1;
  const rough = span / count;
  const pow = 10 ** Math.floor(Math.log10(rough));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * pow).find((candidate) => candidate >= rough) || 10 * pow;
  const first = Math.ceil(min / step) * step;
  const ticks = [];
  for (let t = first; t <= max + step * 1e-9; t += step) ticks.push(Math.round(t / step) * step);
  return ticks;
}

/* spec: { xs, series: [{ name, cls, values }], bands: [lap numbers], threshold, yMin, yMax, yLabel,
 *         label, tip(index) -> string[] } */
function lineChart(spec) {
  const width = 640;
  const height = 250;
  const m = { l: 46, r: 14, t: 12, b: 32 };
  const xs = spec.xs;
  const all = spec.series.flatMap((ser) => ser.values).filter(Number.isFinite);
  if (spec.threshold !== undefined) all.push(spec.threshold);
  let yMin = spec.yMin !== undefined ? spec.yMin : Math.min(...all, 0);
  let yMax = spec.yMax !== undefined ? spec.yMax : Math.max(...all, 0.1);
  if (yMax === yMin) yMax = yMin + 1;
  const pad = spec.yMax === undefined ? (yMax - yMin) * 0.08 : 0;
  yMax += pad;
  const xMin = xs[0];
  const xMax = xs[xs.length - 1];
  const xAt = (x) => (xMax === xMin ? (m.l + width - m.r) / 2 : m.l + ((x - xMin) / (xMax - xMin)) * (width - m.l - m.r));
  const yAt = (y) => m.t + (1 - (y - yMin) / (yMax - yMin)) * (height - m.t - m.b);
  const step = xs.length > 1 ? (xAt(xs[1]) - xAt(xs[0])) : 24;

  const svg = s("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": spec.label });
  for (const t of niceTicks(yMin, yMax, 4)) {
    svg.append(s("line", { class: "grid-line", x1: m.l, x2: width - m.r, y1: yAt(t), y2: yAt(t) }));
    svg.append(s("text", { class: "tick", x: m.l - 6, y: yAt(t) + 4, "text-anchor": "end" }, String(Number(t.toFixed(2)))));
  }
  const xTicks = xs.length <= 12 ? xs : niceTicks(xMin, xMax, 8).filter((t) => Number.isInteger(t));
  for (const t of xTicks) {
    svg.append(s("text", { class: "tick", x: xAt(t), y: height - 10, "text-anchor": "middle" }, t));
  }
  svg.append(s("line", { class: "axis", x1: m.l, x2: width - m.r, y1: height - m.b, y2: height - m.b }));
  svg.append(s("text", { class: "tick", x: width - m.r, y: height - 1, "text-anchor": "end" }, "lap"));
  if (spec.yLabel) svg.append(s("text", { class: "tick", x: m.l, y: 9 }, spec.yLabel));

  for (const lap of spec.bands || []) {
    const left = Math.max(xAt(lap) - step / 2, m.l);
    const right = Math.min(xAt(lap) + step / 2, width - m.r);
    svg.append(s("rect", { class: "band", x: left, y: m.t, width: right - left, height: height - m.t - m.b }));
  }
  if (spec.threshold !== undefined) {
    svg.append(s("line", { class: "threshold", x1: m.l, x2: width - m.r, y1: yAt(spec.threshold), y2: yAt(spec.threshold) }));
  }
  for (const ser of spec.series) {
    const points = xs.map((x, i) => `${xAt(x).toFixed(1)},${yAt(ser.values[i]).toFixed(1)}`).join(" ");
    svg.append(s("polyline", { class: `line ${ser.cls}`, points }));
    if (xs.length <= 60) {
      xs.forEach((x, i) => svg.append(s("circle", { class: `dot ${ser.cls}`, cx: xAt(x), cy: yAt(ser.values[i]), r: 2.5 })));
    }
  }

  const guide = s("line", { class: "guide", y1: m.t, y2: height - m.b, x1: 0, x2: 0, visibility: "hidden" });
  svg.append(guide);
  const tip = h("div", { class: "tip", hidden: true });
  const box = h("div", { class: "chart" }, svg, tip);

  const show = (event) => {
    const rect = svg.getBoundingClientRect();
    const scale = width / rect.width;
    const px = (event.clientX - rect.left) * scale;
    let best = 0;
    xs.forEach((x, i) => {
      if (Math.abs(xAt(x) - px) < Math.abs(xAt(xs[best]) - px)) best = i;
    });
    guide.setAttribute("x1", xAt(xs[best]));
    guide.setAttribute("x2", xAt(xs[best]));
    guide.setAttribute("visibility", "visible");
    clear(tip);
    spec.tip(best).forEach((line, i) => {
      if (i) tip.append(h("br"));
      tip.append(line);
    });
    tip.hidden = false;
    const left = Math.min(Math.max(xAt(xs[best]) / scale + 10, 0), rect.width - tip.offsetWidth - 4);
    tip.style.left = `${left}px`;
    tip.style.top = "8px";
  };
  const hide = () => {
    guide.setAttribute("visibility", "hidden");
    tip.hidden = true;
  };
  svg.addEventListener("mousemove", show);
  svg.addEventListener("mouseleave", hide);
  return box;
}

// ---------------------------------------------------------------- shared pieces
function stat(label, value) {
  return h("div", { class: "stat" }, h("div", { class: "v" }, value), h("div", { class: "k" }, label));
}

function dataTable(headers, rows, caption) {
  return h(
    "table",
    null,
    caption ? h("caption", { class: "hint" }, caption) : null,
    h("thead", null, h("tr", null, headers.map((text, i) => h("th", { class: i ? "num" : "" }, text)))),
    h("tbody", null, rows.map((row) => h("tr", null, row.map((cell, i) => h("td", { class: i ? "num" : "" }, cell))))),
  );
}

function probabilityMeter(probability, threshold) {
  const fill = h("div", { class: "meter-fill" });
  fill.style.width = `${Math.min(Math.max(probability, 0), 1) * 100}%`;
  const mark = h("div", { class: "meter-mark", title: `Decision threshold ${fmt(threshold, 3)}` });
  mark.style.left = `${Math.min(Math.max(threshold, 0), 1) * 100}%`;
  return h(
    "div",
    null,
    h("div", { class: "meter", role: "img", "aria-label": `Pit Probability ${pct(probability)}, threshold ${pct(threshold)}` }, fill, mark),
    h("div", { class: "meter-legend" }, h("span", null, "0%"), h("span", null, `Threshold ${pct(threshold, 1)}`), h("span", null, "100%")),
  );
}

// ---------------------------------------------------------------- page: predict
const PREDICT_FIELDS = [
  { id: "tyre_age", label: "Tyre Age (laps)", min: 0, max: 100, step: 1, hint: "Including laps on the set before the stint" },
  { id: "stint_number", label: "Stint Number", min: 1, max: 10, step: 1 },
  { id: "lap_number", label: "Lap Number", min: 1, max: 120, step: 1 },
  { id: "race_laps", label: "Race Length (laps)", min: 1, max: 120, step: 1, hint: "Used to compute race progress" },
  { id: "track_temperature", label: "Track Temp (°C)", min: -10, max: 90, step: "any" },
  { id: "air_temperature", label: "Air Temp (°C)", min: -30, max: 60, step: "any" },
  { id: "humidity", label: "Humidity (%)", min: 0, max: 100, step: "any" },
  { id: "wind_speed", label: "Wind (m/s)", min: 0, max: 60, step: "any" },
];

const EXAMPLE = {
  values: {
    tyre_age: 14, stint_number: 1, lap_number: 18, race_laps: 57,
    track_temperature: 38, air_temperature: 27, humidity: 45, wind_speed: 2.5,
  },
  compound: "MEDIUM",
  laps: "12, 92.31\n13, 92.40\n14, 92.52\n15, 92.49\n16, 92.71\n17, 92.84",
};

function parseEarlierLaps(text, currentLap) {
  const laps = [];
  const lines = text.split("\n");
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i].trim();
    if (!line) continue;
    const parts = line.split(/[\s,;]+/);
    const lap = Number(parts[0]);
    const seconds = Number(parts[1]);
    if (parts.length !== 2 || !Number.isInteger(lap) || !Number.isFinite(seconds)) {
      throw new Error(`Earlier laps, line ${i + 1}: use "lap number, lap time in seconds".`);
    }
    laps.push({ lap_number: lap, lap_duration: seconds });
  }
  for (let i = 1; i < laps.length; i += 1) {
    if (laps[i].lap_number <= laps[i - 1].lap_number) {
      throw new Error("Earlier laps must be in increasing lap order, oldest first.");
    }
  }
  if (laps.length && laps[laps.length - 1].lap_number >= currentLap) {
    throw new Error("Earlier laps must all come before the current lap.");
  }
  return laps;
}

function renderPredict(view) {
  const result = h("div", { "aria-live": "polite" }, h("p", { class: "empty" }, "Fill in the lap and press Predict."));
  const inputs = {};
  const fieldNodes = PREDICT_FIELDS.map((f) => {
    const input = h("input", {
      id: f.id, name: f.id, type: "number", min: f.min, max: f.max, step: f.step, required: true, inputmode: "decimal",
    });
    inputs[f.id] = input;
    return h("div", { class: "field" }, h("label", { for: f.id }, f.label), input, f.hint ? h("span", { class: "hint" }, f.hint) : null);
  });
  const compound = h(
    "select", { id: "compound", name: "compound" },
    ["SOFT", "MEDIUM", "HARD"].map((c) => h("option", { value: c }, c)),
  );
  const raining = h("input", { id: "raining", type: "checkbox" });
  const laps = h("textarea", {
    id: "laps", name: "laps", spellcheck: "false", rows: 6, placeholder: "12, 92.31\n13, 92.40\n14, 92.52",
  });
  const submit = h("button", { type: "submit", class: "primary" }, "Predict");
  const example = h("button", { type: "button" }, "Load Example");

  example.addEventListener("click", () => {
    for (const [id, value] of Object.entries(EXAMPLE.values)) inputs[id].value = value;
    compound.value = EXAMPLE.compound;
    raining.checked = false;
    laps.value = EXAMPLE.laps;
  });

  const form = h(
    "form",
    { novalidate: true },
    h("div", { class: "fields" },
      fieldNodes.slice(0, 4),
      h("div", { class: "field" }, h("label", { for: "compound" }, "Compound"), compound),
      fieldNodes.slice(4),
      h("div", { class: "field check wide" }, raining, h("label", { for: "raining" }, "It is raining"),
        h("span", { class: "hint" }, "The model is trained on dry races and will warn")),
      h("div", { class: "field wide" },
        h("label", { for: "laps" }, "Earlier laps of this stint (optional)"),
        laps,
        h("span", { class: "hint" }, "One per line: lap number, lap time in seconds. Clean laps only, oldest first. With these, the service works out the lag features itself and predictions are noticeably better."))),
    h("div", { class: "row" }, submit, example),
  );

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    clear(result);
    submit.disabled = true;
    try {
      const num = (id) => {
        const value = Number(inputs[id].value);
        if (inputs[id].value.trim() === "" || !Number.isFinite(value)) throw new Error(`${inputs[id].labels[0].textContent}: enter a number.`);
        return value;
      };
      const lapNumber = num("lap_number");
      const raceLaps = num("race_laps");
      if (lapNumber > raceLaps) throw new Error("Lap number cannot be larger than the race length.");
      const context = {
        tyre_age: num("tyre_age"),
        compound: compound.value,
        stint_number: num("stint_number"),
        lap_number: lapNumber,
        race_progress: lapNumber / raceLaps,
        track_temperature: num("track_temperature"),
        air_temperature: num("air_temperature"),
        humidity: num("humidity"),
        rainfall: raining.checked ? 1 : 0,
        wind_speed: num("wind_speed"),
      };
      const earlier = parseEarlierLaps(laps.value, lapNumber);
      const response = earlier.length
        ? await postJson("/predict/from-history", { ...context, previous_laps: earlier })
        : await postJson("/predict", context);
      result.append(renderPrediction(response, earlier.length > 0));
    } catch (error) {
      result.append(errorNotice(error));
    } finally {
      submit.disabled = false;
    }
  });

  view.append(
    h("h1", null, "Predict"),
    h("p", { class: "lede" }, "Estimate how much time the tyres are costing on a lap and whether to pit. Add earlier laps of the stint for a better answer."),
    h("div", { class: "grid" },
      h("section", { class: "card" }, h("h2", null, "Lap"), form),
      h("section", { class: "card" }, h("h2", null, "Result"), result)),
  );
}

function renderPrediction(r, usedHistory) {
  const parts = [
    h("div", { class: "result-head" },
      h("span", { class: r.pit_recommended ? "badge pit" : "badge stay" }, r.pit_recommended ? "BOX THIS LAP" : "STAY OUT"),
      h("span", { class: "hint" }, usedHistory ? "Lag features computed from your earlier laps" : "No lap history: lag features missing")),
    h("div", { class: "stats" }, stat("Pace loss", `${fmt(r.pace_loss_s)} s`), stat("Pit probability", pct(r.pit_probability, 1))),
    probabilityMeter(r.pit_probability, r.threshold),
  ];
  for (const warning of r.warnings || []) parts.push(h("div", { class: "notice" }, warning));
  if (r.computed_features) {
    const rows = Object.entries(r.computed_features).map(([k, v]) => [k, v === null ? "not enough laps" : fmt(v, 3)]);
    parts.push(h("details", null, h("summary", null, "Computed Lag Features"), dataTable(["Feature", "Value"], rows)));
  }
  parts.push(h("p", { class: "mono" }, `request ${r.request_id}`));
  return h("div", null, parts);
}

// ---------------------------------------------------------------- page: replay
async function renderReplay(view) {
  view.append(
    h("h1", null, "Stint Replay"),
    h("p", { class: "lede" }, "Pick a race, a driver and a stint. The model scores every lap with the features known at that lap, shown against what actually happened."),
  );
  const body = h("div", null, h("p", { class: "empty" }, "Loading"));
  view.append(body);

  let data;
  try {
    data = await api("/replay/sessions");
  } catch (error) {
    clear(body);
    body.append(errorNotice(error));
    return;
  }
  clear(body);
  if (!data.available || !data.sessions.length) {
    body.append(
      h("div", { class: "card" },
        h("div", { class: "notice info" }, "No replay sample is installed."),
        h("p", null, "Build one from the processed data, then restart the service:"),
        h("pre", { class: "mono" }, "python -m src.replay.build_sample\ndocker compose up --build -d api")));
    return;
  }

  const meta = data.meta || {};
  const heldOut = meta.split === "test";
  body.append(
    h("div", { class: heldOut ? "notice info" : "notice" },
      heldOut
        ? `These races are from the held-out test split (${(meta.years || []).join(", ")}). The model never trained on them and they were not used to choose between models.`
        : `These races come from the ${meta.split || "unknown"} split, so the model may have seen them in training. Do not read the fit as accuracy.`),
  );

  const sessionSel = h("select", { id: "session" });
  const driverSel = h("select", { id: "driver" });
  const stintSel = h("select", { id: "stint" });
  const out = h("div", { "aria-live": "polite" });
  body.append(
    h("div", { class: "card" },
      h("div", { class: "fields" },
        h("div", { class: "field" }, h("label", { for: "session" }, "Race"), sessionSel),
        h("div", { class: "field" }, h("label", { for: "driver" }, "Driver"), driverSel),
        h("div", { class: "field" }, h("label", { for: "stint" }, "Stint"), stintSel))),
    out,
  );

  const sessions = data.sessions;
  const current = () => sessions.find((x) => String(x.session_key) === sessionSel.value);
  const fill = (select, items) => {
    clear(select);
    items.forEach(([value, text]) => select.append(h("option", { value }, text)));
  };
  const onSession = () => {
    fill(driverSel, current().drivers.map((d) => [d.driver_number, `#${d.driver_number}`]));
    onDriver();
  };
  const onDriver = () => {
    const driver = current().drivers.find((d) => String(d.driver_number) === driverSel.value);
    fill(stintSel, driver.stints.map((st) => [st.stint_number, `Stint ${st.stint_number}: ${st.compound}, laps ${st.first_lap} to ${st.last_lap}`]));
    load();
  };
  const load = async () => {
    clear(out);
    out.append(h("p", { class: "empty" }, "Loading"));
    const query = new URLSearchParams({
      session_key: sessionSel.value, driver_number: driverSel.value, stint_number: stintSel.value,
    });
    try {
      const stint = await api(`/replay/stint?${query}`);
      clear(out);
      out.append(renderStint(stint));
    } catch (error) {
      clear(out);
      out.append(errorNotice(error));
    }
  };

  fill(sessionSel, sessions.map((x) => [x.session_key, `${x.circuit} ${x.year} (session ${x.session_key})`]));
  sessionSel.addEventListener("change", onSession);
  driverSel.addEventListener("change", onDriver);
  stintSel.addEventListener("change", load);
  onSession();
}

function renderStint(stint) {
  const laps = stint.laps;
  const xs = laps.map((l) => l.lap_number);
  const mae = laps.reduce((sum, l) => sum + Math.abs(l.predicted_pace_loss_s - l.actual_pace_loss_s), 0) / laps.length;
  const agree = laps.filter((l) => l.pit_recommended === l.actual_pit_recommended).length / laps.length;
  const firstWhere = (key) => {
    const hit = laps.find((l) => l[key]);
    return hit ? `lap ${hit.lap_number}` : "never";
  };
  const bands = laps.filter((l) => l.actual_pit_recommended).map((l) => l.lap_number);

  const paceChart = lineChart({
    xs, label: "Predicted and actual pace loss per lap", yLabel: "seconds",
    series: [
      { name: "Actual", cls: "actual", values: laps.map((l) => l.actual_pace_loss_s) },
      { name: "Predicted", cls: "pred", values: laps.map((l) => l.predicted_pace_loss_s) },
    ],
    tip: (i) => [`Lap ${laps[i].lap_number}, tyre age ${fmt(laps[i].tyre_age, 0)}`,
      `Actual ${fmt(laps[i].actual_pace_loss_s)} s`, `Predicted ${fmt(laps[i].predicted_pace_loss_s)} s`],
  });
  const probChart = lineChart({
    xs, label: "Pit probability per lap against the decision threshold", yLabel: "probability", yMin: 0, yMax: 1,
    series: [{ name: "Pit probability", cls: "pred", values: laps.map((l) => l.pit_probability) }],
    threshold: stint.threshold, bands,
    tip: (i) => [`Lap ${laps[i].lap_number}`, `Probability ${pct(laps[i].pit_probability, 1)}`,
      laps[i].pit_recommended ? "Model: box" : "Model: stay out",
      laps[i].actual_pit_recommended ? "Label: box" : "Label: stay out"],
  });

  const rows = laps.map((l) => [
    l.lap_number, fmt(l.tyre_age, 0), fmt(l.actual_pace_loss_s), fmt(l.predicted_pace_loss_s),
    pct(l.pit_probability, 1), l.pit_recommended ? "box" : "stay", l.actual_pit_recommended ? "box" : "stay",
  ]);

  return h("div", null,
    h("div", { class: "card" },
      h("div", { class: "stats" },
        stat("Laps", laps.length), stat("Mean Abs. Error", `${fmt(mae)} s`), stat("Flag Agreement", pct(agree)),
        stat("Model First Box", firstWhere("pit_recommended")), stat("Label First Box", firstWhere("actual_pit_recommended"))),
      h("h2", null, "Pace Loss"), paceChart,
      h("div", { class: "legend" },
        h("span", null, h("i", { class: "key actual" }), "Actual"), h("span", null, h("i", { class: "key pred" }), "Predicted")),
      h("h2", null, "Pit Probability"), probChart,
      h("div", { class: "legend" },
        h("span", null, h("i", { class: "key pred" }), "Model Probability"),
        h("span", null, h("i", { class: "key threshold" }), `Threshold ${pct(stint.threshold, 1)}`),
        h("span", null, h("i", { class: "key band" }), "Laps Labelled Box")),
      h("details", null, h("summary", null, "Lap Table"),
        dataTable(["Lap", "Tyre Age", "Actual s", "Predicted s", "P(pit)", "Model", "Label"], rows))));
}

// ---------------------------------------------------------------- page: model and monitoring
const METRIC_LABELS = {
  mae: "MAE (s)", rmse: "RMSE (s)", r2: "R²", pr_auc: "PR-AUC", roc_auc: "ROC-AUC",
  precision: "Precision", recall: "Recall", f1: "F1", brier: "Brier", prevalence: "Pit rate", threshold: "Threshold",
};

function metricsTable(title, metrics) {
  if (!metrics) return null;
  const rows = Object.entries(metrics).filter(([, v]) => typeof v === "number").map(([k, v]) => [METRIC_LABELS[k] || k, fmt(v, 3)]);
  return h("div", null, h("h2", null, title), dataTable(["Metric", "Value"], rows));
}

async function renderModel(view) {
  view.append(h("h1", null, "Model & Monitoring"), h("p", { class: "lede" }, "What is being served, how it scored on held-out races, and whether the traffic it sees still looks like the data it learned from."));
  const body = h("div", { class: "grid" });
  view.append(body);
  const [info, drift] = await Promise.allSettled([api("/model-info"), api("/monitoring/summary")]);

  const modelCard = h("section", { class: "card" }, h("h2", null, "Champion Model"));
  if (info.status === "rejected") {
    modelCard.append(errorNotice(info.reason));
  } else {
    const m = info.value;
    append(modelCard, [
      h("div", { class: "stats" },
        stat("Version", m.champion_version || "n/a"), stat("Algorithm", m.algorithm || "n/a"),
        stat("Threshold", fmt(m.threshold, 3)), stat("Features", (m.features || []).length)),
      m.ready ? null : h("div", { class: "notice error" }, "The model is not loaded."),
      m.created_utc ? h("p", { class: "hint" }, `Exported ${m.created_utc}`) : null,
      metricsTable("Held-out Regression", (m.test_metrics || {}).test_regression),
      metricsTable("Held-out Pit Flag", (m.test_metrics || {}).test_classification),
      h("details", null, h("summary", null, "Input Features"), h("div", { class: "chips" }, (m.features || []).map((f) => h("span", { class: "chip" }, f)))),
    ]);
  }

  const driftCard = h("section", { class: "card" }, h("h2", null, "Input Drift"));
  if (drift.status === "rejected") {
    driftCard.append(errorNotice(drift.reason));
  } else {
    renderDrift(driftCard, drift.value);
  }
  body.append(modelCard, driftCard);
}

function renderDrift(card, d) {
  if (d.status === "no_report") {
    card.append(h("div", { class: "notice info" }, "No drift report yet. It is written by the weekly Airflow run, or on demand with python -m src.monitoring.drift."));
    return;
  }
  if (d.status === "unreadable") {
    card.append(h("div", { class: "notice error" }, "The drift report exists but cannot be read."));
    return;
  }
  if (d.status === "insufficient_data") {
    card.append(h("div", { class: "notice info" }, `Not enough served requests to judge: ${d.n_current} logged, ${d.min_rows} needed.`));
  } else {
    card.append(h("div", { class: "result-head" },
      h("span", { class: d.drift_detected ? "badge pit" : "badge stay" }, d.drift_detected ? "DRIFT DETECTED" : "NO DRIFT")));
  }
  append(card, [
    h("div", { class: "stats" },
      stat("Requests Checked", d.n_current), stat("Reference Rows", d.n_reference),
      stat("Columns Drifted", pct(d.share_drifted)), stat("Alert At", pct(d.drift_share_threshold))),
    d.generated_at ? h("p", { class: "hint" }, `Generated ${d.generated_at}`) : null,
  ]);
  if ((d.drifted_columns || []).length) {
    card.append(h("div", { class: "chips" }, d.drifted_columns.map((c) => h("span", { class: "chip drift" }, c))));
  }
  const columns = Object.entries(d.columns || {});
  if (columns.length) {
    const rows = columns.map(([name, v]) => [name, String(v.method), fmt(v.value, 3), fmt(v.threshold, 2), v.drifted ? "drifted" : "ok"]);
    card.append(h("details", null, h("summary", null, "Per-column results"), dataTable(["Column", "Test", "Value", "Limit", "Result"], rows)));
  }
}

// ---------------------------------------------------------------- shell
const ROUTES = { predict: renderPredict, replay: renderReplay, model: renderModel };

async function route() {
  const name = ROUTES[location.hash.slice(1)] ? location.hash.slice(1) : "predict";
  document.querySelectorAll("nav a").forEach((a) => {
    if (a.dataset.route === name) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  const view = document.getElementById("view");
  clear(view);
  await ROUTES[name](view);
}

async function checkStatus() {
  const pill = document.getElementById("status");
  try {
    await api("/ready");
    pill.textContent = "Model Ready";
    pill.className = "pill ok";
  } catch (error) {
    pill.textContent = error.status === 503 ? "Model Not Loaded" : "API Unreachable";
    pill.className = "pill bad";
  }
}

window.addEventListener("hashchange", route);
checkStatus();
route();
