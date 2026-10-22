"use strict";

/* Pit Wall: a small client for the F1 pit strategy API.
 * Every value that comes from the user or the server is written with textContent or setAttribute,
 * never parsed as HTML, so nothing here can inject markup. */

const SVG_NS = "http://www.w3.org/2000/svg";

// ---------------------------------------------------------------- readable names
// The API speaks in variable names (tyre_age). People should never see them: every name that reaches the
// screen goes through featureLabel() or humanizeText().
const FEATURE_LABELS = {
  tyre_age: "Tyre Age",
  compound: "Compound",
  stint_number: "Stint Number",
  lap_number: "Lap Number",
  race_progress: "Race Progress",
  track_temperature: "Track Temperature",
  air_temperature: "Air Temperature",
  humidity: "Humidity",
  rainfall: "Rainfall",
  wind_speed: "Wind Speed",
  track_temp_delta: "Track Temp Delta",
  prev_lap_delta: "Previous Lap Delta",
  pace_decay_slope: "Pace Decay Slope",
  prev_gap_to_best: "Previous Gap To Best",
  recent_gap_to_best: "Recent Gap To Best",
  previous_laps: "Earlier Laps",
  lap_duration: "Lap Time",
  race_length: "Race Length",
};

const SMALL_WORDS = new Set(["to", "of", "and", "the"]);

function titleCase(name) {
  return String(name)
    .split("_")
    .filter(Boolean)
    .map((word, i) => (i && SMALL_WORDS.has(word) ? word : word.charAt(0).toUpperCase() + word.slice(1)))
    .join(" ");
}

function featureLabel(name) {
  return Object.prototype.hasOwnProperty.call(FEATURE_LABELS, name) ? FEATURE_LABELS[name] : titleCase(name);
}

/* Replace any snake_case token inside a sentence, such as an API error message, with its readable name. */
function humanizeText(text) {
  return String(text).replace(/\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b/g, (token) => featureLabel(token));
}

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

const fmt = (value, digits = 2) => (Number.isFinite(value) ? value.toFixed(digits) : "N/A");
const pct = (value, digits = 0) => (Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "N/A");
const count = (value) => (Number.isFinite(value) ? value.toLocaleString("en-US") : "N/A");
const signed = (value, digits = 2) => (Number.isFinite(value) ? `${value > 0 ? "+" : ""}${value.toFixed(digits)}` : "N/A");
const capitalize = (text) => String(text).replace(/\b[a-z]/g, (c) => c.toUpperCase());

const ALGORITHM_LABELS = { lightgbm: "LightGBM", random_forest: "Random Forest" };
const algorithmLabel = (name) => (Object.prototype.hasOwnProperty.call(ALGORITHM_LABELS, name) ? ALGORITHM_LABELS[name] : titleCase(name));
const compoundLabel = (name) => capitalize(String(name).toLowerCase());
const callLabel = (recommended) => (recommended ? "Box" : "Stay Out");

/* Drift tests arrive as library strings such as "Wasserstein distance (normed)" or "K-S p_value". */
const methodLabel = (method) => capitalize(humanizeText(method));

function formatTime(iso) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return String(iso);
  return date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

// ---------------------------------------------------------------- API
class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

const CONTAINER_LABELS = { laps: "Row", previous_laps: "Earlier Lap" };
const MAX_ERROR_LINES = 8;

/* ["body", "laps", 2, "tyre_age"] becomes "Row 3, Tyre Age". */
function locLabel(loc) {
  const parts = loc.filter((p) => p !== "body");
  const out = [];
  for (let i = 0; i < parts.length; i += 1) {
    const part = parts[i];
    if (typeof part === "number") continue;
    if (typeof parts[i + 1] === "number" && Object.prototype.hasOwnProperty.call(CONTAINER_LABELS, part)) {
      out.push(`${CONTAINER_LABELS[part]} ${parts[i + 1] + 1}`);
      i += 1;
    } else {
      out.push(featureLabel(part));
    }
  }
  return out.join(", ");
}

function describeDetail(detail) {
  if (typeof detail === "string") return humanizeText(detail);
  if (Array.isArray(detail)) {
    const lines = detail.map((item) => {
      const where = Array.isArray(item.loc) ? locLabel(item.loc) : "";
      return where ? `${where}: ${humanizeText(item.msg)}` : humanizeText(item.msg);
    });
    if (lines.length > MAX_ERROR_LINES) {
      const extra = lines.length - MAX_ERROR_LINES;
      return [...lines.slice(0, MAX_ERROR_LINES), `${extra} more ${extra === 1 ? "problem" : "problems"} not shown.`].join("\n");
    }
    return lines.join("\n");
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

// ---------------------------------------------------------------- building blocks
function pageHead(title, lede) {
  return h("header", { class: "page-head" }, h("h1", null, title), h("p", { class: "lede" }, lede));
}

function panel(title, ...children) {
  return h("section", { class: "panel" }, title ? h("div", { class: "panel-head" }, h("h2", null, title)) : null, children);
}

/* items: [label, value][]. The value is drawn above its label; the label comes first in the DOM for screen readers. */
function scoreboard(items) {
  return h("dl", { class: "scoreboard" },
    items.map(([label, value]) => h("div", { class: "stat" }, h("dt", null, label), h("dd", null, value))));
}

const withUnit = (value, unit) => [value, h("small", null, unit)];

function disclosure(title, content, open) {
  return h("details", { class: "disclosure", open: Boolean(open) }, h("summary", null, title), content);
}

function tag(kind, text) {
  return h("span", { class: `tag ${kind}` }, text);
}

function callTag(recommended) {
  return recommended ? tag("box", "Box") : tag("stay", "Stay Out");
}

function compoundCell(compound) {
  const kind = String(compound).toLowerCase();
  return h("span", { class: "compound" }, h("span", { class: `tyre ${kind}`, "aria-hidden": "true" }), compoundLabel(compound));
}

/* align: "l" or "r" per column, defaulting to left for the first column and right for the rest. */
function dataTable(headers, rows, label, align) {
  const side = (i) => ((align ? align[i] : i ? "r" : "l") === "r" ? "num" : "");
  return h("div", { class: "table-wrap", role: "region", "aria-label": label, tabindex: 0 },
    h("table", null,
      h("thead", null, h("tr", null, headers.map((text, i) => h("th", { class: side(i), scope: "col" }, text)))),
      h("tbody", null, rows.map((row) => h("tr", null, row.map((cell, i) => h("td", { class: side(i) }, cell)))))));
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

/* spec: { xs, series: [{ cls, values }], bands: [lap numbers], threshold, yMin, yMax, yLabel, label, tip(index) -> string[] }
 * The chart is drawn at the width of its container and redrawn when that changes, so text stays a readable size.
 * Hover moves a guide line; with the keyboard, the arrow keys, Home and End step through the laps. */
function lineChart(spec) {
  const xs = spec.xs;
  const box = h("div", { class: "chart" });
  const tip = h("div", { class: "tip", role: "status", hidden: true });
  let drawnWidth = 0;
  let activeIndex = null;

  const draw = (width) => {
    drawnWidth = width;
    const narrow = width < 520;
    const height = narrow ? 220 : 260;
    const m = { l: 44, r: 12, t: 14, b: 30 };
    const all = spec.series.flatMap((ser) => ser.values).filter(Number.isFinite);
    if (spec.threshold !== undefined) all.push(spec.threshold);
    const yMin = spec.yMin !== undefined ? spec.yMin : Math.min(...all, 0);
    let yMax = spec.yMax !== undefined ? spec.yMax : Math.max(...all, 0.1);
    if (yMax === yMin) yMax = yMin + 1;
    if (spec.yMax === undefined) yMax += (yMax - yMin) * 0.08;
    const xMin = xs[0];
    const xMax = xs[xs.length - 1];
    const xAt = (x) => (xMax === xMin ? (m.l + width - m.r) / 2 : m.l + ((x - xMin) / (xMax - xMin)) * (width - m.l - m.r));
    const yAt = (y) => m.t + (1 - (y - yMin) / (yMax - yMin)) * (height - m.t - m.b);
    const step = xs.length > 1 ? xAt(xs[1]) - xAt(xs[0]) : 24;

    const svg = s("svg", {
      viewBox: `0 0 ${width} ${height}`, role: "img", tabindex: 0,
      "aria-label": `${spec.label}. Use the left and right arrow keys to read individual laps.`,
    });
    for (const t of niceTicks(yMin, yMax, 4)) {
      svg.append(s("line", { class: "grid-line", x1: m.l, x2: width - m.r, y1: yAt(t), y2: yAt(t) }));
      svg.append(s("text", { class: "tick", x: m.l - 6, y: yAt(t) + 4, "text-anchor": "end" }, String(Number(t.toFixed(2)))));
    }
    const xTicks = xs.length <= (narrow ? 8 : 12) ? xs : niceTicks(xMin, xMax, narrow ? 5 : 8).filter((t) => Number.isInteger(t));
    for (const t of xTicks) {
      svg.append(s("text", { class: "tick", x: xAt(t), y: height - 12, "text-anchor": "middle" }, t));
    }
    svg.append(s("line", { class: "axis", x1: m.l, x2: width - m.r, y1: height - m.b, y2: height - m.b }));
    svg.append(s("text", { class: "tick", x: width - m.r, y: height - 1, "text-anchor": "end" }, "Lap"));
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

    const showIndex = (index) => {
      activeIndex = index;
      guide.setAttribute("x1", xAt(xs[index]));
      guide.setAttribute("x2", xAt(xs[index]));
      guide.setAttribute("visibility", "visible");
      clear(tip);
      spec.tip(index).forEach((line, i) => {
        if (i) tip.append(h("br"));
        tip.append(line);
      });
      tip.hidden = false;
      tip.style.left = `${Math.min(Math.max(xAt(xs[index]) + 10, 0), Math.max(width - tip.offsetWidth - 4, 0))}px`;
      tip.style.top = "8px";
    };
    const hide = () => {
      activeIndex = null;
      guide.setAttribute("visibility", "hidden");
      tip.hidden = true;
    };
    svg.addEventListener("mousemove", (event) => {
      const rect = svg.getBoundingClientRect();
      const px = (event.clientX - rect.left) * (width / rect.width);
      let best = 0;
      xs.forEach((x, i) => {
        if (Math.abs(xAt(x) - px) < Math.abs(xAt(xs[best]) - px)) best = i;
      });
      showIndex(best);
    });
    svg.addEventListener("mouseleave", hide);
    svg.addEventListener("blur", hide);
    svg.addEventListener("keydown", (event) => {
      const last = xs.length - 1;
      const moves = {
        ArrowLeft: Math.max((activeIndex === null ? last + 1 : activeIndex) - 1, 0),
        ArrowRight: Math.min((activeIndex === null ? -1 : activeIndex) + 1, last),
        Home: 0,
        End: last,
      };
      if (!Object.prototype.hasOwnProperty.call(moves, event.key)) return;
      event.preventDefault();
      showIndex(moves[event.key]);
    });
    clear(box);
    box.append(svg, tip);
    tip.hidden = true;
  };

  draw(640);
  if (typeof ResizeObserver === "function") {
    new ResizeObserver((entries) => {
      const width = Math.round(entries[0].contentRect.width);
      if (width > 0 && Math.abs(width - drawnWidth) > 1) draw(width);
    }).observe(box);
  }
  return box;
}

// ---------------------------------------------------------------- decision readout
function probabilityTrack(probability, threshold) {
  const fill = h("div", { class: "track-fill" });
  fill.style.width = `${Math.min(Math.max(probability, 0), 1) * 100}%`;
  const mark = h("div", { class: "track-mark" });
  mark.style.left = `${Math.min(Math.max(threshold, 0), 1) * 100}%`;
  const label = h("span", { class: "t-mark" }, `Threshold ${pct(threshold, 1)}`);
  label.style.left = `${Math.min(Math.max(threshold, 0.12), 0.88) * 100}%`;
  return h("div", { class: "track-wrap" },
    h("div", { class: "track", role: "img", "aria-label": `Pit probability ${pct(probability, 1)}, threshold ${pct(threshold, 1)}` }, fill, mark),
    h("div", { class: "track-legend" }, h("span", { class: "t-start" }, "0%"), label, h("span", { class: "t-end" }, "100%")));
}

function emptyReadout(text) {
  return h("div", { class: "readout idle" },
    h("p", { class: "call" }, "No Prediction Yet"),
    h("p", { class: "call-note" }, text));
}

function renderPrediction(r, usedHistory) {
  const box = r.pit_recommended;
  const note = box
    ? `Pit probability ${pct(r.pit_probability, 1)} reaches the ${pct(r.threshold, 1)} threshold.`
    : `Pit probability ${pct(r.pit_probability, 1)} is below the ${pct(r.threshold, 1)} threshold.`;
  const children = [
    h("div", null,
      h("p", { class: "call" }, box ? "Box This Lap" : "Stay Out"),
      h("p", { class: "call-note" }, note)),
    probabilityTrack(r.pit_probability, r.threshold),
    scoreboard([["Pace Loss", withUnit(fmt(r.pace_loss_s), "s")], ["Pit Probability", pct(r.pit_probability, 1)]]),
    h("p", { class: "hint" }, usedHistory
      ? "Lag features were computed from your earlier laps."
      : "No earlier laps were given, so the lag features are missing and the estimate is less reliable."),
  ];
  for (const warning of r.warnings || []) children.push(h("div", { class: "notice" }, humanizeText(warning)));
  if (r.computed_features) {
    const rows = Object.entries(r.computed_features).map(([k, v]) => [featureLabel(k), v === null ? "Not Enough Laps" : fmt(v, 3)]);
    children.push(disclosure("Computed Lag Features", dataTable(["Feature", "Value"], rows, "Computed lag features")));
  }
  children.push(h("p", { class: "mono" }, `Request ${r.request_id}`));
  return h("div", { class: "readout", "data-call": box ? "box" : "stay" }, children);
}

// ---------------------------------------------------------------- page: predict
const PREDICT_GROUPS = [
  { title: "Tyres And Stint", fields: ["compound", "tyre_age", "stint_number"] },
  { title: "Race Position", fields: ["lap_number", "race_laps"] },
  { title: "Conditions", fields: ["track_temperature", "air_temperature", "humidity", "wind_speed"] },
];

const PREDICT_FIELDS = {
  tyre_age: { label: "Tyre Age (Laps)", min: 0, max: 100, step: 1, hint: "Including laps on the set before the stint" },
  stint_number: { label: "Stint Number", min: 1, max: 10, step: 1 },
  lap_number: { label: "Lap Number", min: 1, max: 120, step: 1 },
  race_laps: { label: "Race Length (Laps)", min: 1, max: 120, step: 1, hint: "Used to compute race progress" },
  track_temperature: { label: "Track Temperature (°C)", min: -10, max: 90, step: "any" },
  air_temperature: { label: "Air Temperature (°C)", min: -30, max: 60, step: "any" },
  humidity: { label: "Humidity (%)", min: 0, max: 100, step: "any" },
  wind_speed: { label: "Wind Speed (m/s)", min: 0, max: 60, step: "any" },
};

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
      throw new Error(`Earlier Laps, line ${i + 1}: use "lap number, lap time in seconds".`);
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
  const result = h("div", { "aria-live": "polite" }, emptyReadout("Enter a lap and press Predict, or load the example."));
  const inputs = {};
  const numberField = (id) => {
    const f = PREDICT_FIELDS[id];
    const input = h("input", {
      id, name: id, type: "number", min: f.min, max: f.max, step: f.step, required: true, inputmode: "decimal",
    });
    inputs[id] = input;
    return h("div", { class: "field" }, h("label", { for: id }, f.label), input, f.hint ? h("span", { class: "hint" }, f.hint) : null);
  };
  const compound = h("select", { id: "compound", name: "compound" },
    ["SOFT", "MEDIUM", "HARD"].map((c) => h("option", { value: c }, compoundLabel(c))));
  const raining = h("input", { id: "raining", type: "checkbox" });
  const laps = h("textarea", {
    id: "laps", name: "laps", spellcheck: "false", rows: 6, placeholder: "12, 92.31\n13, 92.40\n14, 92.52",
  });
  const submit = h("button", { type: "submit", class: "btn primary" }, "Predict");
  const example = h("button", { type: "button", class: "btn" }, "Load Example");

  example.addEventListener("click", () => {
    for (const [id, value] of Object.entries(EXAMPLE.values)) inputs[id].value = value;
    compound.value = EXAMPLE.compound;
    raining.checked = false;
    laps.value = EXAMPLE.laps;
  });

  const groups = PREDICT_GROUPS.map((group) => h("fieldset", null,
    h("legend", null, group.title),
    h("div", { class: "fields" },
      group.fields.map((id) => (id === "compound"
        ? h("div", { class: "field" }, h("label", { for: "compound" }, "Compound"), compound)
        : numberField(id))),
      group.title === "Conditions"
        ? h("div", { class: "field check wide" }, raining, h("label", { for: "raining" }, "Raining Now"),
          h("span", { class: "hint" }, "The model is trained on dry races and will warn"))
        : null)));

  const form = h("form", { novalidate: true },
    groups,
    h("fieldset", null,
      h("legend", null, "Earlier Laps (Optional)"),
      h("div", { class: "field" },
        h("label", { for: "laps" }, "Lap Times For This Stint"),
        laps,
        h("span", { class: "hint" }, "One per line: lap number, lap time in seconds. Clean laps only, oldest first. With these, the service works out the lag features itself and predictions are noticeably better."))),
    h("div", { class: "row" }, submit, example));

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
      if (lapNumber > raceLaps) throw new Error("Lap Number cannot be larger than the Race Length.");
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
    if (window.matchMedia("(max-width: 880px)").matches) result.scrollIntoView({ block: "start" });
  });

  view.append(
    pageHead("Predict", "Estimate how much time the tyres are costing on a lap and whether to pit. Add earlier laps of the stint for a better answer."),
    h("div", { class: "split" },
      panel("Lap Details", form),
      h("section", { class: "sticky" }, h("h2", { class: "sr-only" }, "Recommendation"), result)),
  );
}

// ---------------------------------------------------------------- page: batch
/* Accepted column headers are the readable names shown on the Predict page. Units in brackets and the
 * snake_case spelling are accepted too, so a file exported from the training data also works. */
const BATCH_COLUMNS = [
  { key: "compound", required: true, text: true, sample: ["MEDIUM", "SOFT"] },
  { key: "tyre_age", required: true, sample: [14, 8] },
  { key: "stint_number", required: true, sample: [1, 1] },
  { key: "lap_number", required: true, sample: [18, 10] },
  { key: "race_progress", required: true, sample: [0.316, 0.175] },
  { key: "track_temperature", required: true, sample: [38, 38] },
  { key: "air_temperature", required: true, sample: [27, 27] },
  { key: "humidity", required: true, sample: [45, 45] },
  { key: "rainfall", required: false, sample: [0, 0] },
  { key: "wind_speed", required: true, sample: [2.5, 2.5] },
  { key: "track_temp_delta", required: false, sample: [0.4, ""] },
  { key: "prev_lap_delta", required: false, sample: [0.12, ""] },
  { key: "pace_decay_slope", required: false, sample: [0.03, ""] },
  { key: "prev_gap_to_best", required: false, sample: [0.78, ""] },
  { key: "recent_gap_to_best", required: false, sample: [0.69, ""] },
];
const NUMBER_PATTERN = /^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/;
const MAX_CSV_BYTES = 1000000;
const MAX_PROBLEMS = 8;

const normaliseHeader = (text) => text.trim().toLowerCase().replace(/\s*\(.*\)\s*$/, "").replace(/[\s-]+/g, "_");
const HEADER_ALIASES = Object.fromEntries(
  [...BATCH_COLUMNS.map((c) => c.key), "race_length"].flatMap((key) => [[key, key], [normaliseHeader(featureLabel(key)), key]]),
);

/* RFC 4180 style parser: quoted fields, doubled quotes, CRLF, a byte order mark, and comma, semicolon or tab separators. */
function parseCsv(text) {
  const src = text.replace(/^﻿/, "");
  const firstLine = src.split(/\r?\n/, 1)[0];
  const delimiter = [",", ";", "\t"].reduce((best, d) => (firstLine.split(d).length > firstLine.split(best).length ? d : best), ",");
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < src.length; i += 1) {
    const ch = src[i];
    if (quoted) {
      if (ch !== '"') field += ch;
      else if (src[i + 1] === '"') { field += '"'; i += 1; } else quoted = false;
    } else if (ch === '"' && field === "") {
      quoted = true;
    } else if (ch === delimiter) {
      row.push(field);
      field = "";
    } else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && src[i + 1] === "\n") i += 1;
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += ch;
    }
  }
  if (quoted) throw new Error("The file has a quote that is never closed.");
  if (field !== "" || row.length) {
    row.push(field);
    rows.push(row);
  }
  return rows.filter((r) => r.some((cell) => cell.trim() !== ""));
}

/* Turn CSV text into the request body, or throw an Error that lists every problem found (up to a cap). */
function buildBatch(text) {
  const table = parseCsv(text);
  if (table.length < 2) throw new Error("Add a header row and at least one lap.");
  const index = {};
  table[0].forEach((name, i) => {
    const key = HEADER_ALIASES[normaliseHeader(name)];
    if (key && !(key in index)) index[key] = i;
  });
  const missing = BATCH_COLUMNS.filter((c) => c.required && !(c.key in index) && !(c.key === "race_progress" && "race_length" in index));
  if (missing.length) throw new Error(`Missing columns: ${missing.map((c) => featureLabel(c.key)).join(", ")}.`);

  const problems = [];
  const laps = table.slice(1).map((cells, r) => {
    const where = `Row ${r + 1}`;
    const cell = (key) => (key in index ? (cells[index[key]] || "").trim() : "");
    const lap = {};
    const number = (key) => {
      const raw = cell(key);
      if (!NUMBER_PATTERN.test(raw)) {
        problems.push(`${where}, ${featureLabel(key)}: "${raw}" is not a number.`);
        return null;
      }
      return Number(raw);
    };
    for (const col of BATCH_COLUMNS) {
      const raw = cell(col.key);
      if (col.key === "race_progress" && raw === "" && "race_length" in index) continue;
      if (raw === "") {
        if (col.required) problems.push(`${where}, ${featureLabel(col.key)}: a value is required.`);
        continue;
      }
      lap[col.key] = col.text ? raw : number(col.key);
    }
    if (!("race_progress" in lap) && "race_length" in index) {
      const length = number("race_length");
      if (length !== null && length > 0 && Number.isFinite(lap.lap_number)) lap.race_progress = lap.lap_number / length;
      else problems.push(`${where}, Race Length: enter a positive number.`);
    }
    return lap;
  });
  if (problems.length) {
    const shown = problems.slice(0, MAX_PROBLEMS);
    if (problems.length > MAX_PROBLEMS) shown.push(`${problems.length - MAX_PROBLEMS} more problems not shown.`);
    throw new Error(shown.join("\n"));
  }
  return { laps };
}

/* Spreadsheet programs run text that starts with = + - or @ as a formula, so such text gets a leading quote. */
function csvCell(value) {
  let text = String(value);
  if (/^[=+\-@\t\r]/.test(text) && !NUMBER_PATTERN.test(text)) text = `'${text}`;
  return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

const toCsv = (rows) => `${rows.map((row) => row.map(csvCell).join(",")).join("\r\n")}\r\n`;

function saveCsv(filename, rows) {
  const url = URL.createObjectURL(new Blob([toCsv(rows)], { type: "text/csv;charset=utf-8" }));
  const link = h("a", { href: url, download: filename });
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const templateRows = () => [
  BATCH_COLUMNS.map((c) => featureLabel(c.key)),
  ...[0, 1].map((n) => BATCH_COLUMNS.map((c) => c.sample[n])),
];

function renderBatchResult(request, response) {
  const preds = response.predictions;
  const boxes = preds.filter((p) => p.pit_recommended).length;
  const meanLoss = preds.reduce((sum, p) => sum + p.pace_loss_s, 0) / preds.length;
  const warningCounts = new Map();
  for (const p of preds) for (const w of p.warnings || []) warningCounts.set(w, (warningCounts.get(w) || 0) + 1);

  const rows = preds.map((p, i) => {
    const lap = request.laps[i];
    return [i + 1, compoundCell(lap.compound), fmt(lap.tyre_age, 0), fmt(lap.lap_number, 0), fmt(p.pace_loss_s), pct(p.pit_probability, 1), callTag(p.pit_recommended)];
  });
  const download = h("button", { type: "button", class: "btn" }, "Download Results");
  download.addEventListener("click", () => {
    saveCsv("pit-wall-results.csv", [
      ["Row", "Compound", "Tyre Age", "Lap Number", "Pace Loss (s)", "Pit Probability", "Call", "Warnings"],
      ...preds.map((p, i) => [i + 1, compoundLabel(request.laps[i].compound), request.laps[i].tyre_age, request.laps[i].lap_number,
        p.pace_loss_s.toFixed(4), p.pit_probability.toFixed(4), callLabel(p.pit_recommended), (p.warnings || []).map(humanizeText).join(" ")]),
    ]);
  });

  const notices = [...warningCounts].map(([text, count]) => h("div", { class: "notice" }, `${count} of ${preds.length} laps: ${humanizeText(text)}`));
  return h("section", { class: "panel" },
    h("div", { class: "panel-head" }, h("h2", null, "Results"), download),
    scoreboard([
      ["Laps Scored", count(preds.length)], ["Box Calls", boxes], ["Stay Out Calls", preds.length - boxes],
      ["Mean Pace Loss", withUnit(fmt(meanLoss), "s")], ["Threshold", pct(response.threshold, 1)],
    ]),
    notices,
    dataTable(["Row", "Compound", "Tyre Age", "Lap Number", "Pace Loss (s)", "Pit Probability", "Call"], rows, "Batch results", ["l", "l", "r", "r", "r", "r", "l"]),
    h("p", { class: "mono" }, `Request ${response.request_id}`));
}

function renderBatch(view) {
  const csv = h("textarea", { id: "csv", name: "csv", spellcheck: "false", rows: 8, "aria-describedby": "csv-hint" });
  const file = h("input", { id: "csv-file", type: "file", accept: ".csv,text/csv,text/plain" });
  const run = h("button", { type: "submit", class: "btn primary" }, "Score Laps");
  const template = h("button", { type: "button", class: "btn" }, "Download Template");
  const example = h("button", { type: "button", class: "btn" }, "Load Example");
  const out = h("div", { "aria-live": "polite" });

  template.addEventListener("click", () => saveCsv("pit-wall-template.csv", templateRows()));
  example.addEventListener("click", () => { csv.value = toCsv(templateRows()); });
  file.addEventListener("change", async () => {
    const chosen = file.files && file.files[0];
    if (!chosen) return;
    clear(out);
    if (chosen.size > MAX_CSV_BYTES) {
      out.append(errorNotice(new Error("That file is larger than 1 MB. Split it into smaller files.")));
      return;
    }
    csv.value = await chosen.text();
  });

  const form = h("form", { novalidate: true },
    h("div", { class: "field" }, h("label", { for: "csv-file" }, "CSV File"), file),
    h("div", { class: "field" },
      h("label", { for: "csv" }, "Or Paste CSV"),
      csv,
      h("span", { class: "hint", id: "csv-hint" },
        "One lap per row, with a header row. Use the column names from the Predict page; the template shows all of them. Optional columns can be left empty, and Race Length can stand in for Race Progress. Row numbers in messages count laps after the header, starting at 1.")),
    h("div", { class: "row" }, run, template, example));

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    clear(out);
    run.disabled = true;
    try {
      const request = buildBatch(csv.value);
      const response = await postJson("/predict/batch", request);
      out.append(renderBatchResult(request, response));
    } catch (error) {
      out.append(errorNotice(error));
    } finally {
      run.disabled = false;
    }
  });

  view.append(
    pageHead("Batch", "Score many laps at once from a CSV file and download the calls."),
    h("div", { class: "stack" }, panel("Laps File", form), out));
}

// ---------------------------------------------------------------- page: replay
async function renderReplay(view) {
  view.append(pageHead("Stint Replay", "Pick a race, a driver and a stint. The model scores every lap with the features known at that lap, shown against what actually happened."));
  const body = h("div", { class: "stack" }, h("p", { class: "empty" }, "Loading"));
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
      panel(null,
        h("div", { class: "notice info" }, "No replay sample is installed."),
        h("p", { class: "hint" }, "Build one from the processed data, then restart the service:"),
        h("pre", { class: "mono" }, "python -m src.replay.build_sample\ndocker compose restart api")));
    return;
  }

  const meta = data.meta || {};
  const heldOut = meta.split === "test";
  body.append(
    h("div", { class: heldOut ? "notice info" : "notice" },
      heldOut
        ? `These races are from the held-out test split (${(meta.years || []).join(", ")}). The model never trained on them and they were not used to choose between models.`
        : `These races come from the ${meta.split || "unknown"} split, so the model may have seen them in training. Do not read the fit as accuracy.`));

  const overview = h("div", { "aria-live": "polite" });
  body.append(overview);
  Promise.allSettled([api("/replay/summary"), api("/model-info")]).then(([summary, info]) => {
    if (summary.status === "rejected") {
      overview.append(errorNotice(summary.reason));
    } else if (summary.value.available) {
      overview.append(renderOverview(summary.value, info.status === "fulfilled" ? info.value : null));
    }
  });

  const sessionSel = h("select", { id: "session" });
  const driverSel = h("select", { id: "driver" });
  const stintSel = h("select", { id: "stint" });
  const out = h("div", { "aria-live": "polite" });
  body.append(
    h("h2", { class: "section-title" }, "Single Stint"),
    panel(null,
      h("div", { class: "fields three" },
        h("div", { class: "field" }, h("label", { for: "session" }, "Race"), sessionSel),
        h("div", { class: "field" }, h("label", { for: "driver" }, "Driver"), driverSel),
        h("div", { class: "field" }, h("label", { for: "stint" }, "Stint"), stintSel))),
    out);

  const sessions = data.sessions;
  const current = () => sessions.find((x) => String(x.session_key) === sessionSel.value);
  const fill = (select, items) => {
    clear(select);
    items.forEach(([value, text]) => select.append(h("option", { value }, text)));
  };
  const onSession = () => {
    fill(driverSel, current().drivers.map((d) => [d.driver_number, `Car ${d.driver_number}`]));
    onDriver();
  };
  const onDriver = () => {
    const driver = current().drivers.find((d) => String(d.driver_number) === driverSel.value);
    fill(stintSel, driver.stints.map((st) => [st.stint_number, `Stint ${st.stint_number}, ${compoundLabel(st.compound)}, Laps ${st.first_lap} To ${st.last_lap}`]));
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

  // Two races at one circuit in a year (a sprint weekend, say) get a number so they can be told apart.
  const seen = new Map();
  const names = sessions.map((x) => {
    const base = `${x.circuit} ${x.year}`;
    seen.set(base, (seen.get(base) || 0) + 1);
    return [x.session_key, base, seen.get(base)];
  });
  fill(sessionSel, names.map(([key, base, nth]) => [key, seen.get(base) > 1 ? `${base} (${nth})` : base]));
  sessionSel.addEventListener("change", onSession);
  driverSel.addEventListener("change", onDriver);
  stintSel.addEventListener("change", load);
  onSession();
}

function renderOverview(summary, info) {
  const o = summary.overall;
  const reference = (info && info.test_metrics) || {};
  const full = reference.test_regression;
  const fullFlag = reference.test_classification;
  const head = (first) => [first, "Laps", "Mean Abs. Error (s)", "Bias (s)", "Flag Accuracy", "F1 Score"];
  const metricRow = (label, m) => [label, count(m.laps), fmt(m.mae), signed(m.bias), pct(m.flag_accuracy), m.f1 === null ? "N/A" : fmt(m.f1, 2)];
  const optional = (value, render) => (value === null ? "N/A" : render(value));
  return panel("Whole Sample",
    scoreboard([
      ["Laps Scored", count(o.laps)],
      ["Mean Abs. Error", withUnit(fmt(o.mae), "s")],
      ["Bias", withUnit(signed(o.bias), "s")],
      ["R²", optional(o.r2, (v) => fmt(v, 2))],
      ["Flag Accuracy", pct(o.flag_accuracy)],
      ["Precision", optional(o.precision, pct)],
      ["Recall", optional(o.recall, pct)],
      ["PR-AUC", optional(o.pr_auc, (v) => fmt(v, 3))],
    ]),
    h("p", { class: "hint" },
      "Bias is the average of predicted minus actual pace loss: positive means the model overstates tyre wear, negative means it understates it. ",
      `The flag is "box" at probability ${pct(summary.threshold, 1)} or more; ${pct(o.prevalence)} of these laps are labelled box.`),
    full && fullFlag
      ? h("div", { class: "notice info" },
        `Full test set, from the training report: mean absolute error ${fmt(full.mae, 3)} s, R² ${fmt(full.r2, 2)}, PR-AUC ${fmt(fullFlag.pr_auc, 3)}, F1 ${fmt(fullFlag.f1, 3)}. `,
        "They match when the sample holds every test race; a sample with fewer races will differ.")
      : null,
    disclosure("By Race",
      dataTable(head("Race"), summary.by_session.map((x) => metricRow(`${x.circuit} ${x.year}`, x.metrics)), "Results by race"), true),
    disclosure("By Tyre Compound",
      dataTable(head("Compound"), summary.by_compound.map((x) => metricRow(compoundCell(x.compound), x.metrics)), "Results by tyre compound")));
}

function renderStint(stint) {
  const laps = stint.laps;
  const xs = laps.map((l) => l.lap_number);
  const mae = laps.reduce((sum, l) => sum + Math.abs(l.predicted_pace_loss_s - l.actual_pace_loss_s), 0) / laps.length;
  const agree = laps.filter((l) => l.pit_recommended === l.actual_pit_recommended).length / laps.length;
  const firstWhere = (key) => {
    const hit = laps.find((l) => l[key]);
    return hit ? `Lap ${hit.lap_number}` : "Never";
  };
  const bands = laps.filter((l) => l.actual_pit_recommended).map((l) => l.lap_number);

  const paceChart = lineChart({
    xs, label: "Predicted and actual pace loss per lap", yLabel: "Pace Loss (s)",
    series: [
      { cls: "actual", values: laps.map((l) => l.actual_pace_loss_s) },
      { cls: "pred", values: laps.map((l) => l.predicted_pace_loss_s) },
    ],
    tip: (i) => [`Lap ${laps[i].lap_number}, Tyre Age ${fmt(laps[i].tyre_age, 0)}`,
      `Actual ${fmt(laps[i].actual_pace_loss_s)} s`, `Predicted ${fmt(laps[i].predicted_pace_loss_s)} s`],
  });
  const probChart = lineChart({
    xs, label: "Pit probability per lap against the decision threshold", yLabel: "Pit Probability", yMin: 0, yMax: 1,
    series: [{ cls: "pred", values: laps.map((l) => l.pit_probability) }],
    threshold: stint.threshold, bands,
    tip: (i) => [`Lap ${laps[i].lap_number}`, `Probability ${pct(laps[i].pit_probability, 1)}`,
      `Model: ${callLabel(laps[i].pit_recommended)}`, `Label: ${callLabel(laps[i].actual_pit_recommended)}`],
  });

  const rows = laps.map((l) => [
    l.lap_number, fmt(l.tyre_age, 0), fmt(l.actual_pace_loss_s), fmt(l.predicted_pace_loss_s),
    pct(l.pit_probability, 1), callTag(l.pit_recommended), callTag(l.actual_pit_recommended),
  ]);

  return panel(null,
    scoreboard([
      ["Laps", laps.length], ["Mean Abs. Error", withUnit(fmt(mae), "s")], ["Flag Agreement", pct(agree)],
      ["Model First Box", firstWhere("pit_recommended")], ["Label First Box", firstWhere("actual_pit_recommended")],
    ]),
    h("div", { class: "chart-block" },
      h("div", { class: "chart-head" }, h("h2", null, "Pace Loss"),
        h("div", { class: "legend" },
          h("span", null, h("i", { class: "key actual" }), "Actual"), h("span", null, h("i", { class: "key pred" }), "Predicted"))),
      paceChart),
    h("div", { class: "chart-block" },
      h("div", { class: "chart-head" }, h("h2", null, "Pit Probability"),
        h("div", { class: "legend" },
          h("span", null, h("i", { class: "key pred" }), "Model Probability"),
          h("span", null, h("i", { class: "key threshold" }), `Threshold ${pct(stint.threshold, 1)}`),
          h("span", null, h("i", { class: "key band" }), "Laps Labelled Box"))),
      probChart),
    disclosure("Lap Table",
      dataTable(["Lap", "Tyre Age", "Actual (s)", "Predicted (s)", "Pit Probability", "Model Call", "Label"], rows, "Lap by lap results", ["r", "r", "r", "r", "r", "l", "l"])));
}

// ---------------------------------------------------------------- page: model and monitoring
const METRIC_LABELS = {
  mae: "Mean Abs. Error (s)", rmse: "Root Mean Sq. Error (s)", r2: "R²", pr_auc: "PR-AUC", roc_auc: "ROC-AUC",
  precision: "Precision", recall: "Recall", f1: "F1 Score", brier: "Brier Score", prevalence: "Pit Rate", threshold: "Threshold",
};

function metricsBlock(title, metrics) {
  if (!metrics) return null;
  const rows = Object.entries(metrics).filter(([, v]) => typeof v === "number").map(([k, v]) => [METRIC_LABELS[k] || titleCase(k), fmt(v, 3)]);
  return h("div", null, h("h3", { class: "block-title" }, title), dataTable(["Metric", "Value"], rows, title));
}

async function renderModel(view) {
  view.append(pageHead("Model & Monitoring", "What is being served, how it scored on held-out races, and whether the traffic it sees still looks like the data it learned from."));
  const body = h("div", { class: "stack" });
  view.append(body);
  const [info, drift] = await Promise.allSettled([api("/model-info"), api("/monitoring/summary")]);

  const modelCard = panel("Champion Model");
  if (info.status === "rejected") {
    modelCard.append(errorNotice(info.reason));
  } else {
    const m = info.value;
    append(modelCard, [
      scoreboard([
        ["Version", m.champion_version || "N/A"], ["Algorithm", m.algorithm ? algorithmLabel(m.algorithm) : "N/A"],
        ["Pit Threshold", fmt(m.threshold, 3)], ["Input Features", (m.features || []).length]]),
      m.ready ? null : h("div", { class: "notice error" }, "The model is not loaded."),
      m.created_utc ? h("p", { class: "hint" }, `Exported ${formatTime(m.created_utc)}`) : null,
      h("div", { class: "split split-even metrics" },
        metricsBlock("Held-Out Regression", (m.test_metrics || {}).test_regression),
        metricsBlock("Held-Out Pit Flag", (m.test_metrics || {}).test_classification)),
      disclosure("Input Features", h("div", { class: "chips" }, (m.features || []).map((f) => h("span", { class: "chip" }, featureLabel(f))))),
    ]);
  }

  const driftCard = panel("Input Drift");
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
    card.append(h("p", { class: "verdict" }, d.drift_detected ? tag("drift", "Drift Detected") : tag("ok", "No Drift")));
  }
  append(card, [
    scoreboard([
      ["Requests Checked", count(d.n_current)], ["Reference Rows", count(d.n_reference)],
      ["Inputs Drifted", pct(d.share_drifted)], ["Alert Threshold", pct(d.drift_share_threshold)]]),
    d.generated_at ? h("p", { class: "hint" }, `Generated ${formatTime(d.generated_at)}`) : null,
  ]);
  if ((d.drifted_columns || []).length) {
    card.append(h("div", { class: "chips" }, d.drifted_columns.map((c) => h("span", { class: "chip drift" }, featureLabel(c)))));
  }
  const columns = Object.entries(d.columns || {});
  if (columns.length) {
    const rows = columns.map(([name, v]) => [featureLabel(name), methodLabel(v.method), fmt(v.value, 3), fmt(v.threshold, 2), v.drifted ? tag("drift", "Drifted") : tag("ok", "Stable")]);
    card.append(disclosure("Results By Input", dataTable(["Input", "Test", "Score", "Limit", "Result"], rows, "Drift results by input", ["l", "l", "r", "r", "l"]), true));
  }
}

// ---------------------------------------------------------------- shell
const ROUTES = {
  predict: { render: renderPredict, title: "Predict" },
  batch: { render: renderBatch, title: "Batch" },
  replay: { render: renderReplay, title: "Stint Replay" },
  model: { render: renderModel, title: "Model & Monitoring" },
};

async function route() {
  const name = Object.prototype.hasOwnProperty.call(ROUTES, location.hash.slice(1)) ? location.hash.slice(1) : "predict";
  document.querySelectorAll("nav a").forEach((a) => {
    if (a.dataset.route === name) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  document.title = `${ROUTES[name].title} | Pit Wall`;
  const view = document.getElementById("view");
  clear(view);
  window.scrollTo(0, 0);
  await ROUTES[name].render(view);
}

async function checkStatus() {
  const pill = document.getElementById("status");
  const text = document.getElementById("status-text");
  try {
    await api("/ready");
    text.textContent = "Model Ready";
    pill.dataset.state = "ok";
  } catch (error) {
    text.textContent = error.status === 503 ? "Model Not Loaded" : "API Unreachable";
    pill.dataset.state = "bad";
  }
}

window.addEventListener("hashchange", route);
checkStatus();
route();
