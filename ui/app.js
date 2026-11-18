// PathSnap evaluator UI (runs on Laptop 2).
// Owner: Harsh Mannan (PES2UG24CS183)
//
// Talks only to the API: GET /trips, POST /predict. It never reads trip files
// or reference routes itself; every number shown comes from the API response.

(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const params = new URLSearchParams(location.search);
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* storage unavailable */ } },
  };

  // ------------------------------------------------------------------ map
  const map = L.map("map", { preferCanvas: true, zoomControl: true }).setView([20, 0], 2);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19, attribution: "&copy; OpenStreetMap contributors",
  }).addTo(map);
  const layers = {
    gps: L.layerGroup().addTo(map),
    snapped: L.layerGroup().addTo(map),
    route: L.layerGroup().addTo(map),
  };
  L.control.layers(null, {
    "Raw GPS fixes": layers.gps,
    "Matched positions": layers.snapped,
    "Predicted route": layers.route,
  }, { collapsed: false }).addTo(map);

  // ------------------------------------------------------------------ state
  let apiBase = params.get("api") || store.get("pathsnap_api") || "http://localhost:8000";
  $("apiUrl").value = apiBase;
  let busy = false;

  function setStatus(text, cls) {
    const s = $("status");
    s.textContent = text;
    s.className = "status " + cls;
  }

  function logLine(text, isErr) {
    const li = document.createElement("li");
    li.textContent = new Date().toLocaleTimeString() + "  " + text;
    if (isErr) li.className = "err";
    $("reqLog").prepend(li);
  }

  function setBusy(on, text) {
    busy = on;
    $("busy").hidden = !on;
    if (text) $("busyText").textContent = text;
    document.querySelectorAll(".trip").forEach((b) => (b.disabled = on));
  }

  async function api(path, opts) {
    const r = await fetch(apiBase.replace(/\/$/, "") + path, opts);
    if (!r.ok) {
      let msg = r.status + " " + r.statusText;
      try { msg = (await r.json()).detail || msg; } catch (e) { /* not json */ }
      throw new Error(msg);
    }
    return r.json();
  }

  // ------------------------------------------------------------------ trips
  async function connect() {
    apiBase = $("apiUrl").value.trim();
    store.set("pathsnap_api", apiBase);
    setStatus("Connecting...", "idle");
    try {
      const h = await api("/health");
      const t = await api("/trips");
      setStatus(`Connected - ${h.model} - ${t.trips.length} trips`, "ok");
      logLine(`GET /trips -> ${t.trips.length} trips`);
      renderTrips(t.trips);
      const auto = params.get("trip");   // optional ?trip=<id> opens a trip directly
      if (auto && t.trips.some((x) => x.trip_id === auto)) runTrip(auto);
    } catch (e) {
      setStatus("Connection failed: " + e.message, "err");
      logLine("connect failed: " + e.message, true);
    }
  }

  function renderTrips(trips) {
    const box = $("tripList");
    box.innerHTML = "";
    if (!trips.length) {
      box.innerHTML = '<p class="muted">No trips in the API data folder.</p>';
      return;
    }
    trips.forEach((t) => {
      const b = document.createElement("button");
      b.className = "trip";
      b.dataset.id = t.trip_id;
      b.innerHTML = `<div class="id">${t.trip_id}</div>` +
        `<div class="meta">${t.n_obs.toLocaleString()} fixes &middot; ${(t.duration_s / 60).toFixed(0)} min</div>`;
      b.addEventListener("click", () => runTrip(t.trip_id));
      box.appendChild(b);
    });
  }

  // ---------------------------------------------------------------- predict
  async function runTrip(id) {
    if (busy) return;
    document.querySelectorAll(".trip").forEach((b) => b.classList.toggle("active", b.dataset.id === id));
    setBusy(true, `Trip ${id}: candidates -> classifier -> Viterbi...`);
    const t0 = performance.now();
    try {
      const res = await api("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ trip_id: id }),
      });
      const rtt = (performance.now() - t0) / 1000;
      draw(res);
      showMetrics(res, rtt);
      const m = res.metrics;
      logLine(`POST /predict ${id} -> ` +
        (m.observation_accuracy !== undefined ? `acc ${(m.observation_accuracy * 100).toFixed(2)}% ` : "") +
        `(${rtt.toFixed(2)} s)`);
    } catch (e) {
      logLine(`POST /predict ${id} failed: ${e.message}`, true);
      setStatus("Request failed: " + e.message, "err");
    } finally {
      setBusy(false);
    }
  }

  function draw(res) {
    Object.values(layers).forEach((l) => l.clearLayers());
    res.gps.forEach((p) => L.circleMarker(p, { radius: 2.5, color: "#1f77b4", weight: 0, fillOpacity: 0.8 }).addTo(layers.gps));
    res.snapped.forEach((p, i) => {
      if (!p) return;
      L.circleMarker(p, { radius: 2, color: "#ff9f1c", weight: 0, fillOpacity: 0.9 })
        .bindTooltip(`fix ${i} - confidence ${res.confidence[i]}`).addTo(layers.snapped);
    });
    res.route.forEach((line) => L.polyline(line, { color: "#c0392b", weight: 4, opacity: 0.85 }).addTo(layers.route));
    if (res.gps.length) {
      L.marker(res.gps[0], { title: "Start" }).bindTooltip("Start").addTo(layers.route);
      L.marker(res.gps[res.gps.length - 1], { title: "End" }).bindTooltip("End").addTo(layers.route);
      map.fitBounds(L.latLngBounds(res.gps), { padding: [30, 30] });
    }
  }

  function pct(v) { return v === undefined || v === null ? "-" : (v * 100).toFixed(2) + " %"; }

  function showMetrics(res, rtt) {
    const m = res.metrics, t = res.timings;
    const rows = [
      ["Observation accuracy (HMM)", pct(m.observation_accuracy), true],
      ["Classifier-only accuracy", pct(m.classifier_only_accuracy)],
      ["Route precision (length)", pct(m.route_precision)],
      ["Route recall (length)", pct(m.route_recall)],
      ["Route F1", pct(m.route_f1)],
      ["Route continuity", pct(m.route_continuity)],
      ["Candidate miss rate", pct(m.candidate_miss_rate)],
      ["GPS fixes / scored", `${res.n_obs.toLocaleString()} / ${m.scored_observations ?? "-"}`],
      ["Arcs in predicted route", m.n_route_arcs],
      ["Server pipeline time", t.total_s.toFixed(2) + " s"],
      ["Round trip (UI)", rtt.toFixed(2) + " s"],
    ];
    $("metrics").innerHTML = rows.map(([k, v, head]) =>
      `<tr class="${head ? "head" : ""}"><td>${k}</td><td>${v}</td></tr>`).join("");
    $("resTrip").textContent = res.trip_id;
    $("resModel").textContent = `Model: ${res.model}. Accuracy is computed on the API server against its ` +
      "internal reference data; the reference route is never sent to this UI.";
    $("resultCard").hidden = false;
  }

  $("connectBtn").addEventListener("click", connect);
  $("apiUrl").addEventListener("keydown", (e) => { if (e.key === "Enter") connect(); });
  connect();
})();
