/* eslint-disable no-undef */
const VEHICLE_COLORS = {
  baseline: "#e74c3c",
  optimized: "#2ecc71",
};

class SimMap {
  constructor(containerId, color, isBaseline = false) {
    this.containerId = containerId;
    this.color = color;
    this.isBaseline = isBaseline;
    this.vehicleMarkers = {};
    this.bboxRect = null;
    this._ready = false;

    const container = document.getElementById(containerId);
    if (!container) {
      throw new Error(`Map container not found: ${containerId}`);
    }

    this.map = L.map(containerId, {
      zoomControl: false,
      preferCanvas: true,
    });

    L.control
      .zoom({ position: containerId.includes("baseline") ? "topleft" : "topright" })
      .addTo(this.map);

    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "© OpenStreetMap",
      maxZoom: 18,
    }).addTo(this.map);

    this.stopLayer = L.layerGroup().addTo(this.map);
    this.routeLayer = L.layerGroup().addTo(this.map);
    this.markerLayer = L.layerGroup().addTo(this.map);
    this.requestLayer = L.layerGroup().addTo(this.map);
    this.vehicleLayer = L.layerGroup().addTo(this.map);
    this._ready = true;
  }

  invalidate() {
    if (!this._ready) return;
    this.map.invalidateSize({ animate: false, pan: false });
  }

  setNetwork(stops, bbox) {
    if (!this._ready) return;
    this.stopLayer.clearLayers();

    (stops || []).forEach((s) => {
      const r = 3 + Math.min(8, (s.demand || 0) * 60);
      L.circleMarker([s.lat, s.lon], {
        radius: r,
        color: "#7f8c8d",
        fillColor: "#bdc3c7",
        fillOpacity: 0.45,
        weight: 1,
      })
        .bindTooltip(s.name || s.id)
        .addTo(this.stopLayer);
    });

    if (bbox) {
      if (this.bboxRect) this.map.removeLayer(this.bboxRect);
      this.bboxRect = L.rectangle(
        [
          [bbox.lat_min, bbox.lon_min],
          [bbox.lat_max, bbox.lon_max],
        ],
        { color: "#3498db", weight: 1, fill: false, dashArray: "4" }
      ).addTo(this.map);
    }

    if (stops && stops.length) {
      const bounds = L.latLngBounds(stops.map((s) => [s.lat, s.lon]));
      this.map.fitBounds(bounds.pad(0.08));
    }

    requestAnimationFrame(() => this.invalidate());
  }

  updateFrame(frame) {
    if (!this._ready || !frame) return;

    this.routeLayer.clearLayers();
    (frame.vehicles || []).forEach((v) => {
      if (v.route && v.route.length >= 2) {
        L.polyline(v.route, {
          color: this.color,
          weight: this.isBaseline ? 2 : 4,
          opacity: this.isBaseline ? 0.35 : 0.75,
          dashArray: this.isBaseline ? "8,6" : null,
        }).addTo(this.routeLayer);
      }
    });

    this.markerLayer.clearLayers();
    (frame.rejected_markers || []).forEach((m) => {
      L.circleMarker([m.lat, m.lon], {
        radius: 8,
        color: "#922b21",
        fillColor: "#e74c3c",
        fillOpacity: 0.9,
        weight: 2,
      })
        .bindTooltip("거절됨")
        .addTo(this.markerLayer);
    });

    this.requestLayer.clearLayers();
    (frame.recent_requests || []).forEach((r) => {
      if (!r.origin || !r.dest) return;
      const col =
        r.status === "served"
          ? "#27ae60"
          : r.status === "rejected"
          ? "#c0392b"
          : r.status === "assigned"
          ? "#f39c12"
          : "#95a5a6";
      L.polyline([r.origin, r.dest], {
        color: col,
        weight: r.status === "rejected" ? 3 : 2,
        opacity: 0.5,
        dashArray: "3,6",
      }).addTo(this.requestLayer);
    });

    const activeIds = new Set();
    (frame.vehicles || []).forEach((v) => {
      activeIds.add(v.id);
      const size = Math.min(28, 12 + (v.onboard || 0) * 3);
      const isBaseline = this.isBaseline;
      const icon = L.divIcon({
        className: "vehicle-icon",
        html: `<div class="vehicle-dot ${isBaseline ? "baseline" : "optimized"}" style="width:${size}px;height:${size}px"></div>`,
        iconSize: [size, size],
        iconAnchor: [size / 2, size / 2],
      });

      if (this.vehicleMarkers[v.id]) {
        this.vehicleMarkers[v.id].setLatLng([v.lat, v.lon]);
        this.vehicleMarkers[v.id].setIcon(icon);
      } else {
        this.vehicleMarkers[v.id] = L.marker([v.lat, v.lon], { icon, zIndexOffset: 500 })
          .bindTooltip(`${v.id} · 탑승 ${v.onboard}명`)
          .addTo(this.vehicleLayer);
      }
      const marker = this.vehicleMarkers[v.id];
      if (marker.getTooltip()) {
        marker.setTooltipContent(`${v.id} · 탑승 ${v.onboard}명`);
      }
    });

    Object.keys(this.vehicleMarkers).forEach((id) => {
      if (!activeIds.has(id)) {
        this.vehicleLayer.removeLayer(this.vehicleMarkers[id]);
        delete this.vehicleMarkers[id];
      }
    });
  }
}

let simData = null;
let frameIndex = 0;
let playing = false;
let playTimer = null;
let mapBaseline = null;
let mapOptimized = null;

const el = (id) => document.getElementById(id);

function initMaps() {
  mapBaseline = new SimMap("map-baseline", VEHICLE_COLORS.baseline, true);
  mapOptimized = new SimMap("map-optimized", VEHICLE_COLORS.optimized, false);

  const refreshMaps = () => {
    mapBaseline.invalidate();
    mapOptimized.invalidate();
  };

  requestAnimationFrame(refreshMaps);
  setTimeout(refreshMaps, 100);
  setTimeout(refreshMaps, 400);
  window.addEventListener("resize", refreshMaps);
}

function setLoading(show) {
  el("loading").classList.toggle("hidden", !show);
  if (!show) {
    requestAnimationFrame(() => {
      mapBaseline?.invalidate();
      mapOptimized?.invalidate();
    });
  }
}

function updateKpi(prefix, metrics) {
  if (!metrics) return;
  el(`${prefix}-wait`).textContent = (metrics.avg_wait_min ?? 0).toFixed(1);
  el(`${prefix}-success`).textContent = `${((metrics.success_rate ?? 0) * 100).toFixed(1)}%`;
  el(`${prefix}-accepted`).textContent = metrics.assigned ?? 0;
  el(`${prefix}-reject`).textContent = metrics.rejected ?? 0;

  if (prefix === "bl") {
    el("bl-reject").classList.toggle("kpi-bad", (metrics.rejected ?? 0) > 5);
  }
}

function updateLiveDiff(bl, op) {
  if (!bl?.metrics || !op?.metrics) return;
  el("live-diff").classList.remove("hidden");
  el("live-bl-success").textContent = `${((bl.metrics.success_rate ?? 0) * 100).toFixed(1)}%`;
  el("live-op-success").textContent = `${((op.metrics.success_rate ?? 0) * 100).toFixed(1)}%`;

  const acceptDiff = (op.metrics.assigned ?? 0) - (bl.metrics.assigned ?? 0);
  el("live-accept-diff").textContent =
    acceptDiff > 0
      ? `수락 +${acceptDiff}건`
      : acceptDiff < 0
      ? `수락 ${acceptDiff}건`
      : `수락 ${bl.metrics.assigned ?? 0}건 동일`;

  const waitDiff = (bl.metrics.avg_wait_min ?? 0) - (op.metrics.avg_wait_min ?? 0);
  el("live-wait-diff").textContent =
    waitDiff > 0
      ? `평균 대기 ${waitDiff.toFixed(1)}분 단축`
      : `평균 대기 ${Math.abs(waitDiff).toFixed(1)}분`;

  const rejDiff = (bl.metrics.rejected ?? 0) - (op.metrics.rejected ?? 0);
  el("live-reject-diff").textContent =
    rejDiff > 0 ? `거절 ${rejDiff}건 감소` : `거절 ${Math.abs(rejDiff)}건`;
}

function getFrame(frames, idx) {
  if (!frames || frames.length === 0) return null;
  const i = Math.max(0, Math.min(idx, frames.length - 1));
  return frames[i];
}

function showFrame(idx) {
  if (!simData) return;
  const blFrames = simData.baseline.frames;
  const opFrames = simData.optimized.frames;
  const maxIdx = Math.max(blFrames.length, opFrames.length, 1) - 1;
  frameIndex = Math.max(0, Math.min(idx, maxIdx));

  const bl = getFrame(blFrames, frameIndex);
  const op = getFrame(opFrames, frameIndex);

  mapBaseline.updateFrame(bl);
  mapOptimized.updateFrame(op);
  if (bl) updateKpi("bl", bl.metrics);
  if (op) updateKpi("op", op.metrics);
  updateLiveDiff(bl, op);
  el("time-label").textContent = bl?.time_label || op?.time_label || "—";
  el("timeline").value = frameIndex;
  refreshFooterSummary();
}

const PLAYING_SUMMARY =
  "재생 중… 시뮬레이션이 끝나면 최종 결과가 표시됩니다.";

function getMaxFrameIndex() {
  if (!simData) return 0;
  return Math.max(simData.baseline.frames.length, simData.optimized.frames.length, 1) - 1;
}

function setPlayingSummary() {
  el("compare-summary").textContent = PLAYING_SUMMARY;
}

function refreshFooterSummary() {
  if (!simData) return;
  if (frameIndex >= getMaxFrameIndex()) {
    updateFinalSummary();
  } else {
    setPlayingSummary();
  }
}

function updateFinalSummary() {
  if (!simData) return;
  const b = simData.baseline.final_metrics;
  const o = simData.optimized.final_metrics;
  const rph = simData.scenario?.requests_per_hour ?? simData.requests_per_hour ?? "?";

  const acceptDelta = o.assigned_requests - b.assigned_requests;
  const succPp = (o.dispatch_success_rate - b.dispatch_success_rate) * 100;
  const rejRed = b.rejected_requests - o.rejected_requests;
  const waitDelta = b.avg_wait_time_min - o.avg_wait_time_min;
  const waitPct =
    b.avg_wait_time_min > 0 ? (waitDelta / b.avg_wait_time_min) * 100 : 0;

  const waitLine =
    waitDelta > 0
      ? `대기 <span class="better">${b.avg_wait_time_min.toFixed(1)}→${o.avg_wait_time_min.toFixed(1)}분 (${waitPct.toFixed(1)}%↓)</span>`
      : `대기 ${b.avg_wait_time_min.toFixed(1)}→${o.avg_wait_time_min.toFixed(1)}분`;

  el("compare-summary").innerHTML =
    `<strong>최종 (seed=${simData.seed}, ${rph} req/h)</strong> — ` +
    `수락 <span class="better">${b.assigned_requests}→${o.assigned_requests} (+${acceptDelta}건)</span>, ` +
    `${waitLine}, ` +
    `성공률 <span class="better">${(b.dispatch_success_rate * 100).toFixed(1)}→${(o.dispatch_success_rate * 100).toFixed(1)}% (+${succPp.toFixed(1)}%p)</span>, ` +
    `거절 <span class="better">${b.rejected_requests}→${o.rejected_requests} (${rejRed}건 감소)</span>`;
}

async function runSimulation() {
  const seed = parseInt(el("seed").value, 10) || 42;
  setLoading(true);
  stopPlay();
  el("btn-run").disabled = true;

  try {
    const res = await fetch(`/api/compare?seed=${seed}`);
    if (!res.ok) throw new Error(await res.text());
    simData = await res.json();

    if (!simData.baseline?.frames?.length || !simData.optimized?.frames?.length) {
      throw new Error("시뮬레이션 프레임 데이터가 비어 있습니다.");
    }

    mapBaseline.setNetwork(simData.stops, simData.bbox);
    mapOptimized.setNetwork(simData.stops, simData.bbox);

    const maxFrames = Math.max(
      simData.baseline.frames.length,
      simData.optimized.frames.length
    );
    el("timeline").max = Math.max(0, maxFrames - 1);
    el("timeline").disabled = false;
    el("btn-play").disabled = false;
    el("btn-pause").disabled = false;

    frameIndex = 0;
    setPlayingSummary();
    showFrame(0);
    startPlay();
  } catch (err) {
    console.error(err);
    alert("시뮬레이션 실패: " + err.message);
  } finally {
    setLoading(false);
    el("btn-run").disabled = false;
  }
}

function startPlay() {
  if (!simData || playing) return;
  playing = true;
  const interval = parseInt(el("speed").value, 10);
  const maxFrames = Math.max(
    simData.baseline.frames.length,
    simData.optimized.frames.length
  );

  playTimer = setInterval(() => {
    if (frameIndex >= maxFrames - 1) {
      stopPlay();
      return;
    }
    showFrame(frameIndex + 1);
  }, interval);
}

function stopPlay() {
  playing = false;
  if (playTimer) {
    clearInterval(playTimer);
    playTimer = null;
  }
}

function bindEvents() {
  el("btn-run").addEventListener("click", runSimulation);
  el("btn-play").addEventListener("click", startPlay);
  el("btn-pause").addEventListener("click", stopPlay);
  el("timeline").addEventListener("input", (e) => {
    stopPlay();
    showFrame(parseInt(e.target.value, 10));
  });
  el("speed").addEventListener("change", () => {
    if (playing) {
      stopPlay();
      startPlay();
    }
  });
}

async function loadNetwork() {
  try {
    const res = await fetch("/api/network");
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    mapBaseline.setNetwork(data.stops, data.bbox);
    mapOptimized.setNetwork(data.stops, data.bbox);
  } catch (err) {
    console.error("Network load failed:", err);
  }
}

function boot() {
  initMaps();
  bindEvents();
  loadNetwork();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
