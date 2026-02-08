const statusEl = document.getElementById("status");
const recommendationsEl = document.getElementById("recommendations");
const opportunitiesEl = document.getElementById("opportunities");
const integrationsEl = document.getElementById("integrations");

const intervalInput = document.getElementById("intervalInput");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const scanBtn = document.getElementById("scanBtn");

function setStatus(text, kind = "") {
  statusEl.textContent = text;
  statusEl.className = `status ${kind}`.trim();
}

function fmtTs(ts) {
  if (!ts) return "";
  return new Date(ts * 1000).toLocaleString();
}

function card(item) {
  const legDetails = (item.leg_details || []).length
    ? `
      <div class="legs-block">
        <div class="legs-title">Parlay Legs</div>
        <ul class="legs-list">
          ${(item.leg_details || []).map((leg) => `<li>${leg.display_leg} <span class="leg-stat">(${leg.stat})</span></li>`).join("")}
        </ul>
      </div>
    `
    : "";

  return `
    <article class="card">
      <div class="row">
        <strong>${item.sport}</strong>
        <span class="pill">${item.market_type}</span>
        <span class="pill book">${item.sportsbook}</span>
      </div>
      <div class="event">${item.event}</div>
      <div class="selection">${item.selection}</div>
      <div class="metrics">
        <span>Odds: ${item.odds_american > 0 ? "+" : ""}${item.odds_american}</span>
        <span>Edge: ${(item.edge * 100).toFixed(2)}%</span>
        <span>Confidence: ${(item.confidence * 100).toFixed(1)}%</span>
      </div>
      ${legDetails}
      <div class="note">${item.rationale}</div>
      <div class="actions">
        <button onclick="queuePick('${item.id}')">Queue</button>
        <button class="secondary" onclick="placePick('${item.id}','${item.sportsbook}')">Execute</button>
      </div>
    </article>
  `;
}

function renderIntegrations(payload) {
  integrationsEl.innerHTML = payload.integrations.map((x) => `
    <div class="integration-card">
      <div class="row"><strong>${x.sportsbook}</strong><span>as of ${payload.as_of}</span></div>
      <div>Market data: ${x.market_data}</div>
      <div>Order placement: ${x.order_placement}</div>
      <div class="note">${x.notes}</div>
      <a href="${x.docs}" target="_blank" rel="noreferrer">Docs/Terms</a>
    </div>
  `).join("");
}

async function refresh() {
  const [statusRes, recRes, oppRes, intRes] = await Promise.all([
    fetch("/api/status"),
    fetch("/api/recommendations?limit=25"),
    fetch("/api/opportunities?limit=100"),
    fetch("/api/integration-matrix"),
  ]);

  const statusData = await statusRes.json();
  const recData = await recRes.json();
  const oppData = await oppRes.json();
  const intData = await intRes.json();

  const runState = statusData.running ? "running" : "stopped";
  setStatus(`Swarm ${runState}. Opportunities: ${statusData.opportunity_count}. Last scan: ${fmtTs(statusData.last_scan_at)}`);

  recommendationsEl.innerHTML = recData.items.length ? recData.items.map(card).join("") : "<div class='empty'>No recommendations yet.</div>";
  opportunitiesEl.innerHTML = oppData.items.length ? oppData.items.map(card).join("") : "<div class='empty'>No opportunities yet.</div>";
  renderIntegrations(intData);
}

async function startSwarm() {
  const interval = Number(intervalInput.value) || 20;
  const res = await fetch("/api/swarm/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ interval_seconds: interval }),
  });
  if (!res.ok) {
    setStatus("Failed to start swarm", "error");
    return;
  }
  setStatus("Swarm started", "ok");
  await refresh();
}

async function stopSwarm() {
  const res = await fetch("/api/swarm/stop", { method: "POST" });
  if (!res.ok) {
    setStatus("Failed to stop swarm", "error");
    return;
  }
  setStatus("Swarm stopped", "ok");
  await refresh();
}

async function scanOnce() {
  const res = await fetch("/api/swarm/scan", { method: "POST" });
  if (!res.ok) {
    setStatus("Manual scan failed", "error");
    return;
  }
  setStatus("Manual scan completed", "ok");
  await refresh();
}

async function queuePick(id) {
  const res = await fetch("/api/execute", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id, sportsbook: "fanduel", action: "queue" }),
  });
  const data = await res.json();
  if (!res.ok) {
    setStatus(data.error || "Queue failed", "error");
    return;
  }
  setStatus("Opportunity queued", "ok");
}

async function placePick(id, sportsbook) {
  const res = await fetch("/api/execute", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id, sportsbook, action: "place" }),
  });
  const data = await res.json();

  if (!res.ok) {
    setStatus(data.error || "Execution failed", "error");
    return;
  }

  if (data.status === "manual_required" && data.deeplink) {
    setStatus("Manual placement required for this sportsbook. Opening site.", "ok");
    window.open(data.deeplink, "_blank");
    return;
  }

  if (data.status === "not_configured") {
    setStatus(data.reason, "error");
    return;
  }

  setStatus(`Execution response: ${data.status || "ok"}`, "ok");
}

startBtn.addEventListener("click", startSwarm);
stopBtn.addEventListener("click", stopSwarm);
scanBtn.addEventListener("click", scanOnce);

refresh();
setInterval(refresh, 10000);

window.queuePick = queuePick;
window.placePick = placePick;
