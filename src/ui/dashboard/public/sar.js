(() => {
  const logEl = document.getElementById("log");
  const canvas = document.getElementById("videoCanvas");
  const ctx = canvas.getContext("2d");
  let ws = null, pullTimer = null;

  function log(msg) {
    const t = new Date().toLocaleTimeString();
    if (logEl) { logEl.innerHTML += `[${t}] ${msg}<br>`; logEl.scrollTop = logEl.scrollHeight; }
    else console.log(msg);
  }

  // --- UI: add SAR controls non-destructively
  function injectControls() {
    const bar = document.createElement("div");
    bar.style.margin = "10px 0";
    bar.innerHTML = `
      <button id="sarModeBtn">SAR Mode</button>
      <button id="susModeBtn">Suspect-Lock</button>
      <label style="margin-left:10px;">
        <input type="checkbox" id="blurChk" checked> Blur bystanders
      </label>
    `;
    // insert near the top of body
    document.body.insertBefore(bar, document.body.firstChild.nextSibling);

    document.getElementById("sarModeBtn").onclick = () => setMode("sar");
    document.getElementById("susModeBtn").onclick = () => setMode("suspect");
    document.getElementById("blurChk").onchange = (e) => setBlur(e.target.checked);
  }

  async function setMode(mode) {
    await fetch("/api/mode", { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify({mode})});
    log(`Mode -> ${mode.toUpperCase()}`);
  }
  async function setBlur(enabled) {
    await fetch("/api/blur", { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify({enabled})});
    log(`Face blur -> ${enabled ? "ON" : "OFF"}`);
  }

  function startFramePull() {
    if (!canvas) return;
    const img = new Image();
    img.onload = () => {
      // letterbox fit
      const ratio = Math.min(canvas.width / img.width, canvas.height / img.height);
      const dw = img.width * ratio, dh = img.height * ratio;
      const dx = (canvas.width - dw) / 2, dy = (canvas.height - dh) / 2;
      ctx.fillStyle = "#222"; ctx.fillRect(0,0,canvas.width,canvas.height);
      ctx.drawImage(img, dx, dy, dw, dh);
    };
    pullTimer = setInterval(() => { img.src = `/frame.jpg?ts=${Date.now()}`; }, 120);
  }

  // override Start/Stop *if they exist*
  const _start = window.handleStart;
  window.handleStart = async function() {
    try { await fetch("/api/pipeline/start", {method: "POST"}); } catch {}
    ws = new WebSocket(`ws://${location.host}/ws/sar`);
    ws.onopen = () => log("WS connected (/ws/sar)");
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "stats") {
        log(`fps=${msg.fps} lat=${msg.latency}ms det=${(msg.detections||[]).length} mode=${msg.mode} blur=${msg.blur}`);
      }
    };
    ws.onclose = () => log("WS closed");
    startFramePull();
    if (typeof _start === "function") _start();
  };

  const _stop = window.handleStop;
  window.handleStop = async function() {
    try { await fetch("/api/pipeline/stop", {method: "POST"}); } catch {}
    if (pullTimer) { clearInterval(pullTimer); pullTimer = null; }
    try { ws && ws.close(); } catch {}
    if (typeof _stop === "function") _stop();
  };

  // init
  injectControls();
  setMode("sar");
  setBlur(true);
})();
