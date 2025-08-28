import React, { useEffect, useRef, useState } from "react";

/* ---------- small atoms ---------- */

function Dot({ ok, warn, size = 10, title = "" }) {
  const color = ok ? "#10b981" : warn ? "#f59e0b" : "#ef4444";
  return (
    <span
      title={title}
      style={{
        display: "inline-block",
        width: size,
        height: size,
        borderRadius: "999px",
        background: color,
        marginRight: 6,
        boxShadow: `0 0 10px ${color}`,
      }}
    />
  );
}

function Toggle({ label, checked, onChange }) {
  return (
    <label
      style={{
        display: "inline-flex",
        gap: 8,
        alignItems: "center",
        marginRight: 12,
        padding: "6px 10px",
        borderRadius: 8,
        background: checked ? "#064e3b" : "#111827",
        border: `1px solid ${checked ? "#10b98155" : "#374151"}`,
        color: checked ? "#10b981" : "#e5e7eb",
        cursor: "pointer",
        userSelect: "none",
      }}
    >
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        style={{ accentColor: "#10b981" }}
      />
      <span>{label}</span>
    </label>
  );
}

function Slider({ label, min, max, step = 1, value, onChange, suffix = "" }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 8 }}>
      <label style={{ color: "#9ca3af" }}>{label}</label>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          style={{ width: 220 }}
        />
        <span
          style={{
            minWidth: 54,
            textAlign: "center",
            background: "rgba(16,185,129,0.12)",
            border: "1px solid rgba(16,185,129,0.35)",
            color: "#10b981",
            padding: "2px 8px",
            borderRadius: 6,
            fontVariantNumeric: "tabular-nums",
          }}
        >
          {value}
          {suffix}
        </span>
      </div>
    </div>
  );
}

/* ---------- dropdowns & panels ---------- */

function Dropdown({ open, anchorRef, onClose, children, width = 320 }) {
  const ref = useRef(null);
  useEffect(() => {
    function onDocClick(e) {
      if (!open) return;
      if (ref.current && !ref.current.contains(e.target) && !anchorRef?.current?.contains(e.target)) {
        onClose?.();
      }
    }
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [open, onClose, anchorRef]);

  if (!open) return null;
  const rect = anchorRef?.current?.getBoundingClientRect?.() ?? { left: 20, bottom: 50 };
  return (
    <div
      ref={ref}
      style={{
        position: "fixed",
        top: rect.bottom + 6,
        left: rect.left,
        width,
        background: "#0B0F1A",
        border: "1px solid rgba(255,255,255,0.08)",
        boxShadow: "0 15px 40px rgba(0,0,0,0.5)",
        borderRadius: 12,
        padding: 12,
        zIndex: 50,
      }}
    >
      {children}
    </div>
  );
}

function Collapsible({ title, defaultOpen = true, children, badge = null }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div
      style={{
        background: "rgba(255,255,255,0.04)",
        border: "1px solid rgba(255,255,255,0.08)",
        borderRadius: 14,
        overflow: "hidden",
      }}
    >
      <div
        onClick={() => setOpen((v) => !v)}
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "12px 14px",
          cursor: "pointer",
          color: "#10b981",
          fontWeight: 600,
        }}
      >
        <span>{title}</span>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {badge}
          <span
            style={{
              display: "inline-block",
              width: 24,
              height: 24,
              borderRadius: 999,
              background: "rgba(16,185,129,0.12)",
              border: "1px solid rgba(16,185,129,0.35)",
              color: "#10b981",
              lineHeight: "24px",
              textAlign: "center",
              fontWeight: 800,
            }}
          >
            {open ? "–" : "+"}
          </span>
        </div>
      </div>
      {open && <div style={{ padding: 12 }}>{children}</div>}
    </div>
  );
}

/* ---------- main app ---------- */

export default function App() {
  /* top process & mode state (fake for UI-only) */
  const [connected, setConnected] = useState(false);
  const [running, setRunning] = useState(false);
  const [mode, setMode] = useState("SAR"); // "SAR" | "Suspect-Lock"
  const [faceBlur, setFaceBlur] = useState(true);

  /* fake stats (UI-only) */
  const [fps, setFps] = useState(0);
  const [gpu, setGpu] = useState(64);
  const [lat, setLat] = useState(14.8);
  const [geoErr, setGeoErr] = useState(7.5);
  const [latency, setLatency] = useState(467);

  /* system lights (UI-only) */
  const [adbOK, setAdbOK] = useState(false);
  const [scrcpyOK, setScrcpyOK] = useState(false);
  const [ffProbeOK, setFfProbeOK] = useState(false);
  const [ffIngestOK, setFfIngestOK] = useState(false);

  /* dropdown anchors */
  const modeBtnRef = useRef(null);
  const settingsBtnRef = useRef(null);
  const [modeOpen, setModeOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);

  /* settings (non-wired, just UI) */
  const [simThreshold, setSimThreshold] = useState(0.78);
  const [persistFrames, setPersistFrames] = useState(12);
  const [autoRevert, setAutoRevert] = useState(60); // minutes
  const [bitrate, setBitrate] = useState(16); // Mbps
  const [capFps, setCapFps] = useState(60);
  const [ocrTelemetry, setOcrTelemetry] = useState(true);
  const [manualTelemetry, setManualTelemetry] = useState(false);
  const [strongBlur, setStrongBlur] = useState(false);

  /* right panels mock */
  const [logs, setLogs] = useState([
    "UI loaded",
    "Tip: Click Connect, then Start to simulate run",
  ]);

  useEffect(() => {
    let id;
    if (running) {
      id = setInterval(() => {
        // jitter the fake stats a bit
        setFps((f) => Math.max(10, Math.min(60, Math.round(f + (Math.random() - 0.5) * 6))));
        setLatency((l) => Math.max(90, Math.min(900, Math.round(l + (Math.random() - 0.5) * 30))));
        setGpu((g) => Math.max(20, Math.min(99, Math.round(g + (Math.random() - 0.5) * 4))));
      }, 700);
    } else {
      setFps(0);
      setLatency(0);
    }
    return () => clearInterval(id);
  }, [running]);

  /* menu actions (UI-only) */
  function doConnect() {
    setConnected(true);
    setAdbOK(true);
    setScrcpyOK(true);
    setLogs((l) => [...l, "Connected (ADB/scrcpy mocked OK)"]);
  }
  function doStart() {
    if (!connected) return;
    setRunning(true);
    setFfProbeOK(true);
    setFfIngestOK(true);
    setFps(42);
    setLatency(180);
    setLogs((l) => [...l, "Pipeline started (FFmpeg ingest mocked OK)"]);
  }
  function doStop() {
    setRunning(false);
    setFfIngestOK(false);
    setLogs((l) => [...l, "Pipeline stopped"]);
  }

  function chooseMode(next) {
    setMode(next);
    setModeOpen(false);
    if (next === "SAR") setFaceBlur(true);
    setLogs((l) => [...l, `Mode set to ${next}`]);
  }

  /* basic theme */
  const themeBg = "linear-gradient(135deg, #0b0f1a 0%, #111827 100%)";
  const panelBg = "rgba(255,255,255,0.04)";
  const border = "1px solid rgba(255,255,255,0.08)";

  return (
    <div
      style={{
        minHeight: "100vh",
        background: themeBg,
        color: "#E5E7EB",
        fontFamily: "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial",
      }}
    >
      {/* Top bar */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 24,
          padding: "14px 18px",
          borderBottom: border,
          position: "sticky",
          top: 0,
          zIndex: 20,
          backdropFilter: "blur(6px)",
          background: "rgba(0,0,0,0.3)",
        }}
      >
        <div style={{ fontWeight: 800, color: "#10b981" }}>Foresight 1.0</div>

        <button
          onClick={doConnect}
          style={menuBtnStyle}
          title="Connect ADB + launch scrcpy (mock)"
        >
          Connect
        </button>

        <button
          onClick={doStart}
          style={{ ...menuBtnStyle, opacity: connected ? 1 : 0.5, cursor: connected ? "pointer" : "not-allowed" }}
          title="Start capture + inference (mock)"
        >
          Start
        </button>

        <button onClick={doStop} style={menuBtnStyle} title="Stop pipeline (mock)">
          Stop
        </button>

        {/* Mode dropdown */}
        <button
          ref={modeBtnRef}
          onClick={() => setModeOpen((v) => !v)}
          style={menuBtnStyle}
          title="Switch SAR / Suspect-Lock"
        >
          Mode ▾
        </button>

        <Dropdown open={modeOpen} anchorRef={modeBtnRef} onClose={() => setModeOpen(false)} width={280}>
          <div style={{ padding: 6 }}>
            <div style={{ fontWeight: 700, marginBottom: 8, color: "#10b981" }}>Modes</div>
            <button
              onClick={() => chooseMode("SAR")}
              style={rowBtnStyle(mode === "SAR")}
            >
              <span>SAR (Search &amp; Rescue)</span>
              {mode === "SAR" && <Dot ok size={10} />}
            </button>
            <button
              onClick={() => chooseMode("Suspect-Lock")}
              style={rowBtnStyle(mode === "Suspect-Lock")}
            >
              <span>Suspect-Lock</span>
              {mode === "Suspect-Lock" && <Dot ok size={10} />}
            </button>
            <div style={{ marginTop: 10, borderTop: border, paddingTop: 10 }}>
              <Toggle label="Face blur (bystanders)" checked={faceBlur} onChange={setFaceBlur} />
            </div>
          </div>
        </Dropdown>

        {/* Settings dropdown */}
        <button
          ref={settingsBtnRef}
          onClick={() => setSettingsOpen((v) => !v)}
          style={menuBtnStyle}
          title="Open settings"
        >
          Settings ▾
        </button>

        <Dropdown open={settingsOpen} anchorRef={settingsBtnRef} onClose={() => setSettingsOpen(false)} width={420}>
          <div style={{ display: "grid", gap: 14 }}>
            <div style={{ fontWeight: 700, color: "#8b5cf6" }}>Privacy & Modes</div>
            <Toggle label="Face blur (bystanders)" checked={faceBlur} onChange={setFaceBlur} />
            <Toggle label="Strong blur" checked={strongBlur} onChange={setStrongBlur} />
            <Slider label="Suspect-Lock similarity" min={0.5} max={0.95} step={0.01} value={simThreshold} onChange={setSimThreshold} />
            <Slider label="Persistence (frames)" min={4} max={24} value={persistFrames} onChange={setPersistFrames} />
            <Slider label="Auto-revert (minutes)" min={5} max={180} step={5} value={autoRevert} onChange={setAutoRevert} />

            <div style={{ height: 1, background: "rgba(255,255,255,0.08)", margin: "6px 0" }} />

            <div style={{ fontWeight: 700, color: "#3b82f6" }}>Capture & Telemetry</div>
            <Slider label="Capture FPS" min={24} max={90} value={capFps} onChange={setCapFps} />
            <Slider label="Video bitrate (Mbps)" min={4} max={40} value={bitrate} onChange={setBitrate} suffix="M" />
            <Toggle label="OCR Telemetry (read OSD)" checked={ocrTelemetry} onChange={setOcrTelemetry} />
            <Toggle label="Manual Telemetry (fallback)" checked={manualTelemetry} onChange={setManualTelemetry} />

            <div style={{ height: 1, background: "rgba(255,255,255,0.08)", margin: "6px 0" }} />

            <div style={{ fontSize: 12, color: "#9ca3af" }}>
              (All settings are UI-only in this base build. Wire them to your backend later.)
            </div>
          </div>
        </Dropdown>

        {/* inline stats like the screenshot */}
        <div style={{ marginLeft: "auto", display: "flex", gap: 20, alignItems: "center", fontWeight: 600 }}>
          <span>
            FPS:{" "}
            <span style={{ color: running && fps < 18 ? "#ef4444" : "#10b981" }}>
              {running ? fps : "0.0"}
            </span>
          </span>
          <span>
            GPU: <span style={{ color: "#10b981" }}>{gpu}%</span>
          </span>
          <span>
            Lat: <span style={{ color: "#10b981" }}>{lat.toFixed(1)}</span>
          </span>
          <span>
            Err: <span style={{ color: "#10b981" }}>±{geoErr}m</span>
          </span>
        </div>
      </div>

      {/* Main content grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 360px", gap: 18, padding: 18 }}>
        {/* Left: video canvas area */}
        <div
          style={{
            background: "#000",
            borderRadius: 16,
            border,
            minHeight: 420,
            position: "relative",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            overflow: "hidden",
          }}
        >
          {!running ? (
            <div style={{ textAlign: "center", color: "#10b981" }}>
              <div
                style={{
                  width: 120,
                  height: 120,
                  borderRadius: 999,
                  background: "rgba(16,185,129,0.12)",
                  border: "2px solid rgba(16,185,129,0.4)",
                  margin: "0 auto 18px auto",
                  display: "grid",
                  placeItems: "center",
                }}
              >
                <svg width="60" height="60" viewBox="0 0 24 24" fill="#10b981">
                  <path d="M8 5v14l11-7z" />
                </svg>
              </div>
              <div style={{ fontSize: 28, letterSpacing: 2 }}>NO SIGNAL…</div>
            </div>
          ) : (
            <div style={{ width: "100%", height: 420, position: "relative" }}>
              {/* This is just a placeholder gradient “feed”. Replace with your <img src=…> later. */}
              <div
                style={{
                  position: "absolute",
                  inset: 0,
                  background:
                    "repeating-linear-gradient(135deg, #0b0f1a, #0b0f1a 20px, #10172a 20px, #10172a 40px)",
                  opacity: 0.9,
                }}
              />
              <div
                style={{
                  position: "absolute",
                  top: 12,
                  left: 12,
                  background: "rgba(0,0,0,0.5)",
                  border: "1px solid rgba(255,255,255,0.1)",
                  borderRadius: 8,
                  padding: "6px 10px",
                  fontSize: 12,
                }}
              >
                {mode} • {fps} fps • Latency {latency} ms • Face-blur {faceBlur ? "ON" : "OFF"}
              </div>
            </div>
          )}
        </div>

        {/* Right: panels */}
        <div style={{ display: "grid", gap: 12 }}>
          <Collapsible title="Detections" badge={<span style={{ color: "#9ca3af" }}>+</span>} defaultOpen>
            <div style={{ display: "grid", gap: 10 }}>
              {[1, 2].map((i) => (
                <div
                  key={i}
                  style={{
                    display: "grid",
                    gridTemplateColumns: "64px 1fr",
                    gap: 10,
                    alignItems: "center",
                    background: panelBg,
                    border,
                    borderRadius: 10,
                    padding: 8,
                  }}
                >
                  <div
                    style={{
                      width: 64,
                      height: 48,
                      background:
                        "linear-gradient(135deg, rgba(16,185,129,0.2), rgba(59,130,246,0.2))",
                      borderRadius: 6,
                    }}
                  />
                  <div>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <strong>person</strong>
                      <span style={{ color: "#9ca3af" }}>0.{8 + i}</span>
                      <span style={{ marginLeft: "auto", color: "#10b981" }}>
                        {mode === "Suspect-Lock" ? "LOCK CANDIDATE" : "SAR"}
                      </span>
                    </div>
                    <div style={{ fontSize: 12, color: "#9ca3af" }}>
                      Lat 14.5891, Lon 121.0173 • ±7.6 m • {new Date().toLocaleTimeString()}
                    </div>
                    <div style={{ marginTop: 6, display: "flex", gap: 8 }}>
                      <MiniBtn>Confirm</MiniBtn>
                      <MiniBtn>Reject</MiniBtn>
                      <MiniBtn>Handoff</MiniBtn>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Collapsible>

          <Collapsible title="Suspect" badge={<span style={{ color: "#9ca3af" }}>+</span>} defaultOpen={false}>
            <div style={{ display: "grid", gap: 10 }}>
              <div
                style={{
                  border: "1px dashed rgba(255,255,255,0.15)",
                  padding: 12,
                  borderRadius: 10,
                  textAlign: "center",
                  color: "#9ca3af",
                }}
              >
                Drop reference image(s) here
              </div>
              <Slider
                label="Similarity threshold"
                min={0.5}
                max={0.95}
                step={0.01}
                value={simThreshold}
                onChange={setSimThreshold}
              />
              <Slider
                label="Persistence frames"
                min={4}
                max={24}
                value={persistFrames}
                onChange={setPersistFrames}
              />
            </div>
          </Collapsible>

          <Collapsible title="Maps" badge={<span style={{ color: "#9ca3af" }}>–</span>} defaultOpen>
            <div
              style={{
                height: 160,
                borderRadius: 10,
                overflow: "hidden",
                border,
                background:
                  "linear-gradient(180deg, #0b0f1a, #111827), url('data:image/svg+xml;utf8,\
<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22320%22 height=%22160%22><rect width=%22320%22 height=%22160%22 fill=%22%23111827%22/><g stroke=%22%23e5e7eb%22 stroke-width=%220.5%22 opacity=%220.2%22><path d=%22M0 20 H320 M0 60 H320 M0 100 H320 M0 140 H320%22/><path d=%22M40 0 V160 M120 0 V160 M200 0 V160 M280 0 V160%22/></g><circle cx=%2280%22 cy=%2260%22 r=%228%22 fill=%22#ef4444%22/><circle cx=%22220%22 cy=%2285%22 r=%225%22 fill=%22#10b981%22/></svg>') center/cover no-repeat",
              }}
            />
          </Collapsible>

          <Collapsible title="Logs" badge={<span style={{ color: "#9ca3af" }}>+</span>} defaultOpen={false}>
            <div
              style={{
                maxHeight: 140,
                overflow: "auto",
                fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
                fontSize: 12,
                background: "rgba(0,0,0,0.4)",
                borderRadius: 8,
                padding: 8,
                border,
              }}
            >
              {logs.map((l, i) => (
                <div key={i} style={{ padding: "2px 0", color: "#9ca3af" }}>
                  {l}
                </div>
              ))}
            </div>
          </Collapsible>
        </div>
      </div>

      {/* Bottom status bar */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr 1fr 1fr",
          gap: 18,
          padding: "12px 18px 22px 18px",
        }}
      >
        <div>
          <strong>Status:</strong>{" "}
          <span style={{ color: running ? "#10b981" : "#ef4444" }}>
            {running ? "running" : "error"}
          </span>
        </div>

        <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          <span title="ADB">
            <Dot ok={adbOK} /> ADB
          </span>
          <span title="scrcpy">
            <Dot ok={scrcpyOK} /> scrcpy
          </span>
          <span title="FFmpeg probe">
            <Dot ok={ffProbeOK} /> FFmpeg:
          </span>
          <span title="FFmpeg ingest">
            <Dot ok={ffIngestOK} /> FFmpeg:
          </span>
        </div>

        <div>
          Latency:{" "}
          <span style={{ color: running ? (latency > 350 ? "#ef4444" : "#10b981") : "#6b7280" }}>
            {running ? `${latency}ms` : "—"}
          </span>{" "}
          &nbsp; | &nbsp; Disk: 73%
        </div>

        <div>
          Model: <span style={{ color: "#10b981" }}>yolo_sar_n.onnx</span> &nbsp; | &nbsp; Telemetry:{" "}
          <span style={{ color: "#10b981" }}>{ocrTelemetry ? "OCR (conf 0.84)" : manualTelemetry ? "Manual" : "N/A"}</span>
        </div>
      </div>
    </div>
  );
}

/* ---------- tiny helpers ---------- */

const menuBtnStyle = {
  padding: "6px 10px",
  borderRadius: 8,
  border: "1px solid rgba(255,255,255,0.08)",
  background: "rgba(255,255,255,0.06)",
  color: "#E5E7EB",
  cursor: "pointer",
  transition: "all .12s ease",
};

function MiniBtn({ children }) {
  return (
    <button
      style={{
        padding: "4px 8px",
        borderRadius: 6,
        background: "rgba(255,255,255,0.06)",
        border: "1px solid rgba(255,255,255,0.08)",
        color: "#E5E7EB",
        cursor: "pointer",
      }}
      onClick={() => {}}
    >
      {children}
    </button>
  );
}

function rowBtnStyle(active) {
  return {
    width: "100%",
    textAlign: "left",
    padding: "10px 10px",
    color: active ? "#10b981" : "#E5E7EB",
    background: active ? "rgba(16,185,129,0.08)" : "transparent",
    border: "1px solid rgba(255,255,255,0.06)",
    borderRadius: 8,
    marginBottom: 8,
    cursor: "pointer",
  };
}
