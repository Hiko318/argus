import React, { useEffect, useRef, useState } from "react";

/* ---------- atoms ---------- */
function Dot({ ok, warn, size = 10 }) {
  const color = ok ? "#10b981" : warn ? "#f59e0b" : "#ef4444";
  return (
    <span
      style={{
        display: "inline-block",
        width: size,
        height: size,
        borderRadius: "999px",
        background: color,
        marginRight: 6,
      }}
    />
  );
}

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
        fontSize: 12,
      }}
    >
      {children}
    </button>
  );
}

function Toggle({ label, checked, onChange }) {
  return (
    <label
      style={{
        display: "inline-flex",
        gap: 8,
        alignItems: "center",
        marginTop: 10,
        padding: "6px 10px",
        borderRadius: 8,
        background: checked ? "rgba(16,185,129,0.12)" : "#0B1220",
        border: `1px solid ${checked ? "rgba(16,185,129,0.35)" : "rgba(255,255,255,0.08)"}`,
        color: checked ? "#10b981" : "#e5e7eb",
        userSelect: "none",
        cursor: "pointer",
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

function Collapsible({ title, children, defaultOpen = true }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div
      style={{
        background: "rgba(255,255,255,0.04)",
        border: "1px solid rgba(255,255,255,0.08)",
        borderRadius: 10,
        overflow: "hidden",
      }}
    >
      <div
        onClick={() => setOpen((v) => !v)}
        style={{
          display: "flex",
          justifyContent: "space-between",
          padding: "10px 12px",
          cursor: "pointer",
          color: "#10b981",
          fontWeight: 600,
        }}
      >
        <span>{title}</span>
        <span>{open ? "–" : "+"}</span>
      </div>
      {open && <div style={{ padding: 10 }}>{children}</div>}
    </div>
  );
}

/* ---------- dropdown ---------- */
function Dropdown({ open, anchorRef, onClose, width = 320, children }) {
  const ref = useRef(null);

  useEffect(() => {
    function onDocClick(e) {
      if (!open) return;
      if (ref.current && !ref.current.contains(e.target) && !anchorRef?.current?.contains(e.target)) {
        onClose?.();
      }
    }
    function onEsc(e) {
      if (e.key === "Escape") onClose?.();
    }
    document.addEventListener("mousedown", onDocClick);
    document.addEventListener("keydown", onEsc);
    return () => {
      document.removeEventListener("mousedown", onDocClick);
      document.removeEventListener("keydown", onEsc);
    };
  }, [open, onClose, anchorRef]);

  if (!open) return null;
  const rect = anchorRef?.current?.getBoundingClientRect?.() ?? { left: 24, bottom: 56 };
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

/* ---------- main app ---------- */
export default function App() {
  const [connected, setConnected] = useState(false);
  const [running, setRunning] = useState(false);
  const [fps, setFps] = useState(0);
  const [gpu] = useState(64);
  const [lat] = useState(14.8);
  const [geoErr] = useState(7.5);
  const [latency, setLatency] = useState(467);
  const [logs, setLogs] = useState(["UI loaded", "Click Connect then Start"]);

  // mode + face blur
  const [mode, setMode] = useState("SAR"); // "SAR" | "Suspect-Lock"
  const [faceBlur, setFaceBlur] = useState(true);

  // dropdown
  const modeBtnRef = useRef(null);
  const [modeOpen, setModeOpen] = useState(false);

  useEffect(() => {
    let id;
    if (running) {
      id = setInterval(() => {
        setFps((f) => Math.max(10, Math.min(60, Math.round(f + (Math.random() - 0.5) * 4))));
        setLatency((l) => Math.max(90, Math.min(900, Math.round(l + (Math.random() - 0.5) * 20))));
      }, 800);
    } else {
      setFps(0);
      setLatency(0);
    }
    return () => clearInterval(id);
  }, [running]);

  function doConnect() {
    setConnected(true);
    setLogs((l) => [...l, "Connected (mock)"]);
  }
  function doStart() {
    if (!connected) return;
    setRunning(true);
    setFps(42);
    setLatency(180);
    setLogs((l) => [...l, "Pipeline started (mock)"]);
  }
  function doStop() {
    setRunning(false);
    setLogs((l) => [...l, "Pipeline stopped"]);
  }
  function chooseMode(next) {
    setMode(next);
    setModeOpen(false);
    if (next === "SAR") setFaceBlur(true);
    setLogs((l) => [...l, `Mode set to ${next}`]);
  }
  const modeItemStyle = (active) => ({
    width: "100%",
    textAlign: "left",
    padding: "10px 10px",
    color: active ? "#10b981" : "#E5E7EB",
    background: active ? "rgba(16,185,129,0.08)" : "transparent",
    border: "1px solid rgba(255,255,255,0.06)",
    borderRadius: 8,
    marginBottom: 8,
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
  });

  return (
    <div
      style={{
        width: "100vw",
        height: "100vh",
        display: "flex",
        flexDirection: "column",
        background: "linear-gradient(135deg, #0b0f1a 0%, #111827 100%)",
        color: "#E5E7EB",
        fontFamily: "ui-sans-serif, system-ui, Segoe UI, Roboto",
      }}
    >
      {/* Top bar */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          padding: "12px 18px",
          borderBottom: "1px solid rgba(255,255,255,0.08)",
        }}
      >
        <div style={{ fontWeight: 800, color: "#10b981" }}>Foresight 1.0</div>
        <button onClick={doConnect} style={menuBtnStyle}>Connect</button>
        <button
          onClick={doStart}
          style={{
            ...menuBtnStyle,
            opacity: connected ? 1 : 0.5,
            cursor: connected ? "pointer" : "not-allowed",
          }}
        >
          Start
        </button>
        <button onClick={doStop} style={menuBtnStyle}>Stop</button>

        {/* Mode dropdown trigger */}
        <button
          ref={modeBtnRef}
          onClick={() => setModeOpen((v) => !v)}
          style={{ ...menuBtnStyle, borderColor: "rgba(16,185,129,0.35)" }}
          title="Switch SAR / Suspect-Lock"
        >
          Mode ▾
        </button>

        {/* Inline stats */}
        <div style={{ marginLeft: "auto", display: "flex", gap: 20 }}>
          <span>FPS: <span style={{ color: fps < 18 ? "#ef4444" : "#10b981" }}>{fps}</span></span>
          <span>GPU: {gpu}%</span>
          <span>Lat: {lat.toFixed(1)}</span>
          <span>Err: ±{geoErr}m</span>
        </div>
      </div>

      {/* Mode dropdown */}
      <Dropdown open={modeOpen} anchorRef={modeBtnRef} onClose={() => setModeOpen(false)} width={300}>
        <div style={{ padding: 6 }}>
          <div style={{ fontWeight: 700, marginBottom: 8, color: "#10b981" }}>Modes</div>
          <button onClick={() => chooseMode("SAR")} style={modeItemStyle(mode === "SAR")}>
            <span>SAR (Search &amp; Rescue)</span>
            {mode === "SAR" && <Dot ok size={10} />}
          </button>
          <button onClick={() => chooseMode("Suspect-Lock")} style={modeItemStyle(mode === "Suspect-Lock")}>
            <span>Suspect-Lock</span>
            {mode === "Suspect-Lock" && <Dot ok size={10} />}
          </button>
          <Toggle label="Face blur (bystanders)" checked={faceBlur} onChange={setFaceBlur} />
        </div>
      </Dropdown>

      {/* Main content (fills screen) */}
      <div
        style={{
          flex: 1,
          display: "grid",
          gridTemplateColumns: "1fr 380px",
          gap: 18,
          padding: 18,
          overflow: "hidden",
        }}
      >
        {/* Left video */}
        <div
          style={{
            width: "100%",
            height: "100%",
            background: "#000",
            borderRadius: 12,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          {!running ? (
            <div style={{ textAlign: "center", color: "#10b981" }}>
              <div
                style={{
                  width: 140,
                  height: 140,
                  borderRadius: "50%",
                  background: "rgba(16,185,129,0.12)",
                  border: "2px solid rgba(16,185,129,0.4)",
                  display: "grid",
                  placeItems: "center",
                  margin: "0 auto 16px auto",
                }}
              >
                <svg width="70" height="70" viewBox="0 0 24 24" fill="#10b981">
                  <path d="M8 5v14l11-7z" />
                </svg>
              </div>
              <div style={{ fontSize: 32, letterSpacing: 2 }}>
                NO SIGNAL… {mode === "SAR" ? "(SAR)" : "(Suspect-Lock)"} {faceBlur ? "• BLUR" : ""}
              </div>
            </div>
          ) : (
            <div style={{ width: "100%", height: "100%", background: "#111" }} />
          )}
        </div>

        {/* Right panels (scrollable) */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 12,
            overflowY: "auto",
          }}
        >
          <Collapsible title="Detections">
            <div>Person detected @14.59, 121.01</div>
            <div style={{ display: "flex", gap: 6, marginTop: 6 }}>
              <MiniBtn>Confirm</MiniBtn>
              <MiniBtn>Reject</MiniBtn>
              <MiniBtn>Handoff</MiniBtn>
            </div>
          </Collapsible>
          <Collapsible title="Suspect">
            <div>Drop reference image(s) here</div>
          </Collapsible>
          <Collapsible title="Maps">
            <div style={{ height: 180, background: "#222" }}>🗺️ Map here</div>
          </Collapsible>
          <Collapsible title="Logs" defaultOpen={false}>
            <div style={{ maxHeight: 150, overflowY: "auto", fontSize: 12 }}>
              {logs.map((l, i) => (
                <div key={i}>{l}</div>
              ))}
            </div>
          </Collapsible>
        </div>
      </div>

      {/* Bottom status bar */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          padding: "8px 18px",
          borderTop: "1px solid rgba(255,255,255,0.08)",
          fontSize: 14,
        }}
      >
        <div>
          Status:{" "}
          <span style={{ color: running ? "#10b981" : "#ef4444" }}>
            {running ? "running" : "error"}
          </span>
        </div>
        <div>
          <Dot ok={connected} /> ADB
          <Dot ok={connected} /> scrcpy
          <Dot ok={running} /> FFmpeg
        </div>
        <div>
          Latency:{" "}
          <span style={{ color: running ? "#10b981" : "#ef4444" }}>
            {latency}ms
          </span>{" "}
          | Disk: 73%
        </div>
        <div>
          Model: <span style={{ color: "#10b981" }}>yolo_sar_n.onnx</span> |
          Telemetry: OCR
        </div>
      </div>
    </div>
  );
}

/* ---------- styles ---------- */
const menuBtnStyle = {
  padding: "6px 12px",
  borderRadius: 6,
  border: "1px solid rgba(255,255,255,0.08)",
  background: "rgba(255,255,255,0.06)",
  color: "#E5E7EB",
  cursor: "pointer",
};
