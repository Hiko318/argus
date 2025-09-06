import React, { useEffect, useRef, useState } from "react";
import VideoOverlay from './components/VideoOverlay';
import MapPanel from './components/MapPanel';
import SuspectLockDialog from './components/SuspectLockDialog';
import HandoffModal from './components/HandoffModal';

/* ---------- settings (edit if you ever change backend) ---------- */
const PREVIEW_URL = "http://127.0.0.1:8000/preview";
const STREAM_URL = "http://127.0.0.1:8000/stream.mjpg";
const REFRESH_MS = 250; // polling interval when using PREVIEW_URL
const STALL_MS = 3000;  // show stalled if no new frames in this many ms

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

function MiniBtn({ children, onClick }) {
  return (
    <button
      onClick={onClick}
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
        border: `1px solid ${
          checked ? "rgba(16,185,129,0.35)" : "rgba(255,255,255,0.08)"
        }`,
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

function Dropdown({ open, anchorRef, onClose, width = 320, children }) {
  const ref = useRef(null);

  useEffect(() => {
    function onDocClick(e) {
      if (!open) return;
      if (
        ref.current &&
        !ref.current.contains(e.target) &&
        !anchorRef?.current?.contains(e.target)
      ) {
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
  const rect =
    anchorRef?.current?.getBoundingClientRect?.() ?? { left: 24, bottom: 56 };
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

/* ---------- helpers ---------- */
// Component definitions moved to separate files

/* ---------- main app ---------- */
export default function App() {
  const [connected, setConnected] = useState(false);
  const [running, setRunning] = useState(false);
  const [method, setMethod] = useState("polling"); // 'polling' or 'mjpeg'
  const [fps, setFps] = useState(0);
  const [lat] = useState(14.8);
  const [geoErr] = useState(7.5);
  const [latency, setLatency] = useState(467);
  const [logs, setLogs] = useState(["UI loaded", "Click Connect then Start"]);

  // mode + face blur
  const [mode, setMode] = useState("SAR");
  const [faceBlur, setFaceBlur] = useState(true);

  // suspect uploads
  const [suspectImgs, setSuspectImgs] = useState([]);
  const [suspectLocked, setSuspectLocked] = useState(false);
  const [lockConfig, setLockConfig] = useState({ mode: 'face', confidence: 0.75 });

  // handoff modal
  const [handoffOpen, setHandoffOpen] = useState(false);

  // mock location data
  const [currentLocation] = useState({ lat: 37.7749, lng: -122.4194 });
  const [suspectData] = useState({ confidence: 0.87, lastSeen: Date.now() });

  // dropdown
  const modeBtnRef = useRef(null);
  const [modeOpen, setModeOpen] = useState(false);

  // stream states
  const [tick, setTick] = useState(0); // increments to bust cache for preview
  const [lastFrameAt, setLastFrameAt] = useState(0);
  const [errorsInARow, setErrorsInARow] = useState(0);

  const stalled = running && Date.now() - lastFrameAt > STALL_MS;

  // fps/latency mock just for top bar vibes
  useEffect(() => {
    let id;
    if (running) {
      id = setInterval(() => {
        setFps((f) =>
          Math.max(10, Math.min(60, Math.round(f + (Math.random() - 0.5) * 6)))
        );
        setLatency((l) =>
          Math.max(90, Math.min(900, Math.round(l + (Math.random() - 0.5) * 40)))
        );
      }, 900);
    } else {
      setFps(0);
      setLatency(0);
    }
    return () => clearInterval(id);
  }, [running]);

  // polling/tick effect (only when running, connected, method === 'polling', and tab visible)
  useEffect(() => {
    let id = null;
    function startTicking() {
      setTick((t) => t + 1); // immediate fetch
      id = setInterval(() => setTick((t) => t + 1), REFRESH_MS);
    }
    function stopTicking() {
      if (id) {
        clearInterval(id);
        id = null;
      }
    }

    function handleVisibilityChange() {
      if (document.hidden) {
        stopTicking();
      } else {
        // restart if conditions permit
        if (connected && running && method === "polling") startTicking();
      }
    }

    document.addEventListener("visibilitychange", handleVisibilityChange);

    if (connected && running && method === "polling" && !document.hidden) {
      startTicking();
    }

    return () => {
      stopTicking();
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [connected, running, method]);

  // handlers
  function pushLog(msg) {
    setLogs((l) => [...l.slice(-199), `${new Date().toLocaleTimeString()}  ${msg}`]);
  }

  function doConnect() {
    setConnected(true);
    pushLog("Connected (UI)");
  }
  function doStart() {
    if (!connected) {
      pushLog("Start blocked: not connected");
      return;
    }
    setRunning(true);
    setTick((t) => t + 1); // immediate
    pushLog("Pipeline started (UI polling)");
  }
  function doStop() {
    setRunning(false);
    pushLog("Pipeline stopped (UI)");
  }
  function chooseMode(next) {
    setMode(next);
    setModeOpen(false);
    if (next === "SAR") setFaceBlur(true);
    pushLog(`Mode set to ${next}`);
  }

  // File upload functions moved to SuspectLockDialog component

  // suspect lock handlers
  function handleLockToggle(locked, config) {
    setSuspectLocked(locked);
    setLockConfig(config);
    pushLog(locked ? `Suspect lock activated (${config.mode}, ${(config.confidence * 100).toFixed(0)}%)` : 'Suspect lock deactivated');
  }

  // handoff handlers
  function handleHandoff(handoffData) {
    pushLog(`Emergency handoff initiated to ${handoffData.type} (${handoffData.priority} priority)`);
    console.log('Handoff data:', handoffData);
    // In real implementation, this would send data to backend
    return Promise.resolve();
  }

  // image load / error handlers
  function handleImageLoad() {
    setLastFrameAt(Date.now());
    setErrorsInARow(0);
  }
  function handleImageError() {
    setErrorsInARow((e) => e + 1);
  }

  // computed image src
  const previewSrc = `${PREVIEW_URL}?t=${tick}`;
  const streamSrc = STREAM_URL;

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

        <button onClick={doConnect} style={menuBtnStyle}>
          Connect
        </button>

        <button
          onClick={doStart}
          style={{
            ...menuBtnStyle,
            opacity: connected ? 1 : 0.5,
            cursor: connected ? "pointer" : "not-allowed",
          }}
          title={!connected ? "Connect first" : "Start polling stream"}
        >
          Start
        </button>

        <button onClick={doStop} style={menuBtnStyle}>
          Stop
        </button>

        <button
          ref={modeBtnRef}
          onClick={() => setModeOpen((v) => !v)}
          style={{ ...menuBtnStyle, borderColor: "rgba(16,185,129,0.35)" }}
        >
          Mode ▾
        </button>

        <a
          href={PREVIEW_URL}
          target="_blank"
          rel="noreferrer"
          style={{ ...menuBtnStyle, textDecoration: "none" }}
          title="Open raw /preview in new tab"
        >
          Open /preview
        </a>

        <div style={{ marginLeft: "auto", display: "flex", gap: 20 }}>
          <span>
            FPS:{" "}
            <span style={{ color: fps < 18 ? "#ef4444" : "#10b981" }}>{fps}</span>
          </span>
          <span>Lat: {lat.toFixed(1)}</span>
          <span>Err: ±{geoErr}m</span>
        </div>
      </div>

      {/* Mode dropdown */}
      <Dropdown open={modeOpen} anchorRef={modeBtnRef} onClose={() => setModeOpen(false)} width={300}>
        <div style={{ padding: 6 }}>
          <div style={{ fontWeight: 700, marginBottom: 8, color: "#10b981" }}>Modes</div>
          <button onClick={() => chooseMode("SAR")} style={modeItemStyle(mode === "SAR")}>
            SAR (Search &amp; Rescue) {mode === "SAR" && <Dot ok size={10} />}
          </button>
          <button onClick={() => chooseMode("Suspect-Lock")} style={modeItemStyle(mode === "Suspect-Lock")}>
            Suspect-Lock {mode === "Suspect-Lock" && <Dot ok size={10} />}
          </button>

          <div style={{ marginTop: 8 }}>
            <Toggle label="Face blur (bystanders)" checked={faceBlur} onChange={setFaceBlur} />
          </div>
        </div>
      </Dropdown>

      {/* Main content */}
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
        <VideoOverlay
          running={running}
          method={method}
          mode={mode}
          faceBlur={faceBlur}
          previewSrc={previewSrc}
          streamSrc={streamSrc}
          stalled={stalled}
          errorsInARow={errorsInARow}
          onImageLoad={handleImageLoad}
          onImageError={handleImageError}
          style={{
            width: "100%",
            height: "100%",
            borderRadius: 12
          }}
              />

        {/* Right panels */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12, overflowY: "auto" }}>
          <Collapsible title="Detections">
            <div>Person detected @14.59, 121.01</div>
            <div style={{ display: "flex", gap: 6, marginTop: 6 }}>
              <MiniBtn>Confirm</MiniBtn>
              <MiniBtn>Reject</MiniBtn>
              <MiniBtn onClick={() => setHandoffOpen(true)}>Handoff</MiniBtn>
            </div>
          </Collapsible>

          <Collapsible title="Suspect Lock">
            <SuspectLockDialog
              suspectImgs={suspectImgs}
              onImagesChange={setSuspectImgs}
              onLockToggle={handleLockToggle}
              isLocked={suspectLocked}
            />
          </Collapsible>

          <Collapsible title="Maps">
            <MapPanel
              currentLocation={currentLocation}
              suspectData={suspectData}
              style={{ height: 180 }}
            />
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
      <div style={{ display: "flex", justifyContent: "space-between", padding: "8px 18px", borderTop: "1px solid rgba(255,255,255,0.08)", fontSize: 14 }}>
        <div>Status: <span style={{ color: running ? "#10b981" : "#ef4444" }}>{running ? "running" : "stopped"}</span>{stalled && <span style={{ color: "#f59e0b" }}> (stalled)</span>}</div>
        <div><Dot ok={connected} /> ADB <Dot ok={connected} /> scrcpy <Dot ok={running && !stalled && errorsInARow===0} /> FFmpeg</div>
        <div>Latency: <span style={{ color: running ? "#10b981" : "#ef4444" }}>{latency}ms</span> | Disk: 73%</div>
        <div>Model: <span style={{ color: "#10b981" }}>yolo_sar_n.onnx</span> | Telemetry: OCR</div>
      </div>

      {/* Handoff Modal */}
      <HandoffModal
        isOpen={handoffOpen}
        onClose={() => setHandoffOpen(false)}
        onHandoff={handleHandoff}
        currentLocation={currentLocation}
        suspectData={suspectData}
      />
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

const uploadBtnStyle = {
  padding: "6px 12px",
  borderRadius: 8,
  border: "1px solid rgba(16,185,129,0.35)",
  background: "rgba(16,185,129,0.12)",
  color: "#10b981",
  cursor: "pointer",
};

const modeItemStyle = (active) => ({
  width: "100%",
  textAlign: "left",
  padding: "10px",
  borderRadius: 8,
  border: "1px solid rgba(255,255,255,0.06)",
  color: active ? "#10b981" : "#E5E7EB",
  background: active ? "rgba(16,185,129,0.08)" : "transparent",
  cursor: "pointer",
});
