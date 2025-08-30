// src/ui/dashboard/src/App.jsx
import { useState, useEffect, useRef } from "react";

const PREVIEW_URL = "http://127.0.0.1:8000/preview";
const STREAM_URL  = "http://127.0.0.1:8000/stream.mjpg";
const HEALTH_URL  = "http://127.0.0.1:8000/health";

export default function App() {
  const [tick, setTick] = useState(0);
  const [connected, setConnected] = useState(false);
  const [started, setStarted] = useState(false);
  const [lastOk, setLastOk] = useState(Date.now());
  const [method, setMethod] = useState("polling"); // "polling" | "streaming"
  const [health, setHealth] = useState(null);

  const pollRef = useRef(null);
  const healthRef = useRef(null);

  const stalled = started && Date.now() - lastOk > 3000;

  // polling tick
  useEffect(() => {
    if (connected && started && method === "polling") {
      pollRef.current = setInterval(() => setTick(t => t + 1), 250);
    } else if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    return () => pollRef.current && clearInterval(pollRef.current);
  }, [connected, started, method]);

  // backend health
  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const r = await fetch(HEALTH_URL);
        const j = await r.json();
        setHealth(j);
      } catch (e) {
        setHealth({ status: "error", error: String(e) });
      }
    };
    if (connected) {
      fetchHealth();
      healthRef.current = setInterval(fetchHealth, 5000);
    }
    return () => healthRef.current && clearInterval(healthRef.current);
  }, [connected]);

  return (
    <div style={{ padding: 16, color: "#e5e7eb", background: "#111827", minHeight: "100vh" }}>
      <h2 style={{ marginBottom: 10 }}>Android Screen Dashboard</h2>

      <div style={{ marginBottom: 10 }}>
        <label style={{ marginRight: 16 }}>
          <input
            type="radio"
            value="polling"
            checked={method === "polling"}
            onChange={(e) => setMethod(e.target.value)}
            disabled={started}
          />{" "}
          Polling (250ms)
        </label>
        <label>
          <input
            type="radio"
            value="streaming"
            checked={method === "streaming"}
            onChange={(e) => setMethod(e.target.value)}
            disabled={started}
          />{" "}
          MJPEG stream
        </label>
      </div>

      <div style={{ marginBottom: 10 }}>
        {!connected ? (
          <button onClick={() => setConnected(true)}>Connect</button>
        ) : (
          <>
            <button onClick={() => { setConnected(false); setStarted(false); setTick(0); }}>Disconnect</button>{" "}
            {!started ? (
              <button onClick={() => { setStarted(true); setLastOk(Date.now()); }}>Start</button>
            ) : (
              <button onClick={() => { setStarted(false); setTick(0); }}>Stop</button>
            )}
          </>
        )}
      </div>

      {connected && (
        <div style={{ marginBottom: 12, padding: 8, background: "#0b0f1a", borderRadius: 8 }}>
          <div>
            <strong>Status:</strong> {started ? "Running" : "Connected"} |{" "}
            <strong>Method:</strong>{" "}
            {method === "polling" ? `Polling (tick ${tick})` : "MJPEG stream"}{" "}
            {stalled && <span style={{ color: "#f59e0b" }}>— Stalled (no new frames)</span>}
          </div>
          {health && (
            <div style={{ fontSize: 12, marginTop: 4 }}>
              <strong>Backend:</strong> {health.status}
              {health.frame_file?.exists && (
                <span>
                  {" "}
                  | <strong>Frame:</strong> {health.frame_file.size_kb} KB, age {health.frame_file.age_seconds}s
                </span>
              )}
              {health.error && <span style={{ color: "#f87171" }}> | {health.error}</span>}
            </div>
          )}
        </div>
      )}

      {connected && started && (
        <div style={{ border: "2px solid #222", borderRadius: 8, overflow: "hidden", maxHeight: "80vh" }}>
          {method === "polling" ? (
            <img
              src={`${PREVIEW_URL}?t=${tick}`}
              alt="Android"
              onLoad={() => setLastOk(Date.now())}
              onError={() => {}}
              style={{ display: "block", maxWidth: "100%", maxHeight: "80vh" }}
            />
          ) : (
            <img
              src={STREAM_URL}
              alt="Android stream"
              onLoad={() => setLastOk(Date.now())}
              onError={() => {}}
              style={{ display: "block", maxWidth: "100%", maxHeight: "80vh" }}
            />
          )}
        </div>
      )}
    </div>
  );
}
