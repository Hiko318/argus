import { useEffect, useState } from "react";

function Toggle({ label, checked, onChange }) {
  return (
    <label style={{ 
      display: "inline-flex", 
      gap: 8, 
      alignItems: "center", 
      marginRight: 16,
      padding: "8px 12px",
      borderRadius: "6px",
      backgroundColor: checked ? "#10b981" : "#374151",
      color: "white",
      cursor: "pointer",
      transition: "all 0.2s"
    }}>
      <input 
        type="checkbox" 
        checked={checked} 
        onChange={(e) => onChange(e.target.checked)}
        style={{ marginRight: "4px" }}
      />
      <span>{label}</span>
    </label>
  );
}

export default function App() {
  // YOLO Detection states
  const [sar, setSar] = useState(true);
  const [lock, setLock] = useState(false);
  
  // Geolocation states
  const [geo, setGeo] = useState({ lat: 0, lon: 0 });
  const [heading, setHeading] = useState(0);
  
  // Video stream state
  const [videoError, setVideoError] = useState(false);

  // Fetch initial state and geolocation
  useEffect(() => {
    // Fetch YOLO state
    fetch("http://localhost:8000/state")
      .then(r => r.json())
      .then(s => { 
        setSar(!!s.sar); 
        setLock(!!s.lock); 
      })
      .catch(() => {});

    // Fetch geolocation
    const fetchGeo = async () => {
      try {
        const res = await fetch("http://localhost:8000/geolocate");
        const data = await res.json();
        setGeo({ lat: data.lat, lon: data.lon });
      } catch (err) {
        console.error("Geo fetch error:", err);
      }
    };

    fetchGeo();
    const interval = setInterval(fetchGeo, 5000); // refresh every 5s
    return () => clearInterval(interval);
  }, []);

  // Update SAR mode
  const updateSar = async (enabled) => {
    setSar(enabled);
    try {
      await fetch("http://localhost:8000/toggle/sar", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled })
      });
    } catch (err) {
      console.error("SAR toggle error:", err);
    }
  };

  // Update suspect lock
  const updateLock = async (enabled) => {
    setLock(enabled);
    try {
      await fetch("http://localhost:8000/toggle/lock", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled })
      });
    } catch (err) {
      console.error("Lock toggle error:", err);
    }
  };

  // Send heading
  const sendHeading = (val) => {
    const headingValue = Number(val);
    setHeading(headingValue);
    fetch("http://localhost:8000/heading", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ heading: headingValue })
    }).catch(err => console.error("Heading error:", err));
  };

  // Set home location
  const setHome = () => {
    navigator.geolocation.getCurrentPosition((pos) => {
      fetch("http://localhost:8000/home", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          lat: pos.coords.latitude,
          lon: pos.coords.longitude
        })
      }).catch(err => console.error("Home error:", err));
    }, (err) => {
      console.error("Geolocation error:", err);
      alert("Could not get your location. Please enable location services.");
    });
  };

  return (
    <div style={{ 
      minHeight: "100vh", 
      background: "linear-gradient(135deg, #0b0f1a 0%, #1a1f35 100%)", 
      color: "#e2e8f0", 
      padding: 20,
      fontFamily: "Arial, sans-serif"
    }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ 
          fontSize: "2.5rem",
          marginBottom: 8,
          background: "linear-gradient(90deg, #10b981, #3b82f6)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
          fontWeight: "bold"
        }}>
          FORESIGHT — SAR Console
        </h1>
        <p style={{ 
          marginTop: 0, 
          opacity: 0.8,
          fontSize: "1.1rem"
        }}>
          Live YOLO detections from mirrored phone (DJI Fly)
        </p>
      </div>

      {/* Control Panel */}
      <div style={{
        backgroundColor: "rgba(255, 255, 255, 0.1)",
        padding: "16px",
        borderRadius: "12px",
        marginBottom: "20px",
        backdropFilter: "blur(10px)",
        border: "1px solid rgba(255, 255, 255, 0.1)"
      }}>
        <h3 style={{ marginTop: 0, marginBottom: 16, color: "#10b981" }}>Detection Controls</h3>
        <div style={{ marginBottom: 16 }}>
          <Toggle label="SAR Mode" checked={sar} onChange={updateSar} />
          <Toggle label="Suspect Lock" checked={lock} onChange={updateLock} />
        </div>

        <h3 style={{ marginBottom: 16, color: "#3b82f6" }}>Navigation Controls</h3>
        <div style={{ display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <label htmlFor="heading">Heading:</label>
            <input
              id="heading"
              type="range"
              min="0"
              max="359"
              value={heading}
              onChange={(e) => sendHeading(e.target.value)}
              style={{ width: "200px" }}
            />
            <span style={{ 
              minWidth: "40px", 
              textAlign: "center",
              backgroundColor: "rgba(59, 130, 246, 0.2)",
              padding: "4px 8px",
              borderRadius: "4px"
            }}>
              {heading}°
            </span>
          </div>
          
          <button 
            onClick={setHome}
            style={{
              padding: "8px 16px",
              backgroundColor: "#3b82f6",
              color: "white",
              border: "none",
              borderRadius: "6px",
              cursor: "pointer",
              transition: "background-color 0.2s"
            }}
            onMouseOver={(e) => e.target.style.backgroundColor = "#2563eb"}
            onMouseOut={(e) => e.target.style.backgroundColor = "#3b82f6"}
          >
            Set Home
          </button>
        </div>

        <div style={{ 
          marginTop: 12, 
          fontSize: "0.9rem", 
          opacity: 0.8 
        }}>
          Current Location: {geo.lat.toFixed(4)}, {geo.lon.toFixed(4)}
        </div>
      </div>

      {/* Video Streams */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px", marginBottom: "20px" }}>
        {/* YOLO Detection Stream */}
        <div>
          <h3 style={{ marginBottom: 12, color: "#10b981" }}>YOLO Detection Stream</h3>
          <div style={{
            width: "100%", 
            aspectRatio: "16 / 9",
            borderRadius: 16, 
            overflow: "hidden", 
            boxShadow: "0 10px 30px rgba(0,0,0,0.5)",
            background: "#111",
            position: "relative"
          }}>
            <img
              src="http://localhost:8000/video.mjpg"
              alt="YOLO Detection Stream"
              style={{ 
                width: "100%", 
                height: "100%", 
                objectFit: "cover", 
                display: "block" 
              }}
              onError={() => setVideoError(true)}
              onLoad={() => setVideoError(false)}
            />
            {videoError && (
              <div style={{
                position: "absolute",
                top: "50%",
                left: "50%",
                transform: "translate(-50%, -50%)",
                textAlign: "center",
                color: "#ef4444"
              }}>
                <div>❌ RTSP Stream Not Available</div>
                <div style={{ fontSize: "0.8rem", marginTop: 8 }}>
                  Make sure Terminal 1 (capture_rtsp.bat) is running
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Desktop Capture Stream */}
        <div>
          <h3 style={{ marginBottom: 12, color: "#f59e0b" }}>Desktop Capture</h3>
          <div style={{
            width: "100%", 
            aspectRatio: "16 / 9",
            borderRadius: 16, 
            overflow: "hidden", 
            boxShadow: "0 10px 30px rgba(0,0,0,0.5)",
            background: "#111"
          }}>
            <img
              src="http://localhost:8000/mjpg"
              alt="Desktop Capture"
              style={{ 
                width: "100%", 
                height: "100%", 
                objectFit: "cover", 
                display: "block" 
              }}
            />
          </div>
        </div>
      </div>

      {/* Status Panel */}
      <div style={{
        backgroundColor: "rgba(255, 255, 255, 0.05)",
        padding: "16px",
        borderRadius: "12px",
        border: "1px solid rgba(255, 255, 255, 0.1)"
      }}>
        <h3 style={{ marginTop: 0, marginBottom: 12, color: "#8b5cf6" }}>System Status</h3>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 16 }}>
          <div>
            <strong>SAR Mode:</strong> 
            <span style={{ color: sar ? "#10b981" : "#ef4444", marginLeft: 8 }}>
              {sar ? "ACTIVE" : "INACTIVE"}
            </span>
          </div>
          <div>
            <strong>Suspect Lock:</strong> 
            <span style={{ color: lock ? "#f59e0b" : "#6b7280", marginLeft: 8 }}>
              {lock ? "ENGAGED" : "DISENGAGED"}
            </span>
          </div>
          <div>
            <strong>Heading:</strong> 
            <span style={{ color: "#3b82f6", marginLeft: 8 }}>{heading}°</span>
          </div>
          <div>
            <strong>RTSP Stream:</strong> 
            <span style={{ color: videoError ? "#ef4444" : "#10b981", marginLeft: 8 }}>
              {videoError ? "DISCONNECTED" : "CONNECTED"}
            </span>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div style={{ 
        marginTop: 32, 
        textAlign: "center", 
        opacity: 0.6,
        fontSize: "0.9rem"
      }}>
        <p>
          🎯 Terminal 1: RTSP Capture | 🤖 Terminal 2: FastAPI Server | 🌐 Terminal 3: React Dashboard
        </p>
        <p style={{ marginTop: 8 }}>
          API: <a href="http://localhost:8000" target="_blank" style={{ color: "#10b981" }}>localhost:8000</a> | 
          Dashboard: <a href="http://localhost:5173" target="_blank" style={{ color: "#3b82f6", marginLeft: 8 }}>localhost:5173</a>
        </p>
      </div>
    </div>
  );
}