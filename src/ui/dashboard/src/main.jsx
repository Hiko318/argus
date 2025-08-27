import React, { useEffect, useState } from "react";
import ReactDOM from "react-dom/client";
import { MapContainer, TileLayer, Marker, Popup, Circle } from "react-leaflet";

function App() {
  const [tick, setTick] = useState(0);
  const [telemetry, setTelemetry] = useState(null);
  const [dets, setDets] = useState([]);

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws");
    ws.onmessage = (evt) => {
      const msg = JSON.parse(evt.data);
      if (msg.type === "tick") {
        setTick(msg.time);
        setTelemetry(msg.telemetry);
        setDets(msg.detections || []);
      }
    };
    ws.onclose = () => console.log("WS closed");
    return () => ws.close();
  }, []);

  const center = telemetry ? [telemetry.lat, telemetry.lon] : [14.5995, 120.9842];

  return (
    <div style={{display:"grid", gridTemplateColumns:"280px 1fr", height:"100vh"}}>
      <aside style={{padding:"12px", borderRight:"1px solid #e5e7eb"}}>
        <h1 style={{margin:0}}>FORESIGHT</h1>
        <div style={{marginTop:8}}>
          <label>
            <input type="radio" name="mode" defaultChecked /> SAR Mode
          </label>
          <br/>
          <label>
            <input type="radio" name="mode" /> Suspect-Lock
          </label>
        </div>
        <div style={{marginTop:12}}>
          <strong>Tick:</strong> {tick}s
        </div>
        <div style={{marginTop:12}}>
          <strong>Detections</strong>
          <ul>
            {dets.map(d => (
              <li key={d.id}>
                {d.cls} {Math.round(d.conf*100)}% @ err ±{d.geo?.err_m ?? "?"}m
              </li>
            ))}
          </ul>
        </div>
      </aside>
      <main>
        <MapContainer center={center} zoom={17} style={{height:"100%", width:"100%"}}>
          <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"/>
          {telemetry && (
            <>
              <Marker position={[telemetry.lat, telemetry.lon]}>
                <Popup>Drone AGL {telemetry.alt}m</Popup>
              </Marker>
              <Circle center={[telemetry.lat, telemetry.lon]} radius={10}/>
            </>
          )}
          {dets.map(d => (
            <Marker key={d.id} position={[d.geo.lat, d.geo.lon]}>
              <Popup>{d.cls} {Math.round(d.conf*100)}%</Popup>
            </Marker>
          ))}
        </MapContainer>
      </main>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
