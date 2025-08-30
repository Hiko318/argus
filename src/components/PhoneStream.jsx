import React, { useEffect, useRef, useState } from "react";

export default function PhoneStream({ running, backendBase = "http://127.0.0.1:8000" }) {
  const imgRef = useRef(null);
  const [ok, setOk] = useState(false);

  useEffect(() => {
    const base = `${backendBase}/preview`;
    let stop = false;

    function open() {
      if (!running || stop) {
        setOk(false);
        if (imgRef.current) imgRef.current.src = "";
        return;
      }
      const u = `${base}?_=${Date.now()}-${Math.random().toString(36).slice(2)}`;
      if (imgRef.current) imgRef.current.src = u;
    }

    open();
    const keepAlive = setInterval(open, 5000);

    return () => {
      stop = true;
      clearInterval(keepAlive);
      if (imgRef.current) imgRef.current.src = "";
    };
  }, [running, backendBase]);

  return (
    <div style={{ width: "100%", height: "100%", display: "grid", placeItems: "center", background: "#000", position: "relative" }}>
      <img
        ref={imgRef}
        alt="phone"
        style={{ width: "100%", height: "100%", objectFit: "contain" }}
        onLoad={() => setOk(true)}
        onError={() => setOk(false)}
      />
      {!ok && (
        <div style={{ position: "absolute", color: "#9ca3af", fontSize: 14, padding: "6px 10px", background: "rgba(0,0,0,0.5)", borderRadius: 8 }}>
          connecting…
        </div>
      )}
    </div>
  );
}
