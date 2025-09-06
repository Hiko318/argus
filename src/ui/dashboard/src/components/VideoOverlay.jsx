import React, { useState, useEffect } from 'react';

const VideoOverlay = ({ 
  running, 
  method, 
  previewSrc, 
  streamSrc, 
  mode, 
  faceBlur, 
  onImageLoad, 
  onImageError,
  stalled,
  errorsInARow 
}) => {
  const IdleBanner = ({ mode, faceBlur }) => {
    return (
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
        <div style={{ fontSize: 28, letterSpacing: 1 }}>
          NO SIGNAL… {mode === "SAR" ? "(SAR)" : "(Suspect-Lock)"}{
            faceBlur ? " • BLUR" : ""
          }
        </div>
      </div>
    );
  };

  const Overlay = ({ msg }) => {
    return (
      <div
        style={{
          position: "absolute",
          bottom: 12,
          left: "50%",
          transform: "translateX(-50%)",
          background: "rgba(0,0,0,0.55)",
          border: "1px solid rgba(255,255,255,0.12)",
          padding: "8px 12px",
          borderRadius: 8,
          color: "#E5E7EB",
          fontSize: 13,
          backdropFilter: "blur(2px)",
        }}
      >
        {msg}
      </div>
    );
  };

  return (
    <div
      style={{
        position: "relative",
        width: "100%",
        height: "100%",
        background: "#000",
        borderRadius: 12,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        overflow: "hidden",
      }}
    >
      {!running ? (
        <IdleBanner mode={mode} faceBlur={faceBlur} />
      ) : method === "polling" ? (
        <>
          <img
            src={previewSrc}
            alt="Live Feed"
            style={{ width: "100%", height: "100%", objectFit: "contain" }}
            onLoad={onImageLoad}
            onError={onImageError}
            draggable={false}
          />
          {stalled && <Overlay msg="Stalled — no new frames. Check backend/ffmpeg." />}
          {errorsInARow > 0 && !stalled && <Overlay msg="Fetching… (retrying)" />}
        </>
      ) : (
        <>
          <img
            src={streamSrc}
            alt="Live MJPEG"
            style={{ width: "100%", height: "100%", objectFit: "contain" }}
            onLoad={onImageLoad}
            onError={onImageError}
            draggable={false}
          />
          {stalled && <Overlay msg="Stream appears stalled — check backend." />}
        </>
      )}
    </div>
  );
};

export default VideoOverlay;