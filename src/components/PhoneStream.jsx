import { useEffect, useRef } from "react";
import Hls from "hls.js";

export default function PhoneStream() {
  const videoRef = useRef(null);

  useEffect(() => {
    const video = videoRef.current;
    const src = "/stream/index.m3u8"; // HLS playlist we’ll generate with ffmpeg

    if (Hls.isSupported()) {
      const hls = new Hls({ liveDurationInfinity: true });
      hls.loadSource(src);
      hls.attachMedia(video);
      return () => hls.destroy();
    } else {
      video.src = src; // Safari fallback
    }
  }, []);

  return (
    <div className="relative w-full h-full bg-black rounded-lg overflow-hidden shadow">
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        className="w-full h-full object-contain"
      />
      {/* YOLO overlay canvas goes here later */}
      <canvas id="sar-overlay" className="absolute inset-0 pointer-events-none"></canvas>
    </div>
  );
}
