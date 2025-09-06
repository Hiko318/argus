import React, { useState, useEffect } from 'react';

const MapPanel = ({ 
  detections = [],
  homeLocation = { lat: 14.8, lon: 121.01 },
  geoError = 7.5,
  className = ""
}) => {
  const [mapMode, setMapMode] = useState('satellite'); // 'satellite', 'terrain', 'hybrid'
  const [showDetections, setShowDetections] = useState(true);
  const [showSearchArea, setShowSearchArea] = useState(true);

  // Mock detection data for demonstration
  const mockDetections = [
    { id: 1, lat: 14.59, lon: 121.01, confidence: 0.85, timestamp: Date.now() - 30000 },
    { id: 2, lat: 14.61, lon: 121.02, confidence: 0.72, timestamp: Date.now() - 120000 },
  ];

  const activeDetections = detections.length > 0 ? detections : mockDetections;

  const MapControls = () => (
    <div style={{
      position: 'absolute',
      top: 8,
      right: 8,
      display: 'flex',
      flexDirection: 'column',
      gap: 4,
      zIndex: 10
    }}>
      <button
        onClick={() => setMapMode(mapMode === 'satellite' ? 'terrain' : 'satellite')}
        style={{
          padding: '4px 8px',
          fontSize: 11,
          background: 'rgba(0,0,0,0.7)',
          color: '#fff',
          border: '1px solid rgba(255,255,255,0.2)',
          borderRadius: 4,
          cursor: 'pointer'
        }}
      >
        {mapMode === 'satellite' ? '🗺️' : '🛰️'}
      </button>
      
      <button
        onClick={() => setShowDetections(!showDetections)}
        style={{
          padding: '4px 8px',
          fontSize: 11,
          background: showDetections ? 'rgba(16,185,129,0.8)' : 'rgba(0,0,0,0.7)',
          color: '#fff',
          border: '1px solid rgba(255,255,255,0.2)',
          borderRadius: 4,
          cursor: 'pointer'
        }}
      >
        📍
      </button>
      
      <button
        onClick={() => setShowSearchArea(!showSearchArea)}
        style={{
          padding: '4px 8px',
          fontSize: 11,
          background: showSearchArea ? 'rgba(16,185,129,0.8)' : 'rgba(0,0,0,0.7)',
          color: '#fff',
          border: '1px solid rgba(255,255,255,0.2)',
          borderRadius: 4,
          cursor: 'pointer'
        }}
      >
        ⭕
      </button>
    </div>
  );

  const DetectionMarkers = () => (
    showDetections && activeDetections.map(detection => {
      const age = (Date.now() - detection.timestamp) / 1000; // seconds
      const opacity = Math.max(0.3, 1 - (age / 300)); // fade over 5 minutes
      
      return (
        <div
          key={detection.id}
          style={{
            position: 'absolute',
            left: `${45 + (detection.lon - homeLocation.lon) * 1000}%`,
            top: `${45 + (homeLocation.lat - detection.lat) * 1000}%`,
            transform: 'translate(-50%, -50%)',
            zIndex: 5
          }}
        >
          <div style={{
            width: 12,
            height: 12,
            borderRadius: '50%',
            background: `rgba(255, 0, 0, ${opacity})`,
            border: '2px solid rgba(255, 255, 255, 0.8)',
            boxShadow: '0 0 8px rgba(255, 0, 0, 0.6)',
            animation: age < 10 ? 'pulse 2s infinite' : 'none'
          }} />
          
          <div style={{
            position: 'absolute',
            top: -25,
            left: '50%',
            transform: 'translateX(-50%)',
            background: 'rgba(0,0,0,0.8)',
            color: '#fff',
            padding: '2px 6px',
            borderRadius: 4,
            fontSize: 10,
            whiteSpace: 'nowrap',
            opacity: opacity
          }}>
            {(detection.confidence * 100).toFixed(0)}%
          </div>
        </div>
      );
    })
  );

  const SearchArea = () => (
    showSearchArea && (
      <div style={{
        position: 'absolute',
        left: '50%',
        top: '50%',
        transform: 'translate(-50%, -50%)',
        width: `${geoError * 4}px`,
        height: `${geoError * 4}px`,
        border: '2px dashed rgba(16,185,129,0.6)',
        borderRadius: '50%',
        zIndex: 3
      }} />
    )
  );

  const HomeMarker = () => (
    <div style={{
      position: 'absolute',
      left: '50%',
      top: '50%',
      transform: 'translate(-50%, -50%)',
      zIndex: 4
    }}>
      <div style={{
        width: 16,
        height: 16,
        background: '#10b981',
        border: '3px solid #fff',
        borderRadius: '50%',
        boxShadow: '0 0 12px rgba(16,185,129,0.6)'
      }} />
      
      <div style={{
        position: 'absolute',
        top: -30,
        left: '50%',
        transform: 'translateX(-50%)',
        background: 'rgba(16,185,129,0.9)',
        color: '#fff',
        padding: '2px 6px',
        borderRadius: 4,
        fontSize: 10,
        whiteSpace: 'nowrap'
      }}>
        HOME
      </div>
    </div>
  );

  return (
    <div className={className} style={{
      position: 'relative',
      height: 180,
      background: mapMode === 'satellite' 
        ? 'linear-gradient(45deg, #2d5016 0%, #4a7c59 50%, #2d5016 100%)'
        : 'linear-gradient(45deg, #8B7355 0%, #A0937D 50%, #8B7355 100%)',
      borderRadius: 8,
      overflow: 'hidden',
      border: '1px solid rgba(255,255,255,0.1)'
    }}>
      {/* Map background pattern */}
      <div style={{
        position: 'absolute',
        inset: 0,
        backgroundImage: mapMode === 'satellite'
          ? `radial-gradient(circle at 20% 30%, rgba(255,255,255,0.1) 1px, transparent 1px),
             radial-gradient(circle at 80% 70%, rgba(255,255,255,0.05) 1px, transparent 1px)`
          : `linear-gradient(45deg, rgba(255,255,255,0.05) 25%, transparent 25%),
             linear-gradient(-45deg, rgba(255,255,255,0.05) 25%, transparent 25%)`,
        backgroundSize: mapMode === 'satellite' ? '40px 40px' : '20px 20px'
      }} />
      
      {/* Grid overlay */}
      <div style={{
        position: 'absolute',
        inset: 0,
        backgroundImage: `
          linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px),
          linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)
        `,
        backgroundSize: '30px 30px'
      }} />
      
      <MapControls />
      <SearchArea />
      <HomeMarker />
      <DetectionMarkers />
      
      {/* Coordinates display */}
      <div style={{
        position: 'absolute',
        bottom: 8,
        left: 8,
        background: 'rgba(0,0,0,0.7)',
        color: '#fff',
        padding: '4px 8px',
        borderRadius: 4,
        fontSize: 11,
        fontFamily: 'monospace'
      }}>
        {homeLocation.lat.toFixed(4)}, {homeLocation.lon.toFixed(4)}
        <br />
        ±{geoError}m accuracy
      </div>
      
      {/* Detection count */}
      {showDetections && activeDetections.length > 0 && (
        <div style={{
          position: 'absolute',
          top: 8,
          left: 8,
          background: 'rgba(255,0,0,0.8)',
          color: '#fff',
          padding: '4px 8px',
          borderRadius: 4,
          fontSize: 11,
          fontWeight: 'bold'
        }}>
          {activeDetections.length} detection{activeDetections.length !== 1 ? 's' : ''}
        </div>
      )}
      
      <style jsx>{`
        @keyframes pulse {
          0% { transform: translate(-50%, -50%) scale(1); }
          50% { transform: translate(-50%, -50%) scale(1.2); }
          100% { transform: translate(-50%, -50%) scale(1); }
        }
      `}</style>
    </div>
  );
};

export default MapPanel;