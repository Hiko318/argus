import React, { useState, useRef } from 'react';

const SuspectLockDialog = ({ 
  suspectImgs = [],
  onImagesChange,
  onLockToggle,
  isLocked = false,
  className = ""
}) => {
  const filePickerRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);
  const [lockConfidence, setLockConfidence] = useState(0.75);
  const [lockMode, setLockMode] = useState('face'); // 'face', 'full', 'clothing'

  const openFilePicker = () => {
    filePickerRef.current?.click();
  };

  const onFilesChosen = (e) => {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;
    
    const next = files.map((f) => ({ 
      name: f.name, 
      url: URL.createObjectURL(f),
      size: f.size,
      type: f.type
    }));
    
    onImagesChange([...suspectImgs, ...next]);
    e.target.value = "";
  };

  const clearUploads = () => {
    suspectImgs.forEach((i) => URL.revokeObjectURL(i.url));
    onImagesChange([]);
  };

  const removeImage = (index) => {
    const newImages = [...suspectImgs];
    URL.revokeObjectURL(newImages[index].url);
    newImages.splice(index, 1);
    onImagesChange(newImages);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    
    const files = Array.from(e.dataTransfer.files).filter(file => 
      file.type.startsWith('image/')
    );
    
    if (files.length > 0) {
      const next = files.map((f) => ({ 
        name: f.name, 
        url: URL.createObjectURL(f),
        size: f.size,
        type: f.type
      }));
      
      onImagesChange([...suspectImgs, ...next]);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  const uploadBtnStyle = {
    padding: "6px 12px",
    borderRadius: 8,
    border: "1px solid rgba(16,185,129,0.35)",
    background: "rgba(16,185,129,0.12)",
    color: "#10b981",
    cursor: "pointer",
    fontSize: 14
  };

  const lockBtnStyle = {
    padding: "8px 16px",
    borderRadius: 8,
    border: isLocked ? "1px solid rgba(239,68,68,0.35)" : "1px solid rgba(16,185,129,0.35)",
    background: isLocked ? "rgba(239,68,68,0.12)" : "rgba(16,185,129,0.12)",
    color: isLocked ? "#ef4444" : "#10b981",
    cursor: "pointer",
    fontSize: 14,
    fontWeight: 600
  };

  const miniBtn = {
    padding: "4px 8px",
    borderRadius: 6,
    background: "rgba(255,255,255,0.06)",
    border: "1px solid rgba(255,255,255,0.08)",
    color: "#E5E7EB",
    cursor: "pointer",
    fontSize: 12,
  };

  return (
    <div className={className}>
      {/* Upload Section */}
      <div style={{ marginBottom: 16 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
          <button onClick={openFilePicker} style={uploadBtnStyle}>
            📷 Upload photo(s)
          </button>
          <input
            ref={filePickerRef}
            type="file"
            accept="image/*"
            multiple
            onChange={onFilesChosen}
            style={{ display: "none" }}
          />
          <span style={{ color: "#9ca3af", fontSize: 12 }}>or drag & drop below</span>
        </div>

        {/* Drop Zone */}
        <div 
          style={{ 
            border: dragOver 
              ? "2px dashed rgba(16,185,129,0.5)" 
              : "1px dashed rgba(255,255,255,0.15)", 
            padding: 16, 
            borderRadius: 10, 
            textAlign: "center", 
            color: dragOver ? "#10b981" : "#9ca3af",
            background: dragOver ? "rgba(16,185,129,0.05)" : "transparent",
            transition: "all 0.2s ease"
          }}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          {dragOver ? (
            <div>
              <div style={{ fontSize: 24, marginBottom: 8 }}>📁</div>
              <div>Drop images here</div>
            </div>
          ) : (
            <div>
              <div style={{ fontSize: 20, marginBottom: 8 }}>📷</div>
              <div>Drop reference image(s) here</div>
              <div style={{ fontSize: 11, marginTop: 4, opacity: 0.7 }}>Supports JPG, PNG, WebP</div>
            </div>
          )}
        </div>
      </div>

      {/* Image Gallery */}
      {suspectImgs.length > 0 && (
        <div style={{ marginBottom: 16 }}>
          <div style={{ 
            display: "grid", 
            gridTemplateColumns: "repeat(auto-fill, minmax(80px, 1fr))", 
            gap: 8, 
            marginBottom: 12 
          }}>
            {suspectImgs.map((img, i) => (
              <div 
                key={`${img.name}-${i}`} 
                style={{ 
                  position: "relative", 
                  height: 80, 
                  borderRadius: 8, 
                  overflow: "hidden", 
                  border: "1px solid rgba(255,255,255,0.08)",
                  background: "#000"
                }}
              >
                <img 
                  src={img.url} 
                  alt={img.name} 
                  style={{ 
                    width: "100%", 
                    height: "100%", 
                    objectFit: "cover" 
                  }} 
                />
                
                {/* Remove button */}
                <button
                  onClick={() => removeImage(i)}
                  style={{
                    position: "absolute",
                    top: 4,
                    right: 4,
                    width: 20,
                    height: 20,
                    borderRadius: "50%",
                    background: "rgba(239,68,68,0.8)",
                    border: "none",
                    color: "#fff",
                    cursor: "pointer",
                    fontSize: 12,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center"
                  }}
                  title="Remove image"
                >
                  ×
                </button>
                
                {/* File info */}
                <div style={{
                  position: "absolute",
                  bottom: 0,
                  left: 0,
                  right: 0,
                  background: "rgba(0,0,0,0.7)",
                  color: "#fff",
                  padding: "2px 4px",
                  fontSize: 9,
                  overflow: "hidden",
                  whiteSpace: "nowrap",
                  textOverflow: "ellipsis"
                }}>
                  {img.name}
                  <br />
                  {formatFileSize(img.size)}
                </div>
              </div>
            ))}
          </div>
          
          <div style={{ display: "flex", gap: 8 }}>
            <button onClick={clearUploads} style={miniBtn}>
              🗑️ Clear All
            </button>
            <span style={{ color: "#9ca3af", fontSize: 12, alignSelf: "center" }}>
              {suspectImgs.length} image{suspectImgs.length !== 1 ? 's' : ''} uploaded
            </span>
          </div>
        </div>
      )}

      {/* Lock Configuration */}
      {suspectImgs.length > 0 && (
        <div style={{ 
          background: "rgba(255,255,255,0.02)", 
          border: "1px solid rgba(255,255,255,0.08)", 
          borderRadius: 8, 
          padding: 12,
          marginBottom: 16
        }}>
          <div style={{ fontWeight: 600, marginBottom: 12, color: "#10b981" }}>
            🎯 Lock Configuration
          </div>
          
          {/* Lock Mode */}
          <div style={{ marginBottom: 12 }}>
            <label style={{ display: "block", marginBottom: 6, fontSize: 12, color: "#9ca3af" }}>
              Detection Mode:
            </label>
            <div style={{ display: "flex", gap: 8 }}>
              {[
                { value: 'face', label: '👤 Face', desc: 'Face recognition' },
                { value: 'full', label: '🧍 Full Body', desc: 'Full person matching' },
                { value: 'clothing', label: '👕 Clothing', desc: 'Clothing patterns' }
              ].map(mode => (
                <button
                  key={mode.value}
                  onClick={() => setLockMode(mode.value)}
                  style={{
                    ...miniBtn,
                    background: lockMode === mode.value ? "rgba(16,185,129,0.12)" : "rgba(255,255,255,0.06)",
                    color: lockMode === mode.value ? "#10b981" : "#E5E7EB",
                    border: lockMode === mode.value ? "1px solid rgba(16,185,129,0.35)" : "1px solid rgba(255,255,255,0.08)"
                  }}
                  title={mode.desc}
                >
                  {mode.label}
                </button>
              ))}
            </div>
          </div>
          
          {/* Confidence Threshold */}
          <div style={{ marginBottom: 12 }}>
            <label style={{ display: "block", marginBottom: 6, fontSize: 12, color: "#9ca3af" }}>
              Confidence Threshold: {(lockConfidence * 100).toFixed(0)}%
            </label>
            <input
              type="range"
              min="0.3"
              max="0.95"
              step="0.05"
              value={lockConfidence}
              onChange={(e) => setLockConfidence(parseFloat(e.target.value))}
              style={{
                width: "100%",
                accentColor: "#10b981"
              }}
            />
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#6b7280", marginTop: 2 }}>
              <span>Less strict</span>
              <span>More strict</span>
            </div>
          </div>
        </div>
      )}

      {/* Lock Toggle */}
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <button 
          onClick={() => onLockToggle(!isLocked, { mode: lockMode, confidence: lockConfidence })}
          style={lockBtnStyle}
          disabled={suspectImgs.length === 0}
        >
          {isLocked ? "🔓 Unlock Suspect" : "🔒 Lock on Suspect"}
        </button>
        
        {isLocked && (
          <div style={{ 
            display: "flex", 
            alignItems: "center", 
            gap: 6,
            color: "#ef4444",
            fontSize: 12
          }}>
            <div style={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              background: "#ef4444",
              animation: "pulse 2s infinite"
            }} />
            <span>SUSPECT LOCK ACTIVE</span>
          </div>
        )}
        
        {suspectImgs.length === 0 && (
          <span style={{ color: "#6b7280", fontSize: 12 }}>
            Upload reference images to enable lock
          </span>
        )}
      </div>
      
      <style jsx>{`
        @keyframes pulse {
          0% { opacity: 1; }
          50% { opacity: 0.5; }
          100% { opacity: 1; }
        }
      `}</style>
    </div>
  );
};

export default SuspectLockDialog;