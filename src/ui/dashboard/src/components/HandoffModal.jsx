import React, { useState, useRef } from 'react';

const HandoffModal = ({ 
  isOpen = false,
  onClose,
  onHandoff,
  currentLocation = null,
  suspectData = null,
  className = ""
}) => {
  const [handoffType, setHandoffType] = useState('police'); // 'police', 'fire', 'ems', 'coast_guard', 'other'
  const [priority, setPriority] = useState('medium'); // 'low', 'medium', 'high', 'critical'
  const [contactMethod, setContactMethod] = useState('radio'); // 'radio', 'phone', 'dispatch', 'direct'
  const [notes, setNotes] = useState('');
  const [includeVideo, setIncludeVideo] = useState(true);
  const [includeLocation, setIncludeLocation] = useState(true);
  const [includeSuspectData, setIncludeSuspectData] = useState(true);
  const [estimatedETA, setEstimatedETA] = useState('5-10');
  const [contactInfo, setContactInfo] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  
  const modalRef = useRef(null);

  if (!isOpen) return null;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    
    const handoffData = {
      type: handoffType,
      priority,
      contactMethod,
      contactInfo,
      notes,
      estimatedETA,
      timestamp: new Date().toISOString(),
      location: includeLocation ? currentLocation : null,
      suspectData: includeSuspectData ? suspectData : null,
      includeVideo,
      operator: 'SAR-001' // This would come from auth context
    };
    
    try {
      await onHandoff(handoffData);
      onClose();
      // Reset form
      setNotes('');
      setContactInfo('');
      setHandoffType('police');
      setPriority('medium');
    } catch (error) {
      console.error('Handoff failed:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleBackdropClick = (e) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  const agencyTypes = [
    { value: 'police', label: '👮 Police', desc: 'Law enforcement' },
    { value: 'fire', label: '🚒 Fire Dept', desc: 'Fire & rescue' },
    { value: 'ems', label: '🚑 EMS', desc: 'Emergency medical' },
    { value: 'coast_guard', label: '⚓ Coast Guard', desc: 'Maritime rescue' },
    { value: 'other', label: '🏢 Other Agency', desc: 'Custom agency' }
  ];

  const priorityLevels = [
    { value: 'low', label: '🟢 Low', color: '#10b981' },
    { value: 'medium', label: '🟡 Medium', color: '#f59e0b' },
    { value: 'high', label: '🟠 High', color: '#f97316' },
    { value: 'critical', label: '🔴 Critical', color: '#ef4444' }
  ];

  const contactMethods = [
    { value: 'radio', label: '📻 Radio', desc: 'Emergency radio' },
    { value: 'phone', label: '📞 Phone', desc: 'Direct phone call' },
    { value: 'dispatch', label: '🎯 Dispatch', desc: 'Through dispatch center' },
    { value: 'direct', label: '🚁 Direct', desc: 'Direct coordination' }
  ];

  const modalStyle = {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background: 'rgba(0,0,0,0.8)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1000,
    padding: 20
  };

  const contentStyle = {
    background: '#1f2937',
    borderRadius: 12,
    padding: 24,
    maxWidth: 600,
    width: '100%',
    maxHeight: '90vh',
    overflowY: 'auto',
    border: '1px solid rgba(255,255,255,0.1)',
    boxShadow: '0 25px 50px -12px rgba(0,0,0,0.5)'
  };

  const inputStyle = {
    width: '100%',
    padding: '8px 12px',
    borderRadius: 6,
    border: '1px solid rgba(255,255,255,0.15)',
    background: 'rgba(255,255,255,0.05)',
    color: '#fff',
    fontSize: 14
  };

  const buttonStyle = {
    padding: '8px 16px',
    borderRadius: 6,
    border: 'none',
    cursor: 'pointer',
    fontSize: 14,
    fontWeight: 600
  };

  const primaryBtnStyle = {
    ...buttonStyle,
    background: '#ef4444',
    color: '#fff'
  };

  const secondaryBtnStyle = {
    ...buttonStyle,
    background: 'rgba(255,255,255,0.1)',
    color: '#e5e7eb',
    border: '1px solid rgba(255,255,255,0.2)'
  };

  return (
    <div style={modalStyle} onClick={handleBackdropClick}>
      <div ref={modalRef} style={contentStyle} className={className}>
        {/* Header */}
        <div style={{ 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'space-between', 
          marginBottom: 20,
          paddingBottom: 16,
          borderBottom: '1px solid rgba(255,255,255,0.1)'
        }}>
          <div>
            <h2 style={{ margin: 0, color: '#ef4444', fontSize: 20, fontWeight: 700 }}>
              🚨 Emergency Handoff
            </h2>
            <p style={{ margin: '4px 0 0 0', color: '#9ca3af', fontSize: 14 }}>
              Transfer operation to another agency
            </p>
          </div>
          <button 
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              color: '#9ca3af',
              fontSize: 24,
              cursor: 'pointer',
              padding: 4
            }}
          >
            ×
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          {/* Agency Type */}
          <div style={{ marginBottom: 20 }}>
            <label style={{ display: 'block', marginBottom: 8, color: '#e5e7eb', fontWeight: 600 }}>
              Target Agency
            </label>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 8 }}>
              {agencyTypes.map(agency => (
                <button
                  key={agency.value}
                  type="button"
                  onClick={() => setHandoffType(agency.value)}
                  style={{
                    ...buttonStyle,
                    background: handoffType === agency.value ? 'rgba(239,68,68,0.2)' : 'rgba(255,255,255,0.05)',
                    color: handoffType === agency.value ? '#ef4444' : '#e5e7eb',
                    border: handoffType === agency.value ? '1px solid rgba(239,68,68,0.5)' : '1px solid rgba(255,255,255,0.1)',
                    textAlign: 'left',
                    padding: '10px 12px'
                  }}
                  title={agency.desc}
                >
                  <div style={{ fontWeight: 600 }}>{agency.label}</div>
                  <div style={{ fontSize: 11, opacity: 0.8 }}>{agency.desc}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Priority Level */}
          <div style={{ marginBottom: 20 }}>
            <label style={{ display: 'block', marginBottom: 8, color: '#e5e7eb', fontWeight: 600 }}>
              Priority Level
            </label>
            <div style={{ display: 'flex', gap: 8 }}>
              {priorityLevels.map(level => (
                <button
                  key={level.value}
                  type="button"
                  onClick={() => setPriority(level.value)}
                  style={{
                    ...buttonStyle,
                    background: priority === level.value ? `${level.color}20` : 'rgba(255,255,255,0.05)',
                    color: priority === level.value ? level.color : '#e5e7eb',
                    border: priority === level.value ? `1px solid ${level.color}80` : '1px solid rgba(255,255,255,0.1)',
                    flex: 1
                  }}
                >
                  {level.label}
                </button>
              ))}
            </div>
          </div>

          {/* Contact Method */}
          <div style={{ marginBottom: 20 }}>
            <label style={{ display: 'block', marginBottom: 8, color: '#e5e7eb', fontWeight: 600 }}>
              Contact Method
            </label>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8 }}>
              {contactMethods.map(method => (
                <button
                  key={method.value}
                  type="button"
                  onClick={() => setContactMethod(method.value)}
                  style={{
                    ...buttonStyle,
                    background: contactMethod === method.value ? 'rgba(16,185,129,0.2)' : 'rgba(255,255,255,0.05)',
                    color: contactMethod === method.value ? '#10b981' : '#e5e7eb',
                    border: contactMethod === method.value ? '1px solid rgba(16,185,129,0.5)' : '1px solid rgba(255,255,255,0.1)',
                    textAlign: 'left',
                    padding: '8px 12px'
                  }}
                  title={method.desc}
                >
                  <div style={{ fontWeight: 600 }}>{method.label}</div>
                  <div style={{ fontSize: 11, opacity: 0.8 }}>{method.desc}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Contact Information */}
          <div style={{ marginBottom: 20 }}>
            <label style={{ display: 'block', marginBottom: 8, color: '#e5e7eb', fontWeight: 600 }}>
              Contact Information
            </label>
            <input
              type="text"
              value={contactInfo}
              onChange={(e) => setContactInfo(e.target.value)}
              placeholder="Radio frequency, phone number, or contact details"
              style={inputStyle}
              required
            />
          </div>

          {/* ETA */}
          <div style={{ marginBottom: 20 }}>
            <label style={{ display: 'block', marginBottom: 8, color: '#e5e7eb', fontWeight: 600 }}>
              Estimated Response Time
            </label>
            <select
              value={estimatedETA}
              onChange={(e) => setEstimatedETA(e.target.value)}
              style={inputStyle}
            >
              <option value="immediate">Immediate (0-2 min)</option>
              <option value="5-10">5-10 minutes</option>
              <option value="10-20">10-20 minutes</option>
              <option value="20-30">20-30 minutes</option>
              <option value="30+">30+ minutes</option>
              <option value="unknown">Unknown</option>
            </select>
          </div>

          {/* Data Sharing Options */}
          <div style={{ marginBottom: 20 }}>
            <label style={{ display: 'block', marginBottom: 12, color: '#e5e7eb', fontWeight: 600 }}>
              Share with Handoff
            </label>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {[
                { key: 'includeVideo', label: '📹 Live video feed', value: includeVideo, setter: setIncludeVideo },
                { key: 'includeLocation', label: '📍 Current location & tracking', value: includeLocation, setter: setIncludeLocation },
                { key: 'includeSuspectData', label: '👤 Suspect identification data', value: includeSuspectData, setter: setIncludeSuspectData }
              ].map(option => (
                <label key={option.key} style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={option.value}
                    onChange={(e) => option.setter(e.target.checked)}
                    style={{ accentColor: '#10b981' }}
                  />
                  <span style={{ color: '#e5e7eb', fontSize: 14 }}>{option.label}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Notes */}
          <div style={{ marginBottom: 24 }}>
            <label style={{ display: 'block', marginBottom: 8, color: '#e5e7eb', fontWeight: 600 }}>
              Additional Notes
            </label>
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Situation details, suspect description, hazards, special instructions..."
              rows={4}
              style={{
                ...inputStyle,
                resize: 'vertical',
                minHeight: 80
              }}
            />
          </div>

          {/* Current Status Summary */}
          {(currentLocation || suspectData) && (
            <div style={{ 
              background: 'rgba(59,130,246,0.1)', 
              border: '1px solid rgba(59,130,246,0.3)', 
              borderRadius: 8, 
              padding: 12, 
              marginBottom: 20 
            }}>
              <div style={{ color: '#60a5fa', fontWeight: 600, marginBottom: 8 }}>📊 Current Status</div>
              {currentLocation && (
                <div style={{ color: '#e5e7eb', fontSize: 13, marginBottom: 4 }}>
                  📍 Location: {currentLocation.lat?.toFixed(6)}, {currentLocation.lng?.toFixed(6)}
                </div>
              )}
              {suspectData && (
                <div style={{ color: '#e5e7eb', fontSize: 13 }}>
                  👤 Suspect: {suspectData.confidence ? `${(suspectData.confidence * 100).toFixed(0)}% match` : 'Detected'}
                </div>
              )}
            </div>
          )}

          {/* Action Buttons */}
          <div style={{ display: 'flex', gap: 12, justifyContent: 'flex-end' }}>
            <button 
              type="button" 
              onClick={onClose}
              style={secondaryBtnStyle}
              disabled={isSubmitting}
            >
              Cancel
            </button>
            <button 
              type="submit"
              style={{
                ...primaryBtnStyle,
                opacity: isSubmitting ? 0.7 : 1,
                cursor: isSubmitting ? 'not-allowed' : 'pointer'
              }}
              disabled={isSubmitting || !contactInfo.trim()}
            >
              {isSubmitting ? '🔄 Initiating Handoff...' : '🚨 Initiate Handoff'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default HandoffModal;