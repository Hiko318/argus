# Foresight SAR System - Operational Runbook

## Overview

This document provides operational procedures for deploying and managing the Foresight Search and Rescue (SAR) System in field operations.

## System Architecture

### Components
- **Backend Service**: Python-based detection and processing engine
- **Frontend Interface**: Web-based operator dashboard
- **Electron App**: Desktop application wrapper
- **Detection Pipeline**: YOLO-based human detection
- **Geolocation Service**: GPS coordinate processing
- **Evidence Storage**: Automated evidence capture and export

### Ports and Services
- Backend API: `http://localhost:8004`
- WebSocket Telemetry: `ws://localhost:8004/ws/telemetry`
- WebSocket Detections: `ws://localhost:8004/ws/detections`

## Pre-Deployment Checklist

### Hardware Requirements
- [ ] CUDA-compatible GPU (GTX 1060 or better)
- [ ] 8GB+ RAM
- [ ] 50GB+ available storage
- [ ] Stable internet connection
- [ ] Compatible drone/camera system

### Software Prerequisites
- [ ] Python 3.8+ installed
- [ ] Node.js 16+ installed
- [ ] Git installed
- [ ] CUDA drivers (if using GPU)

### Configuration
- [ ] `.env` file configured with API keys
- [ ] Model files downloaded and placed in correct directories
- [ ] Network firewall configured for required ports
- [ ] Backup storage configured

## Deployment Procedures

### Quick Start
1. **Clone Repository**
   ```bash
   git clone https://github.com/Hiko318/foresight.git
   cd foresight
   ```

2. **Environment Setup**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

4. **Start Services**
   ```bash
   # Terminal 1: Backend
   python src/backend/main.py
   
   # Terminal 2: Frontend (if needed)
   cd src/frontend && python -m http.server 3000
   
   # Terminal 3: Electron App
   cd foresight-electron && npm start
   ```

### Production Deployment
1. Use process managers (PM2, systemd)
2. Configure reverse proxy (nginx)
3. Set up SSL certificates
4. Configure log rotation
5. Set up monitoring and alerting

## Operational Procedures

### Starting a SAR Operation
1. **System Initialization**
   - Verify all services are running
   - Check model loading status
   - Confirm GPS/geolocation services
   - Test camera/drone connectivity

2. **Pre-Flight Checks**
   - Calibrate detection thresholds
   - Set search area boundaries
   - Configure evidence storage location
   - Test communication systems

3. **Operation Launch**
   - Start video feed processing
   - Monitor detection confidence levels
   - Track search coverage
   - Maintain communication logs

### During Operations
- **Monitor System Performance**
  - CPU/GPU utilization
  - Memory usage
  - Detection latency
  - Network connectivity

- **Evidence Management**
  - Review detection alerts
  - Validate positive detections
  - Export evidence packages
  - Maintain chain of custody

- **Communication**
  - Coordinate with field teams
  - Report findings to command
  - Update search status
  - Log all activities

### Post-Operation Procedures
1. **Data Preservation**
   - Export all evidence
   - Generate operation report
   - Archive video footage
   - Backup system logs

2. **System Maintenance**
   - Clear temporary files
   - Update detection models
   - Review system performance
   - Plan improvements

## Troubleshooting

### Common Issues

#### Service Won't Start
- Check port availability: `netstat -an | findstr :8004`
- Verify Python environment: `python --version`
- Check dependencies: `pip list`
- Review error logs in `out/logs/`

#### Poor Detection Performance
- Verify GPU utilization
- Check model file integrity
- Adjust confidence thresholds
- Review video quality settings

#### WebSocket Connection Errors
- Confirm backend service is running
- Check firewall settings
- Verify WebSocket URLs in frontend
- Test with browser developer tools

#### High Memory Usage
- Monitor video resolution settings
- Check for memory leaks in logs
- Restart services if needed
- Consider batch processing

### Emergency Procedures

#### System Crash During Operation
1. Immediately save current evidence
2. Restart core services
3. Verify data integrity
4. Resume operations
5. Investigate crash cause post-operation

#### Data Corruption
1. Stop all processing
2. Assess corruption extent
3. Restore from backups if available
4. Re-process affected data
5. Implement additional safeguards

## Performance Optimization

### GPU Optimization
- Use CUDA-optimized models
- Batch process when possible
- Monitor GPU memory usage
- Consider model quantization

### Network Optimization
- Use local processing when possible
- Compress video streams
- Implement adaptive quality
- Cache frequently accessed data

### Storage Optimization
- Implement automatic cleanup
- Use efficient video codecs
- Compress archived data
- Monitor disk space

## Security Considerations

### Data Protection
- Encrypt sensitive data at rest
- Use secure communication channels
- Implement access controls
- Regular security audits

### Privacy Compliance
- Follow local privacy laws
- Obtain proper authorizations
- Implement data retention policies
- Provide audit trails

## Maintenance Schedule

### Daily
- [ ] Check system status
- [ ] Review detection logs
- [ ] Monitor storage usage
- [ ] Backup critical data

### Weekly
- [ ] Update detection models
- [ ] Review performance metrics
- [ ] Clean temporary files
- [ ] Test backup systems

### Monthly
- [ ] Security updates
- [ ] Performance optimization
- [ ] Documentation updates
- [ ] Training refresher

## Contact Information

### Technical Support
- System Administrator: [Contact Info]
- Model Training Team: [Contact Info]
- Network Operations: [Contact Info]

### Emergency Contacts
- SAR Command Center: [Contact Info]
- IT Emergency Line: [Contact Info]
- Vendor Support: [Contact Info]

## Appendices

### A. Configuration Templates
### B. Log Analysis Scripts
### C. Performance Benchmarks
### D. Compliance Checklists