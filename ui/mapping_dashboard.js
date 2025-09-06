class MappingDashboard {
    constructor(mapElementId, sarInterface) {
        this.mapElement = document.getElementById(mapElementId);
        this.sarInterface = sarInterface;
        this.map = null;
        
        // Map layers
        this.droneMarker = null;
        this.flightPathLayer = null;
        this.searchHeatmapLayer = null;
        this.detectionMarkers = new Map();
        this.searchAreaPolygons = [];
        this.waypointMarkers = [];
        
        // Search area tracking
        this.searchGrid = new Map(); // Grid-based search coverage
        this.gridSize = 50; // meters per grid cell
        this.searchIntensity = new Map(); // Track search intensity per area
        
        // Flight data
        this.flightPath = [];
        this.totalDistance = 0;
        this.searchedArea = 0;
        this.lastPosition = null;
        this.altitudeProfile = [];
        
        // Offline map support
        this.offlineMapData = null;
        this.mapBounds = null;
        
        this.initializeMap();
        this.setupEventHandlers();
    }
    
    initializeMap() {
        // Initialize map with default view
        this.map = L.map(this.mapElement, {
            center: [37.7749, -122.4194],
            zoom: 15,
            zoomControl: true,
            attributionControl: false
        });
        
        // Try to load offline map tiles first
        this.loadOfflineMap().then(() => {
            console.log('Offline map loaded successfully');
        }).catch(() => {
            console.log('Falling back to online map');
            this.loadOnlineMap();
        });
        
        // Initialize layers
        this.initializeLayers();
        
        // Add custom controls
        this.addCustomControls();
    }
    
    async loadOfflineMap() {
        try {
            // Check if offline map data exists
            const response = await fetch('/api/offline-map-config');
            if (response.ok) {
                this.offlineMapData = await response.json();
                
                // Load offline tiles
                const offlineTileLayer = L.tileLayer('/api/offline-tiles/{z}/{x}/{y}.png', {
                    maxZoom: 18,
                    attribution: 'Offline Map Data'
                });
                
                offlineTileLayer.addTo(this.map);
                
                // Set map bounds if available
                if (this.offlineMapData.bounds) {
                    this.mapBounds = L.latLngBounds(
                        this.offlineMapData.bounds.southwest,
                        this.offlineMapData.bounds.northeast
                    );
                    this.map.setMaxBounds(this.mapBounds);
                }
                
                return true;
            }
        } catch (error) {
            console.error('Failed to load offline map:', error);
            throw error;
        }
    }
    
    loadOnlineMap() {
        // Fallback to online map
        const onlineTileLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors',
            maxZoom: 18
        });
        
        onlineTileLayer.addTo(this.map);
    }
    
    initializeLayers() {
        // Flight path layer
        this.flightPathLayer = L.polyline([], {
            color: '#00ff00',
            weight: 3,
            opacity: 0.8,
            smoothFactor: 1
        }).addTo(this.map);
        
        // Search heatmap layer group
        this.searchHeatmapLayer = L.layerGroup().addTo(this.map);
        
        // Drone marker with custom icon
        const droneIcon = L.divIcon({
            className: 'drone-marker',
            html: `
                <div style="position: relative;">
                    <i class="fas fa-helicopter" style="color: #00ff00; font-size: 24px; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);"></i>
                    <div style="position: absolute; top: -30px; left: 50%; transform: translateX(-50%); background: rgba(0,0,0,0.8); color: white; padding: 2px 6px; border-radius: 3px; font-size: 10px; white-space: nowrap;">
                        DRONE
                    </div>
                </div>
            `,
            iconSize: [24, 24],
            iconAnchor: [12, 12]
        });
        
        this.droneMarker = L.marker([37.7749, -122.4194], {
            icon: droneIcon,
            rotationAngle: 0
        }).addTo(this.map);
        
        // Add drone info popup
        this.droneMarker.bindPopup(`
            <div>
                <strong>Drone Status</strong><br>
                <span id="drone-popup-coords">---.-----°, ---.-----°</span><br>
                <span id="drone-popup-altitude">Altitude: --- m</span><br>
                <span id="drone-popup-speed">Speed: --- m/s</span><br>
                <span id="drone-popup-heading">Heading: ---°</span>
            </div>
        `);
    }
    
    addCustomControls() {
        // Search area toggle control
        const searchAreaControl = L.control({ position: 'topleft' });
        searchAreaControl.onAdd = () => {
            const div = L.DomUtil.create('div', 'leaflet-bar leaflet-control');
            div.innerHTML = `
                <a href="#" title="Toggle Search Areas" style="background: #fff; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; text-decoration: none; color: #333;">
                    <i class="fas fa-search" style="font-size: 14px;"></i>
                </a>
            `;
            
            div.onclick = (e) => {
                e.preventDefault();
                this.toggleSearchAreas();
            };
            
            return div;
        };
        searchAreaControl.addTo(this.map);
        
        // Flight path toggle control
        const flightPathControl = L.control({ position: 'topleft' });
        flightPathControl.onAdd = () => {
            const div = L.DomUtil.create('div', 'leaflet-bar leaflet-control');
            div.innerHTML = `
                <a href="#" title="Toggle Flight Path" style="background: #fff; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; text-decoration: none; color: #333;">
                    <i class="fas fa-route" style="font-size: 14px;"></i>
                </a>
            `;
            
            div.onclick = (e) => {
                e.preventDefault();
                this.toggleFlightPath();
            };
            
            return div;
        };
        flightPathControl.addTo(this.map);
        
        // Center on drone control
        const centerControl = L.control({ position: 'topleft' });
        centerControl.onAdd = () => {
            const div = L.DomUtil.create('div', 'leaflet-bar leaflet-control');
            div.innerHTML = `
                <a href="#" title="Center on Drone" style="background: #fff; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; text-decoration: none; color: #333;">
                    <i class="fas fa-crosshairs" style="font-size: 14px;"></i>
                </a>
            `;
            
            div.onclick = (e) => {
                e.preventDefault();
                this.centerOnDrone();
            };
            
            return div;
        };
        centerControl.addTo(this.map);
        
        // Map legend
        const legendControl = L.control({ position: 'bottomright' });
        legendControl.onAdd = () => {
            const div = L.DomUtil.create('div', 'map-legend');
            div.innerHTML = `
                <div style="background: rgba(255,255,255,0.9); padding: 10px; border-radius: 5px; font-size: 12px;">
                    <strong>Legend</strong><br>
                    <div style="margin: 5px 0;">
                        <i class="fas fa-helicopter" style="color: #00ff00;"></i> Drone Position
                    </div>
                    <div style="margin: 5px 0;">
                        <i class="fas fa-user" style="color: #ff4444;"></i> Person Detected
                    </div>
                    <div style="margin: 5px 0;">
                        <span style="display: inline-block; width: 15px; height: 3px; background: #00ff00;"></span> Flight Path
                    </div>
                    <div style="margin: 5px 0;">
                        <span style="display: inline-block; width: 15px; height: 15px; background: rgba(255,255,0,0.3); border: 1px solid #ffff00;"></span> Searched Area
                    </div>
                </div>
            `;
            return div;
        };
        legendControl.addTo(this.map);
    }
    
    setupEventHandlers() {
        // Map click handler for manual waypoints
        this.map.on('click', (e) => {
            this.handleMapClick(e);
        });
        
        // Map zoom handler
        this.map.on('zoomend', () => {
            this.updateSearchHeatmap();
        });
    }
    
    updateDronePosition(telemetryData) {
        if (!telemetryData.gps) return;
        
        const newPos = [telemetryData.gps.latitude, telemetryData.gps.longitude];
        
        // Update drone marker
        this.droneMarker.setLatLng(newPos);
        
        // Update rotation if heading is available
        if (telemetryData.heading !== undefined) {
            this.droneMarker.setRotationAngle(telemetryData.heading);
        }
        
        // Update drone popup info
        const popupContent = `
            <div>
                <strong>Drone Status</strong><br>
                ${telemetryData.gps.latitude.toFixed(6)}°, ${telemetryData.gps.longitude.toFixed(6)}°<br>
                Altitude: ${telemetryData.altitude?.toFixed(1) || '---'} m<br>
                Speed: ${telemetryData.speed?.toFixed(1) || '---'} m/s<br>
                Heading: ${telemetryData.heading?.toFixed(0) || '---'}°
            </div>
        `;
        this.droneMarker.setPopupContent(popupContent);
        
        // Add to flight path
        this.flightPath.push(newPos);
        this.flightPathLayer.setLatLngs(this.flightPath);
        
        // Calculate distance and update search coverage
        if (this.lastPosition) {
            const distance = this.calculateDistance(
                this.lastPosition[0], this.lastPosition[1],
                newPos[0], newPos[1]
            );
            this.totalDistance += distance;
            
            // Update search grid
            this.updateSearchGrid(newPos, telemetryData.altitude || 100);
        }
        
        this.lastPosition = newPos;
        
        // Store altitude for profile
        this.altitudeProfile.push({
            position: newPos,
            altitude: telemetryData.altitude || 0,
            timestamp: Date.now()
        });
        
        // Keep only last 1000 altitude points
        if (this.altitudeProfile.length > 1000) {
            this.altitudeProfile = this.altitudeProfile.slice(-1000);
        }
    }
    
    updateSearchGrid(position, altitude) {
        // Calculate camera footprint based on altitude and FOV
        const fov = 84; // degrees (DJI O4 horizontal FOV)
        const footprintRadius = altitude * Math.tan((fov / 2) * Math.PI / 180);
        
        // Convert to grid coordinates
        const gridX = Math.floor(position[0] * 111320 / this.gridSize); // rough lat to meters
        const gridY = Math.floor(position[1] * 111320 * Math.cos(position[0] * Math.PI / 180) / this.gridSize);
        
        // Mark surrounding grid cells as searched
        const gridRadius = Math.ceil(footprintRadius / this.gridSize);
        
        for (let dx = -gridRadius; dx <= gridRadius; dx++) {
            for (let dy = -gridRadius; dy <= gridRadius; dy++) {
                const distance = Math.sqrt(dx * dx + dy * dy) * this.gridSize;
                if (distance <= footprintRadius) {
                    const gridKey = `${gridX + dx},${gridY + dy}`;
                    const intensity = this.searchIntensity.get(gridKey) || 0;
                    this.searchIntensity.set(gridKey, intensity + 1);
                }
            }
        }
        
        // Update search area calculation
        this.searchedArea = this.searchIntensity.size * (this.gridSize * this.gridSize) / 1000000; // km²
    }
    
    updateSearchHeatmap() {
        // Clear existing heatmap
        this.searchHeatmapLayer.clearLayers();
        
        // Create heatmap rectangles
        const zoom = this.map.getZoom();
        if (zoom < 12) return; // Don't show at low zoom levels
        
        this.searchIntensity.forEach((intensity, gridKey) => {
            const [gridX, gridY] = gridKey.split(',').map(Number);
            
            // Convert grid coordinates back to lat/lng
            const lat = (gridX * this.gridSize) / 111320;
            const lng = (gridY * this.gridSize) / (111320 * Math.cos(lat * Math.PI / 180));
            
            // Create rectangle for grid cell
            const bounds = [
                [lat, lng],
                [lat + this.gridSize / 111320, lng + this.gridSize / (111320 * Math.cos(lat * Math.PI / 180))]
            ];
            
            // Color based on search intensity
            const maxIntensity = Math.max(...this.searchIntensity.values());
            const normalizedIntensity = intensity / maxIntensity;
            const opacity = Math.min(0.1 + normalizedIntensity * 0.4, 0.5);
            
            const rectangle = L.rectangle(bounds, {
                color: '#ffff00',
                fillColor: '#ffff00',
                fillOpacity: opacity,
                weight: 1,
                opacity: 0.3
            });
            
            rectangle.bindTooltip(`Searched ${intensity} times`);
            this.searchHeatmapLayer.addLayer(rectangle);
        });
    }
    
    addDetectionMarker(detection) {
        if (!detection.geolocation) return;
        
        const { latitude, longitude } = detection.geolocation;
        const markerId = detection.id || `detection_${Date.now()}`;
        
        // Remove existing marker if it exists
        if (this.detectionMarkers.has(markerId)) {
            this.map.removeLayer(this.detectionMarkers.get(markerId));
        }
        
        // Create detection marker with enhanced icon
        const markerIcon = L.divIcon({
            className: 'detection-marker',
            html: `
                <div style="position: relative;">
                    <i class="fas fa-user" style="color: #ff4444; font-size: 20px; text-shadow: 2px 2px 4px rgba(0,0,0,0.8);"></i>
                    <div style="position: absolute; top: -25px; left: 50%; transform: translateX(-50%); background: rgba(255,68,68,0.9); color: white; padding: 2px 6px; border-radius: 3px; font-size: 10px; white-space: nowrap;">
                        #${detection.id || 'Unknown'}
                    </div>
                </div>
            `,
            iconSize: [20, 20],
            iconAnchor: [10, 10]
        });
        
        const marker = L.marker([latitude, longitude], {
            icon: markerIcon
        }).addTo(this.map);
        
        // Enhanced popup content
        const popupContent = `
            <div style="min-width: 200px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <strong>Person #${detection.id || 'Unknown'}</strong>
                    <span style="background: ${this.getConfidenceColor(detection.confidence)}; color: white; padding: 2px 6px; border-radius: 3px; font-size: 11px;">
                        ${(detection.confidence * 100).toFixed(0)}%
                    </span>
                </div>
                
                <div style="margin-bottom: 8px;">
                    <strong>Location:</strong><br>
                    ${latitude.toFixed(6)}°, ${longitude.toFixed(6)}°
                </div>
                
                <div style="margin-bottom: 8px;">
                    <strong>Distance:</strong> ${detection.distance ? detection.distance.toFixed(0) + 'm' : 'Unknown'}<br>
                    <strong>Time:</strong> ${new Date(detection.timestamp || Date.now()).toLocaleTimeString()}
                </div>
                
                <div style="display: flex; gap: 5px; margin-top: 10px;">
                    <button onclick="window.sarInterface.showDetectionPopup(${JSON.stringify(detection).replace(/"/g, '&quot;')})" 
                            style="flex: 1; padding: 5px; background: #4CAF50; color: white; border: none; border-radius: 3px; cursor: pointer; font-size: 11px;">
                        Details
                    </button>
                    <button onclick="window.sarInterface.handoffToTeam(${JSON.stringify(detection).replace(/"/g, '&quot;')})" 
                            style="flex: 1; padding: 5px; background: #ff9800; color: white; border: none; border-radius: 3px; cursor: pointer; font-size: 11px;">
                        Handoff
                    </button>
                </div>
            </div>
        `;
        
        marker.bindPopup(popupContent);
        
        // Add click handler to center on detection
        marker.on('click', () => {
            this.map.setView([latitude, longitude], Math.max(this.map.getZoom(), 16));
        });
        
        // Store marker reference
        this.detectionMarkers.set(markerId, marker);
        
        // Add detection circle to show uncertainty
        if (detection.accuracy) {
            const accuracyCircle = L.circle([latitude, longitude], {
                radius: detection.accuracy,
                color: '#ff4444',
                fillColor: '#ff4444',
                fillOpacity: 0.1,
                weight: 1,
                opacity: 0.5
            }).addTo(this.map);
            
            accuracyCircle.bindTooltip(`Accuracy: ±${detection.accuracy.toFixed(0)}m`);
        }
    }
    
    getConfidenceColor(confidence) {
        if (confidence >= 0.8) return '#4CAF50';
        if (confidence >= 0.6) return '#ff9800';
        return '#f44336';
    }
    
    toggleSearchAreas() {
        if (this.map.hasLayer(this.searchHeatmapLayer)) {
            this.map.removeLayer(this.searchHeatmapLayer);
        } else {
            this.map.addLayer(this.searchHeatmapLayer);
            this.updateSearchHeatmap();
        }
    }
    
    toggleFlightPath() {
        if (this.map.hasLayer(this.flightPathLayer)) {
            this.map.removeLayer(this.flightPathLayer);
        } else {
            this.map.addLayer(this.flightPathLayer);
        }
    }
    
    centerOnDrone() {
        if (this.droneMarker) {
            const dronePos = this.droneMarker.getLatLng();
            this.map.setView(dronePos, Math.max(this.map.getZoom(), 15));
        }
    }
    
    handleMapClick(e) {
        // Add waypoint marker on right-click
        if (e.originalEvent.button === 2) { // Right click
            this.addWaypoint(e.latlng);
        }
    }
    
    addWaypoint(latlng) {
        const waypointIcon = L.divIcon({
            className: 'waypoint-marker',
            html: `<i class="fas fa-map-pin" style="color: #2196F3; font-size: 16px;"></i>`,
            iconSize: [16, 16],
            iconAnchor: [8, 16]
        });
        
        const waypoint = L.marker(latlng, { icon: waypointIcon }).addTo(this.map);
        
        waypoint.bindPopup(`
            <div>
                <strong>Waypoint</strong><br>
                ${latlng.lat.toFixed(6)}°, ${latlng.lng.toFixed(6)}°<br>
                <button onclick="this.remove()" style="margin-top: 5px; padding: 3px 8px; background: #f44336; color: white; border: none; border-radius: 3px; cursor: pointer;">
                    Remove
                </button>
            </div>
        `);
        
        this.waypointMarkers.push(waypoint);
    }
    
    calculateDistance(lat1, lon1, lat2, lon2) {
        const R = 6371000; // Earth's radius in meters
        const dLat = (lat2 - lat1) * Math.PI / 180;
        const dLon = (lon2 - lon1) * Math.PI / 180;
        const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
                Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
                Math.sin(dLon/2) * Math.sin(dLon/2);
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
        return R * c;
    }
    
    getSearchedArea() {
        return this.searchedArea;
    }
    
    getTotalDistance() {
        return this.totalDistance;
    }
    
    exportFlightData() {
        return {
            flightPath: this.flightPath,
            altitudeProfile: this.altitudeProfile,
            searchedArea: this.searchedArea,
            totalDistance: this.totalDistance,
            detections: Array.from(this.detectionMarkers.keys()),
            searchIntensity: Object.fromEntries(this.searchIntensity)
        };
    }
}

// Export for use in main SAR interface
window.MappingDashboard = MappingDashboard;