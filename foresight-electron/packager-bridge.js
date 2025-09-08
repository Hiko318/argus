/**
 * Packager Bridge for Electron Frontend
 * 
 * Provides interface between Electron GUI and Python SAR packager.
 * Handles evidence packaging requests from the frontend.
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const { dialog } = require('electron');

class PackagerBridge {
    constructor() {
        this.pythonPath = 'python'; // Adjust if needed
        this.packagerScript = path.join(__dirname, '..', 'packager', 'packager.py');
        this.isPackaging = false;
    }

    /**
     * Package evidence with mission data
     * @param {Object} missionData - Mission metadata
     * @param {Array} files - List of files to include
     * @param {string} packageName - Name for the package
     * @param {Function} progressCallback - Progress update callback
     * @returns {Promise<string>} - Path to created package
     */
    async packageEvidence(missionData, files = [], packageName = null, progressCallback = null) {
        if (this.isPackaging) {
            throw new Error('Packaging already in progress');
        }

        this.isPackaging = true;
        
        try {
            // Generate package name if not provided
            if (!packageName) {
                const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
                packageName = `mission_${timestamp}`;
            }

            // Prepare mission metadata
            const metadata = {
                mission_id: missionData.mission_id || `mission_${Date.now()}`,
                mission_name: missionData.mission_name || 'SAR Mission',
                start_time: missionData.start_time || new Date().toISOString(),
                end_time: missionData.end_time || new Date().toISOString(),
                operator: missionData.operator || 'Unknown',
                aircraft_type: missionData.aircraft_type || 'Drone',
                sensor_config: missionData.sensor_config || {},
                geolocation_data: missionData.geolocation_data || {},
                detection_summary: missionData.detection_summary || {}
            };

            // Create temporary metadata file
            const tempDir = path.join(__dirname, 'temp');
            if (!fs.existsSync(tempDir)) {
                fs.mkdirSync(tempDir, { recursive: true });
            }

            const metadataFile = path.join(tempDir, 'mission_metadata.json');
            fs.writeFileSync(metadataFile, JSON.stringify(metadata, null, 2));

            // Prepare Python command
            const args = [
                this.packagerScript,
                '--metadata', metadataFile,
                '--package-name', packageName,
                '--compress'
            ];

            // Add files if provided
            if (files && files.length > 0) {
                args.push('--files', ...files);
            }

            console.log('Starting packaging with command:', this.pythonPath, args.join(' '));

            // Execute packaging
            const result = await this.executePythonScript(args, progressCallback);

            // Clean up temporary file
            try {
                fs.unlinkSync(metadataFile);
            } catch (e) {
                console.warn('Failed to clean up temporary metadata file:', e.message);
            }

            return result;

        } finally {
            this.isPackaging = false;
        }
    }

    /**
     * Execute Python packaging script
     * @param {Array} args - Command line arguments
     * @param {Function} progressCallback - Progress callback
     * @returns {Promise<string>} - Package path
     */
    executePythonScript(args, progressCallback) {
        return new Promise((resolve, reject) => {
            const process = spawn(this.pythonPath, args, {
                cwd: path.dirname(this.packagerScript)
            });

            let stdout = '';
            let stderr = '';

            process.stdout.on('data', (data) => {
                const output = data.toString();
                stdout += output;
                
                // Parse progress if callback provided
                if (progressCallback) {
                    this.parseProgress(output, progressCallback);
                }
                
                console.log('Packager output:', output.trim());
            });

            process.stderr.on('data', (data) => {
                const error = data.toString();
                stderr += error;
                console.error('Packager error:', error.trim());
            });

            process.on('close', (code) => {
                if (code === 0) {
                    // Extract package path from output
                    const packagePath = this.extractPackagePath(stdout);
                    resolve(packagePath || 'Package created successfully');
                } else {
                    reject(new Error(`Packaging failed with code ${code}: ${stderr}`));
                }
            });

            process.on('error', (error) => {
                reject(new Error(`Failed to start packaging process: ${error.message}`));
            });
        });
    }

    /**
     * Parse progress information from Python output
     * @param {string} output - Python script output
     * @param {Function} callback - Progress callback
     */
    parseProgress(output, callback) {
        // Look for progress indicators in output
        const lines = output.split('\n');
        
        for (const line of lines) {
            if (line.includes('Creating SAR package')) {
                callback({ stage: 'initializing', progress: 10, message: 'Initializing package...' });
            } else if (line.includes('Copied:')) {
                callback({ stage: 'copying', progress: 30, message: 'Copying files...' });
            } else if (line.includes('Created metadata.json')) {
                callback({ stage: 'metadata', progress: 60, message: 'Creating metadata...' });
            } else if (line.includes('Created SHA256SUMS')) {
                callback({ stage: 'manifest', progress: 80, message: 'Generating manifest...' });
            } else if (line.includes('Created compressed package')) {
                callback({ stage: 'complete', progress: 100, message: 'Package complete!' });
            }
        }
    }

    /**
     * Extract package path from Python output
     * @param {string} output - Python script output
     * @returns {string|null} - Package path
     */
    extractPackagePath(output) {
        const lines = output.split('\n');
        
        for (const line of lines) {
            if (line.includes('Created compressed package:')) {
                return line.split('Created compressed package:')[1].trim();
            } else if (line.includes('Created package directory:')) {
                return line.split('Created package directory:')[1].trim();
            }
        }
        
        return null;
    }

    /**
     * Show save dialog for package export
     * @param {string} defaultName - Default package name
     * @returns {Promise<string|null>} - Selected path or null if cancelled
     */
    async showSaveDialog(defaultName = 'sar_mission_package') {
        const result = await dialog.showSaveDialog({
            title: 'Save SAR Mission Package',
            defaultPath: `${defaultName}.zip`,
            filters: [
                { name: 'ZIP Archives', extensions: ['zip'] },
                { name: 'All Files', extensions: ['*'] }
            ]
        });

        return result.canceled ? null : result.filePath;
    }

    /**
     * Verify if packager script exists
     * @returns {boolean} - True if packager is available
     */
    isPackagerAvailable() {
        return fs.existsSync(this.packagerScript);
    }

    /**
     * Get packager status
     * @returns {Object} - Status information
     */
    getStatus() {
        return {
            available: this.isPackagerAvailable(),
            packaging: this.isPackaging,
            script_path: this.packagerScript
        };
    }
}

// Export for use in main process
module.exports = PackagerBridge;

// Example usage for renderer process
if (typeof window !== 'undefined') {
    // This would be used in the renderer process via IPC
    window.packageEvidence = async function(missionData, files = []) {
        try {
            // Send IPC message to main process
            const result = await window.electronAPI.packageEvidence(missionData, files);
            console.log('Package created:', result);
            return result;
        } catch (error) {
            console.error('Packaging failed:', error);
            throw error;
        }
    };
}