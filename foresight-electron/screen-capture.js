const { exec } = require('child_process');
const fs = require('fs');
const path = require('path');
const { promisify } = require('util');
const execAsync = promisify(exec);

// Use node-ffi or edge-js for Windows API calls in a production app
// For this demo, we'll use a simpler approach with screenshots

/**
 * Capture a window by its title
 * @param {string} windowTitle - The title of the window to capture
 * @returns {Promise<string>} - Base64 encoded image
 */
async function captureWindow(windowTitle) {
    try {
        // Create temp directory if it doesn't exist
        const tempDir = path.join(__dirname, 'temp');
        if (!fs.existsSync(tempDir)) {
            fs.mkdirSync(tempDir);
        }

        const outputPath = path.join(tempDir, 'capture.png');
        
        // Use PowerShell to capture the window
        // This is a simplified approach - in production, use a proper screen capture library
        const powershellCommand = `
            Add-Type -AssemblyName System.Windows.Forms
            Add-Type -AssemblyName System.Drawing
            
            # Find window by title
            $windows = [System.Diagnostics.Process]::GetProcesses() | 
                Where-Object { $_.MainWindowTitle -like "*${windowTitle}*" -and $_.MainWindowHandle -ne 0 }
            
            if ($windows -and $windows.Count -gt 0) {
                $window = $windows[0]
                $windowHandle = $window.MainWindowHandle
                
                # Get window bounds
                $rect = New-Object System.Drawing.Rectangle
                
                # Make window visible if minimized
                $SW_RESTORE = 9
                $user32 = Add-Type -MemberDefinition '[DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);' -Name 'User32ShowWindow' -Namespace Win32Functions -PassThru
                $null = $user32::ShowWindow($windowHandle, $SW_RESTORE)
                
                # Get window bounds
                $user32 = Add-Type -MemberDefinition '[DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hWnd, ref System.Drawing.Rectangle rect);' -Name 'User32GetWindowRect' -Namespace Win32Functions -PassThru
                $null = $user32::GetWindowRect($windowHandle, [ref]$rect)
                
                # Capture the window
                $bounds = [System.Drawing.Rectangle]::new($rect.Left, $rect.Top, $rect.Right - $rect.Left, $rect.Bottom - $rect.Top)
                $bitmap = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
                $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
                $graphics.CopyFromScreen($bounds.Left, $bounds.Top, 0, 0, $bounds.Size)
                $bitmap.Save("${outputPath}")
                $graphics.Dispose()
                $bitmap.Dispose()
                
                Write-Output "Captured window to ${outputPath}"
            } else {
                Write-Output "Window with title '${windowTitle}' not found"
                exit 1
            }
        `;
        
        // Save the PowerShell script to a file
        const scriptPath = path.join(tempDir, 'capture.ps1');
        fs.writeFileSync(scriptPath, powershellCommand);
        
        // Execute the PowerShell script
        await execAsync(`powershell -ExecutionPolicy Bypass -File "${scriptPath}"`);
        
        // Read the captured image and convert to base64
        if (fs.existsSync(outputPath)) {
            const imageBuffer = fs.readFileSync(outputPath);
            const base64Image = imageBuffer.toString('base64');
            
            // Clean up
            try {
                fs.unlinkSync(outputPath);
            } catch (err) {
                console.error('Failed to delete temp file:', err);
            }
            
            return base64Image;
        } else {
            throw new Error('Capture failed - output file not found');
        }
    } catch (error) {
        console.error('Window capture error:', error);
        
        // Return a black image as fallback
        return createBlackImage(480, 960);
    }
}

/**
 * Create a black image as fallback
 * @param {number} width - Image width
 * @param {number} height - Image height
 * @returns {string} - Base64 encoded black image
 */
function createBlackImage(width, height) {
    // This is a minimal 1x1 black PNG in base64
    // In a real app, you'd generate a proper black image of the requested size
    return 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+P+/HgAEtAJJXF+wHAAAAABJRU5ErkJggg==';
}

module.exports = {
    captureWindow
};