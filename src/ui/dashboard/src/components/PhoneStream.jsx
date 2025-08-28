import React, { useEffect, useRef } from 'react';

const PhoneStream = () => {
    const videoRef = useRef(null);

    useEffect(() => {
        const setupStream = async () => {
            try {
                // Create a video element
                const video = videoRef.current;
                if (video) {
                    // Configure RTSP stream URL
                    const rtspUrl = 'rtsp://127.0.0.1:8554/scrcpy';
                    
                    // Set up video source (you might need a streaming library or WebRTC here)
                    // This is a placeholder - you'll need to implement the actual streaming logic
                    video.src = rtspUrl;
                    await video.play();
                }
            } catch (error) {
                console.error('Error setting up stream:', error);
            }
        };

        setupStream();

        // Cleanup function
        return () => {
            const video = videoRef.current;
            if (video) {
                video.pause();
                video.src = '';
            }
        };
    }, []);

    return (
        <div className="phone-stream">
            <video
                ref={videoRef}
                style={{
                    width: '100%',
                    maxHeight: '80vh',
                    objectFit: 'contain'
                }}
                controls
                autoPlay
            />
        </div>
    );
};

export default PhoneStream;
