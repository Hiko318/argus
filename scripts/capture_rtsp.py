import cv2

cap = cv2.VideoCapture("rtsp://127.0.0.1:8554/scrcpy", cv2.CAP_FFMPEG)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    
    cv2.imshow("rtsp", frame)
    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
