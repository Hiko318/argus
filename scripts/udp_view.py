import sys, cv2

URL = sys.argv[1] if len(sys.argv) > 1 else "udp://127.0.0.1:5000"

cap = cv2.VideoCapture(URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise SystemExit(f"Cannot open UDP stream: {URL}")

print(f"[ok] Reading from {URL}. Press ESC to close.")
while True:
    ok, frame = cap.read()
    if not ok:
        continue
    cv2.imshow("udp_view", frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
