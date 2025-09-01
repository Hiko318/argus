# scripts/ingest_probe.py
import argparse, json, os, cv2, math, time
from pathlib import Path
import pandas as pd

def load_telemetry(path):
    # Accept JSON list or CSV with columns: ts_ms, lat, lon, alt_m, yaw_deg, pitch_deg, roll_deg
    p = Path(path)
    if p.suffix.lower() in (".json",):
        with open(path, "r") as f:
            data = json.load(f)
        # expect list of dicts with ts_ms
        return data
    else:
        df = pd.read_csv(path)
        return df.to_dict(orient="records")

def write_ndjson_line(fh, d):
    fh.write(json.dumps(d, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--telemetry", required=False)
    ap.add_argument("--out", default="logs/ingest_probe.ndjson")
    ap.add_argument("--mission", default="dayone")
    ap.add_argument("--frame_ts_start_ms", type=int, default=None,
                    help="optional: explicit start timestamp in ms for first frame")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit("cannot open video: " + args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    telemetry = None
    if args.telemetry:
        telemetry = load_telemetry(args.telemetry)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fh = open(args.out, "w", encoding="utf8")

    # compute timestamps: if telemetry has a first ts_ms we align, otherwise use now()
    base_ts = None
    if args.frame_ts_start_ms:
        base_ts = args.frame_ts_start_ms
    elif telemetry and len(telemetry) and "ts_ms" in telemetry[0]:
        base_ts = telemetry[0]["ts_ms"]
    else:
        base_ts = int(time.time() * 1000)

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_ts = int(base_ts + idx * (1000.0 / fps))
        # approximate telemetry: pick nearest telemetry entry by ts_ms (if present)
        tel = None
        if telemetry:
            # find closest timestamp
            tel = min(telemetry, key=lambda t: abs(int(t.get("ts_ms", base_ts)) - frame_ts))
        # Save a small JPEG frame (optionally)
        frame_path = f"logs/{args.mission}_frame_{idx:06d}.jpg"
        cv2.imwrite(frame_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        rec = {
            "mission": args.mission,
            "frame_idx": idx,
            "frame_ts_ms": frame_ts,
            "frame_path": frame_path,
            "telemetry": tel,
        }
        write_ndjson_line(fh, rec)
        idx += 1

    fh.close()
    cap.release()
    print("Wrote", args.out)

if __name__ == "__main__":
    main()
