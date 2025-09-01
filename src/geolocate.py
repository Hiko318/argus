# src/geolocate.py
"""
Geolocation helper — fixed convention:
 - camera forward = -Z in camera coords (cam_dir = [x, y, -1])
 - rotation: R_world_from_camera = (R_cam @ R_body).T
This file also prints a small debug summary when run as __main__.
"""

import math
import numpy as np
from pyproj import CRS, Transformer

def pixel_to_latlon(u, v, intrinsics, drone_pose, gimbal_tilt_deg, dem_query_fn=None, debug=False):
    # 1) normalized direction in camera frame (camera forward = -Z)
    x = (u - intrinsics['cx']) / intrinsics['fx']
    y = (v - intrinsics['cy']) / intrinsics['fy']
    cam_dir = np.array([x, y, -1.0])
    cam_dir = cam_dir / np.linalg.norm(cam_dir)

    # 2) build rotation matrices (Z * Y * X order)
    yaw = math.radians(drone_pose.get('yaw_deg', 0.0))
    pitch = math.radians(drone_pose.get('pitch_deg', 0.0))
    roll = math.radians(drone_pose.get('roll_deg', 0.0))
    gimbal = math.radians(gimbal_tilt_deg)

    Rz = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw),  math.cos(yaw), 0],
        [0, 0, 1]
    ])
    Ry = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll),  math.cos(roll)]
    ])
    R_body = Rz @ Ry @ Rx

    # camera gimbal rotation around camera X (positive down)
    R_cam = np.array([
        [1, 0, 0],
        [0, math.cos(gimbal), -math.sin(gimbal)],
        [0, math.sin(gimbal),  math.cos(gimbal)]
    ])

    # 3) choose the working convention: world-from-camera = (R_cam @ R_body).T
    R_world_from_camera = (R_cam @ R_body).T

    # direction in world frame
    dir_world = R_world_from_camera @ cam_dir
    dE, dN, dU = dir_world

    if debug:
        print("DEBUG: cam_dir:", cam_dir.tolist())
        print("DEBUG: dir_world:", dir_world.tolist())
        print(f"DEBUG: dU = {float(dU):+.6f}")

    # 4) ENU local tangent setup
    lat0, lon0, alt0 = drone_pose['lat'], drone_pose['lon'], drone_pose['alt_m']
    crs_enu = CRS.from_proj4(f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +units=m")
    to_enu = Transformer.from_crs("epsg:4326", crs_enu, always_xy=True)
    to_ll = Transformer.from_crs(crs_enu, "epsg:4326", always_xy=True)

    drone_e, drone_n = to_enu.transform(lon0, lat0)
    drone_u = alt0

    # if ray points upward, bail
    if dU >= 0:
        return None

    # flat-plane intersection
    t = -drone_u / dU
    e = drone_e + dE * t
    n = drone_n + dN * t
    lon_est, lat_est = to_ll.transform(e, n)

    # optional DEM refinement
    if dem_query_fn:
        for _ in range(4):
            ground_alt = dem_query_fn(lat_est, lon_est)
            if ground_alt is None:
                break
            t = (ground_alt - drone_u) / dU
            e = drone_e + dE * t
            n = drone_n + dN * t
            lon_est, lat_est = to_ll.transform(e, n)

    return {"lat": lat_est, "lon": lon_est, "est_alt_m": float(drone_u + dU * t)}

# ---- demo DEM stub ----
def _demo_dem(lat, lon):
    return 0.0

# ---- CLI self-test ----
if __name__ == "__main__":
    intr = {"fx":800.0,"fy":800.0,"cx":640.0,"cy":360.0}
    # test: drone at 50m, level; gimbal 30° down
    drone = {"lat":0.0,"lon":0.0,"alt_m":50.0,"yaw_deg":0.0,"pitch_deg":0.0,"roll_deg":0.0}
    res = pixel_to_latlon(640, 360, intr, drone, gimbal_tilt_deg=30.0, dem_query_fn=_demo_dem, debug=True)
    print("Self-test result:", res)
