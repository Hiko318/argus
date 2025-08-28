import math
def dest_from_bearing(lat_deg, lon_deg, distance_m, bearing_deg):
    R = 6371000.0
    phi1 = math.radians(lat_deg)
    lam1 = math.radians(lon_deg)
    theta = math.radians(bearing_deg)
    dR = distance_m / R
    phi2 = math.asin(math.sin(phi1)*math.cos(dR) + math.cos(phi1)*math.sin(dR)*math.cos(theta))
    lam2 = lam1 + math.atan2(math.sin(theta)*math.sin(dR)*math.cos(phi1),
                             math.cos(dR)-math.sin(phi1)*math.sin(phi2))
    return math.degrees(phi2), math.degrees(lam2)
