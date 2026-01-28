from __future__ import annotations
from typing import List, Optional, Dict, Tuple, Literal


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Tuple
from pathlib import Path
from collections import deque
from threading import Lock

import math
import time
import json
import os
import httpx


app = FastAPI(title="ATC Risk Tool", version="0.1.0")

# Allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



VERT_H_NM = 15.0
VERT_CAUTION_FT = 1000.0
VERT_WARNING_FT = 500.0


# --- MVP separation minima ---
H_MIN_NM = 5.0
V_MIN_FT = 1000.0

# --- Helpers: geo math ---
EARTH_RADIUS_M = 6371000.0
NM_TO_M = 1852.0


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in nautical miles."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    meters = EARTH_RADIUS_M * c
    return meters / NM_TO_M


def destination_point(lat: float, lon: float, bearing_deg: float, distance_nm: float) -> Tuple[float, float]:
    """Project point along great circle by distance (NM) at bearing (deg)."""
    brng = math.radians(bearing_deg)
    d = distance_nm * NM_TO_M  # meters
    phi1 = math.radians(lat)
    lam1 = math.radians(lon)

    delta = d / EARTH_RADIUS_M
    sin_phi2 = math.sin(phi1) * math.cos(delta) + math.cos(phi1) * math.sin(delta) * math.cos(brng)
    phi2 = math.asin(max(-1.0, min(1.0, sin_phi2)))

    y = math.sin(brng) * math.sin(delta) * math.cos(phi1)
    x = math.cos(delta) - math.sin(phi1) * math.sin(phi2)
    lam2 = lam1 + math.atan2(y, x)

    lon2 = (math.degrees(lam2) + 540) % 360 - 180
    return math.degrees(phi2), lon2






def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing in degrees from (lat1,lon1) to (lat2,lon2)."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlmb = math.radians(lon2 - lon1)
    y = math.sin(dlmb) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlmb)
    brng = math.degrees(math.atan2(y, x))
    return (brng + 360.0) % 360.0


def ang_diff_deg(a: float, b: float) -> float:
    """Smallest absolute angular difference in degrees."""
    d = (a - b + 540.0) % 360.0 - 180.0
    return abs(d)









# -------------------------
# Pydantic models (MUST be before endpoints that use them)
# -------------------------
class AircraftState(BaseModel):
    id: str
    callsign: Optional[str] = None
    lat: float
    lon: float
    alt_ft: float
    gs_kt: float
    track_deg: float
    vr_fpm: float = 0.0  # vertical rate feet/min


class Conflict(BaseModel):
    a_id: str
    b_id: str
    first_breach_s: float
    t_min_h_s: float
    min_h_nm: float
    min_v_ft: float
    risk_score: float
    explanation: str
    cpa_lat: float
    cpa_lon: float





class Alert(BaseModel):
    id: str
    type: str
    severity: str
    title: str
    details: str
    involvedAircraftIds: List[str]
    lat: float
    lon: float
    time_s: float
    data: Optional[dict] = None


class AlertsRequest(BaseModel):
    aircraft: List[AircraftState]
    lookahead_s: int = 600
    step_s: int = 15




def conflict_severity(c: Conflict) -> str:
    # Simple, sensible thresholds (tune later)
    soon = c.first_breach_s <= 120  # <= 2 minutes
    very_close = (c.min_h_nm <= 2.0) or (c.min_v_ft <= 500.0)

    if soon or very_close:
        return "warning"
    if c.first_breach_s <= 300 or c.min_h_nm <= 3.5 or c.min_v_ft <= 800.0:
        return "caution"
    return "info"


def conflict_to_alert(c: Conflict, by_id: Dict[str, AircraftState]) -> Alert:
    sev = conflict_severity(c)

    a = by_id.get(c.a_id)
    b = by_id.get(c.b_id)
    ca = (a.callsign if a else None) or c.a_id
    cb = (b.callsign if b else None) or c.b_id

    return Alert(
        id=f"sep:{c.a_id}:{c.b_id}:{int(c.first_breach_s)}",
        type="separation",
        severity=sev,  # "info" | "caution" | "warning"
        title=f"Separation {sev.upper()}: {ca} × {cb}",
        details=c.explanation,
        involvedAircraftIds=[c.a_id, c.b_id],
        lat=c.cpa_lat,
        lon=c.cpa_lon,
        time_s=c.first_breach_s,
        data={"conflict": c.model_dump()},
    )




@app.post("/alerts", response_model=List[Alert])
def alerts(req: AlertsRequest):
    aircraft = req.aircraft
    lookahead_s = req.lookahead_s
    step_s = req.step_s
    


    
    
    # --- Vertical crossing / level-off risk ---
    for i in range(len(aircraft)):
        for j in range(i + 1, len(aircraft)):
            a = aircraft[i]
            b = aircraft[j]

            ta = predict_track(a, lookahead_s, step_s)
            tb = predict_track(b, lookahead_s, step_s)

            best = None  # (t, h_nm, v_ft, lat, lon)
            crossed = False
            prev_sign = None

            for (t, latA, lonA, altA), (_, latB, lonB, altB) in zip(ta, tb):
                h = haversine_nm(latA, lonA, latB, lonB)
                if h > VERT_H_NM:
                    continue

                v = abs(altA - altB)
                mid_lat = (latA + latB) / 2.0
                mid_lon = (lonA + lonB) / 2.0

                # track whether they cross altitude (sign change)
                sign = 1 if (altA - altB) > 0 else (-1 if (altA - altB) < 0 else 0)
                if prev_sign is not None and sign != 0 and prev_sign != 0 and sign != prev_sign:
                    crossed = True
                prev_sign = sign

                if best is None or v < best[2]:
                    best = (t, h, v, mid_lat, mid_lon)

            if not best:
                continue

            t, h, v, lat, lon = best

            # only alert if meaningful
            if not crossed and v > VERT_CAUTION_FT:
                continue

            severity = "warning" if v <= VERT_WARNING_FT else "caution"
            title = "Vertical crossing risk" if crossed else "Low vertical separation"

            alerts.append({
                "id": f"vert:{a.id}:{b.id}:{int(t)}",
                "type": "vertical",
                "severity": severity,
                "title": title,
                "details": f"Predicted min vertical {v:.0f} ft at t+{t}s (H≈{h:.1f} NM).",
                "involvedAircraftIds": [a.id, b.id],
                "lat": lat,
                "lon": lon,
                "time_s": float(t),
                "data": {"min_v_ft": float(v), "min_h_nm": float(h), "crossed": crossed},
            })





    out: List[Alert] = []

    # -----------------------
    # 1) Separation alerts (existing)
    # -----------------------
    conflicts = find_conflicts(aircraft, lookahead_s, step_s)
    for c in conflicts:
        sev = "warning" if c.first_breach_s <= 120 else "caution"
        out.append(
            Alert(
                id=f"sep-{c.a_id}-{c.b_id}",
                type="separation",
                severity=sev,
                title="Predicted loss of separation",
                details=c.explanation,
                involvedAircraftIds=[c.a_id, c.b_id],
                lat=c.cpa_lat,
                lon=c.cpa_lon,
                time_s=c.first_breach_s,
                data={"conflict": c.model_dump()},
            )
        )

    # Precompute projected tracks once for other alert types
    tracks: Dict[str, List[Tuple[int, float, float, float]]] = {
        a.id: predict_track(a, lookahead_s, step_s) for a in aircraft
    }
    by_id = {a.id: a for a in aircraft}
    ids = [a.id for a in aircraft]

    # Helper to check if a pair already has a separation alert (avoid duplicates)
    sep_pairs = set()
    for c in conflicts:
        sep_pairs.add(tuple(sorted([c.a_id, c.b_id])))

    # -----------------------
    # 2) Vertical crossing / level-off risk (pairwise)
    #    Trigger when vertical gets tight (<1500ft) within ~8NM,
    #    but NOT already a separation conflict.
    # -----------------------
    V_TIGHT_FT = 1500.0
    H_WINDOW_NM = 8.0

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            ida, idb = ids[i], ids[j]
            pair_key = tuple(sorted([ida, idb]))
            if pair_key in sep_pairs:
                continue

            ta = tracks[ida]
            tb = tracks[idb]

            best_v = float("inf")
            best_h = float("inf")
            best_t = None
            best_lat = None
            best_lon = None

            for (t, latA, lonA, altA), (_, latB, lonB, altB) in zip(ta, tb):
                h = haversine_nm(latA, lonA, latB, lonB)
                if h > H_WINDOW_NM:
                    continue
                v = abs(altA - altB)
                if v < best_v:
                    best_v = v
                    best_h = h
                    best_t = t
                    best_lat = (latA + latB) / 2.0
                    best_lon = (lonA + lonB) / 2.0

            if best_t is None:
                continue

            if best_v < V_TIGHT_FT:
                # severity rule-of-thumb
                if best_v < 900 or best_t <= 120:
                    sev = "warning"
                elif best_v < 1200 or best_t <= 240:
                    sev = "caution"
                else:
                    sev = "info"

                ca = by_id[ida].callsign or ida
                cb = by_id[idb].callsign or idb
                details = (
                    f"{ca} and {cb} are predicted to get vertically tight "
                    f"({best_v:.0f} ft) within {best_h:.1f} NM in {best_t/60:.1f} min."
                )

                out.append(
                    Alert(
                        id=f"vert-{ida}-{idb}",
                        type="vertical",
                        severity=sev,
                        title="Altitude crossing / level-off risk",
                        details=details,
                        involvedAircraftIds=[ida, idb],
                        lat=float(best_lat),
                        lon=float(best_lon),
                        time_s=float(best_t),
                        data={
                            "min_v_ft": float(best_v),
                            "h_nm": float(best_h),
                            "t_s": float(best_t),
                        },
                    )
                )

    # -----------------------
    # 3) Wake turbulence proxy (current snapshot)
    #    Follower within 5NM, same track-ish, same altitude-ish, behind leader
    # -----------------------
    WAKE_MAX_NM = 5.0
    WAKE_ALT_FT = 300.0
    WAKE_TRACK_DEG = 20.0

    for leader in aircraft:
        for follower in aircraft:
            if leader.id == follower.id:
                continue

            h = haversine_nm(leader.lat, leader.lon, follower.lat, follower.lon)
            if h <= 0.2 or h > WAKE_MAX_NM:
                continue

            if abs(leader.alt_ft - follower.alt_ft) > WAKE_ALT_FT:
                continue

            if ang_diff_deg(leader.track_deg, follower.track_deg) > WAKE_TRACK_DEG:
                continue

            # is leader "ahead" of follower?
            brng = bearing_deg(follower.lat, follower.lon, leader.lat, leader.lon)
            if ang_diff_deg(brng, follower.track_deg) > 30.0:
                continue

            # severity by distance
            if h < 2.0:
                sev = "warning"
            elif h < 3.5:
                sev = "caution"
            else:
                sev = "info"

            ca = leader.callsign or leader.id
            cb = follower.callsign or follower.id
            details = f"{cb} is following {ca} at ~{h:.1f} NM, similar track/alt — potential wake exposure."

            out.append(
                Alert(
                    id=f"wake-{leader.id}-{follower.id}",
                    type="wake",
                    severity=sev,
                    title="Wake turbulence proxy",
                    details=details,
                    involvedAircraftIds=[leader.id, follower.id],
                    lat=follower.lat,
                    lon=follower.lon,
                    time_s=0.0,
                    data={"leader_id": leader.id, "follower_id": follower.id, "h_nm": h},
                )
            )


    # Montreal default (lat, lon)
    DEFAULT_CENTER = (45.5019, -73.5674)

    # -----------------------
    # 4) Sector congestion (global around Montreal)
    # -----------------------
    R_NM = 60.0
    count = 0
    for a in aircraft:
        if haversine_nm(DEFAULT_CENTER[0], DEFAULT_CENTER[1], a.lat, a.lon) <= R_NM:
            count += 1

    if count >= 12:
        if count >= 35:
            sev = "warning"
        elif count >= 20:
            sev = "caution"
        else:
            sev = "info"

        # crude delay estimate (placeholder, but useful UI-wise)
        delay_min = max(0.0, (count - 20) * 0.3)

        out.append(
            Alert(
                id="congestion-montreal",
                type="congestion",
                severity=sev,
                title="Sector congestion (Montreal)",
                details=f"{count} aircraft within {R_NM:.0f} NM. Rough delay estimate: ~{delay_min:.0f} min (placeholder).",
                involvedAircraftIds=[],
                lat=float(DEFAULT_CENTER[0]),
                lon=float(DEFAULT_CENTER[1]),
                time_s=0.0,
                data={"count": count, "radius_nm": R_NM, "delay_est_min": delay_min},
            )
        )

    # Sort by severity then time
    sev_rank = {"warning": 3, "caution": 2, "info": 1}
    out.sort(key=lambda a: (sev_rank.get(a.severity, 0), -a.time_s), reverse=True)

    return out






class PredictRequest(BaseModel):
    aircraft: List[AircraftState]
    lookahead_s: int = 600      # default 10 min
    step_s: int = 15            # sample every 15 sec


class PredictResponse(BaseModel):
    now_ts: float
    conflicts: List[Conflict]


class TrackPoint(BaseModel):
    t_s: int
    lat: float
    lon: float
    alt_ft: float


class ProjectedTrack(BaseModel):
    id: str
    callsign: Optional[str] = None
    points: List[TrackPoint]


class ProjectRequest(BaseModel):
    aircraft: List[AircraftState]
    lookahead_s: int = 600
    step_s: int = 15


# -------------------------
# DVR (rolling buffer) — now AircraftState exists, so no NameError
# -------------------------
DVR_MAX_FRAMES = 360          # ~10 minutes at 5s interval
DVR_INTERVAL_S = 5
dvr_lock = Lock()
dvr_frames = deque(maxlen=DVR_MAX_FRAMES)  # each item: {"ts": float, "aircraft": [dicts]}


class DVRPushRequest(BaseModel):
    aircraft: List[AircraftState]


@app.post("/dvr/push")
def dvr_push(req: DVRPushRequest):
    with dvr_lock:
        dvr_frames.append({
            "ts": time.time(),
            "aircraft": [a.model_dump() for a in req.aircraft],
        })
    return {"ok": True, "frames": len(dvr_frames)}


@app.get("/dvr/meta")
def dvr_meta():
    with dvr_lock:
        n = len(dvr_frames)
        if n == 0:
            return {"frames": 0, "interval_s": DVR_INTERVAL_S}
        return {
            "frames": n,
            "interval_s": DVR_INTERVAL_S,
            "oldest_ts": dvr_frames[0]["ts"],
            "newest_ts": dvr_frames[-1]["ts"],
        }


@app.get("/dvr/aircraft", response_model=List[AircraftState])
def dvr_aircraft(i: int):
    with dvr_lock:
        if i < 0 or i >= len(dvr_frames):
            raise HTTPException(status_code=400, detail="index out of range")
        frame = dvr_frames[i]
    return [AircraftState(**a) for a in frame["aircraft"]]


# -------------------------
# Core logic
# -------------------------
def predict_track(a: AircraftState, lookahead_s: int, step_s: int) -> List[Tuple[int, float, float, float]]:
    """Return list of (t_seconds, lat, lon, alt_ft)."""
    pts = []
    for t in range(0, lookahead_s + 1, step_s):
        dist_nm = a.gs_kt * (t / 3600.0)  # knots * hours = NM
        lat2, lon2 = destination_point(a.lat, a.lon, a.track_deg, dist_nm)
        alt2 = a.alt_ft + (a.vr_fpm * (t / 60.0))
        pts.append((t, lat2, lon2, alt2))
    return pts


def explain(a: AircraftState, b: AircraftState, first_breach_s: int, min_h_nm: float, min_v_ft: float, t_min_h_s: int) -> str:
    mins = first_breach_s / 60.0
    mins_min = t_min_h_s / 60.0
    ca = a.callsign or a.id
    cb = b.callsign or b.id
    return (
        f"{ca} and {cb} are predicted to breach separation in {mins:.1f} min. "
        f"Minimum predicted separation occurs around {mins_min:.1f} min: {min_h_nm:.2f} NM horizontal, {min_v_ft:.0f} ft vertical. "
        f"At current tracks/speeds, their trajectories converge within the lookahead window."
    )


def risk_score(first_breach_s: int, min_h_nm: float, min_v_ft: float) -> float:
    # Higher risk if sooner + closer
    h_term = max(0.0, (H_MIN_NM - min_h_nm) / H_MIN_NM)
    v_term = max(0.0, (V_MIN_FT - min_v_ft) / V_MIN_FT)
    sev = h_term + v_term  # 0..2
    t_min = max(0.5, first_breach_s / 60.0)
    return sev * (1.0 / t_min)


def find_conflicts(aircraft: List[AircraftState], lookahead_s: int, step_s: int) -> List[Conflict]:
    tracks: Dict[str, List[Tuple[int, float, float, float]]] = {
        a.id: predict_track(a, lookahead_s, step_s) for a in aircraft
    }
    by_id = {a.id: a for a in aircraft}
    ids = [a.id for a in aircraft]
    conflicts: List[Conflict] = []

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            ida, idb = ids[i], ids[j]
            ta, tb = tracks[ida], tracks[idb]

            first_breach: Optional[int] = None
            min_h = float("inf")
            min_v = float("inf")
            t_min_h = 0

            # IMPORTANT: CPA must be per-pair, not global
            cpa_lat = 0.0
            cpa_lon = 0.0

            for (t, latA, lonA, altA), (_, latB, lonB, altB) in zip(ta, tb):
                h = haversine_nm(latA, lonA, latB, lonB)
                v = abs(altA - altB)

                if h < min_h:
                    min_h = h
                    min_v = v
                    t_min_h = t
                    cpa_lat = (latA + latB) / 2.0
                    cpa_lon = (lonA + lonB) / 2.0

                if first_breach is None and (h < H_MIN_NM and v < V_MIN_FT):
                    first_breach = t

            if first_breach is not None:
                a = by_id[ida]
                b = by_id[idb]
                score = risk_score(first_breach, min_h, min_v)
                conflicts.append(
                    Conflict(
                        a_id=ida,
                        b_id=idb,
                        first_breach_s=float(first_breach),
                        t_min_h_s=float(t_min_h),
                        min_h_nm=float(min_h),
                        min_v_ft=float(min_v),
                        risk_score=float(score),
                        explanation=explain(a, b, first_breach, min_h, min_v, t_min_h),
                        cpa_lat=float(cpa_lat),
                        cpa_lon=float(cpa_lon),
                    )
                )

    conflicts.sort(key=lambda c: c.risk_score, reverse=True)
    return conflicts


def predict_track_points(a: AircraftState, lookahead_s: int, step_s: int) -> List[TrackPoint]:
    pts: List[TrackPoint] = []
    for t in range(0, lookahead_s + 1, step_s):
        dist_nm = a.gs_kt * (t / 3600.0)
        lat2, lon2 = destination_point(a.lat, a.lon, a.track_deg, dist_nm)
        alt2 = a.alt_ft + (a.vr_fpm * (t / 60.0))
        pts.append(TrackPoint(t_s=t, lat=lat2, lon=lon2, alt_ft=alt2))
    return pts


# -------------------------
# OpenSky
# -------------------------
OPENSKY_BASE = "https://opensky-network.org/api"
# --- OpenSky cache / throttle (prevents 429s) ---
OPENSKY_MIN_INTERVAL_S = 8.0  # call OpenSky at most once every 5 seconds (tune 5-15)
_opensky_cache = {"ts": 0.0, "data": []}  # last AircraftState list (dicts)
_opensky_lock = Lock()


def to_aircraft_state_from_opensky(state_row):
    """
    OpenSky 'states' rows are arrays. Key fields we use:
    0 icao24
    1 callsign
    5 lon
    6 lat
    7 baro_altitude (meters)
    9 velocity (m/s)
    10 true_track (deg)
    11 vertical_rate (m/s)
    13 geo_altitude (meters)
    """
    icao24 = (state_row[0] or "").strip()
    callsign = (state_row[1] or "").strip() or None
    lon = state_row[5]
    lat = state_row[6]
    baro_alt_m = state_row[7]
    geo_alt_m = state_row[13]
    vel_ms = state_row[9]
    track_deg = state_row[10]
    vr_ms = state_row[11]

    if lat is None or lon is None or track_deg is None or vel_ms is None:
        return None

    alt_m = geo_alt_m if geo_alt_m is not None else baro_alt_m
    if alt_m is None:
        return None

    alt_ft = alt_m * 3.28084
    gs_kt = vel_ms * 1.943844
    vr_fpm = (vr_ms or 0.0) * 196.8504  # m/s -> ft/min

    return AircraftState(
        id=icao24,
        callsign=callsign,
        lat=float(lat),
        lon=float(lon),
        alt_ft=float(alt_ft),
        gs_kt=float(gs_kt),
        track_deg=float(track_deg),
        vr_fpm=float(vr_fpm),
    )


# -------------------------
# API routes
# -------------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/demo/aircraft", response_model=List[AircraftState])
def demo_aircraft():
    return [
        AircraftState(id="A1", callsign="ACA123", lat=45.45, lon=-73.90, alt_ft=36000, gs_kt=450, track_deg=70, vr_fpm=0),
        AircraftState(id="B2", callsign="WJA456", lat=45.60, lon=-73.30, alt_ft=36000, gs_kt=460, track_deg=250, vr_fpm=0),
        AircraftState(id="C3", callsign="TSK789", lat=45.20, lon=-73.40, alt_ft=34000, gs_kt=430, track_deg=20, vr_fpm=0),
    ]





@app.get("/opensky/aircraft", response_model=List[AircraftState])
async def opensky_aircraft(
    lamin: float = 45.0,
    lamax: float = 46.2,
    lomin: float = -74.8,
    lomax: float = -72.5,
    limit: int = 200,
):
    now = time.time()

    # Read cache snapshot
    with _opensky_lock:
        cached_ts = float(_opensky_cache.get("ts", 0.0))
        cached_data = list(_opensky_cache.get("data", []))  # list of dicts
        age = now - cached_ts

    # If cache is fresh enough, serve it immediately (prevents spamming OpenSky)
    if cached_data and age < OPENSKY_MIN_INTERVAL_S:
        return [AircraftState(**a) for a in cached_data][:limit]

    params = {"lamin": lamin, "lamax": lamax, "lomin": lomin, "lomax": lomax}

    # Optional auth (recommended if you have an OpenSky account)
    auth = None
    user = os.getenv("OPENSKY_USERNAME")
    pwd = os.getenv("OPENSKY_PASSWORD")
    if user and pwd:
        auth = (user, pwd)

    try:
        async with httpx.AsyncClient(timeout=15.0, auth=auth) as client:
            r = await client.get(f"{OPENSKY_BASE}/states/all", params=params)

        # If rate-limited, fall back to cached data
        if r.status_code == 429:
            if cached_data:
                return [AircraftState(**a) for a in cached_data][:limit]
            raise HTTPException(status_code=429, detail="OpenSky rate-limited and cache empty")

        r.raise_for_status()
        data = r.json()

        states = data.get("states") or []
        out: List[AircraftState] = []
        for row in states:
            a = to_aircraft_state_from_opensky(row)
            if a is not None and a.gs_kt > 50:
                out.append(a)
            if len(out) >= limit:
                break

        # Update cache with fresh successful response
        with _opensky_lock:
            _opensky_cache["ts"] = now
            _opensky_cache["data"] = [a.model_dump() for a in out]

        return out

    except Exception:
        # On any error, serve cached response if available
        if cached_data:
            return [AircraftState(**a) for a in cached_data][:limit]
        raise



@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    now_ts = time.time()
    conflicts = find_conflicts(req.aircraft, req.lookahead_s, req.step_s)
    return PredictResponse(now_ts=now_ts, conflicts=conflicts)


@app.post("/project", response_model=List[ProjectedTrack])
def project(req: ProjectRequest):
    out: List[ProjectedTrack] = []
    for a in req.aircraft:
        out.append(
            ProjectedTrack(
                id=a.id,
                callsign=a.callsign,
                points=predict_track_points(a, req.lookahead_s, req.step_s),
            )
        )
    return out


# -------------------------
# Preset replay (from backend/data/replay.json)
# -------------------------
REPLAY_PATH = Path(__file__).parent / "data" / "replay.json"
_REPLAY_CACHE = None  # lazy-loaded


def load_replay():
    global _REPLAY_CACHE
    if _REPLAY_CACHE is not None:
        return _REPLAY_CACHE
    if not REPLAY_PATH.exists():
        _REPLAY_CACHE = {"frames": []}
        return _REPLAY_CACHE
    _REPLAY_CACHE = json.loads(REPLAY_PATH.read_text())
    return _REPLAY_CACHE


@app.get("/replay/meta")
def replay_meta():
    data = load_replay()
    frames = data.get("frames", [])
    return {"frames": len(frames), "path": str(REPLAY_PATH)}


@app.get("/replay/aircraft", response_model=List[AircraftState])
def replay_aircraft(frame: int = 0):
    data = load_replay()
    frames = data.get("frames", [])
    if not frames:
        raise HTTPException(status_code=404, detail="No replay frames found. Add backend/data/replay.json")
    if frame < 0 or frame >= len(frames):
        raise HTTPException(status_code=400, detail=f"frame out of range: 0..{len(frames)-1}")

    ac = frames[frame].get("aircraft", [])
    return [AircraftState(**a) for a in ac]
