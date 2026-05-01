import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

from config import SAMPLE_HZ, CONFRONTATION_GAP_S


def kalman_line_deviation(x: np.ndarray, y: np.ndarray,
                          window: int = 15) -> np.ndarray:
    """Residual distance from a smoothed racing-line proxy.

    High values indicate the defender is deviating from the normal line,
    which signals a blocking manoeuvre.
    """
    x_s = uniform_filter1d(x.astype(float), size=window)
    y_s = uniform_filter1d(y.astype(float), size=window)
    return np.sqrt((x - x_s) ** 2 + (y - y_s) ** 2)


def slipstream_coefficient(time_gap_s: np.ndarray,
                           rel_vel: np.ndarray) -> np.ndarray:
    """Exponential decay of slipstream benefit with time gap."""
    raw = np.exp(-np.clip(time_gap_s, 0, None) / 0.5) * np.clip(rel_vel, 0, None)
    return np.where(time_gap_s > 0, raw, 0.0)


def energy_delta_proxy(att_throttle: np.ndarray, def_throttle: np.ndarray,
                       att_drs: np.ndarray, def_drs: np.ndarray) -> np.ndarray:
    """Throttle + DRS advantage proxy (ERS data not public)."""
    return (att_throttle + 20.0 * att_drs) - (def_throttle + 20.0 * def_drs)



def compute_track_curvature(x: np.ndarray, y: np.ndarray,
                            smooth_window: int = 7) -> np.ndarray:
    """Curvature = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)."""
    x = uniform_filter1d(x.astype(float), size=smooth_window)
    y = uniform_filter1d(y.astype(float), size=smooth_window)
    dx, dy = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    denom = np.clip((dx ** 2 + dy ** 2) ** 1.5, 1e-10, None)
    return np.nan_to_num(np.abs(dx * ddy - dy * ddx) / denom,
                         nan=0.0, posinf=0.0)


def distance_to_next_corner(curvature: np.ndarray, dist: np.ndarray,
                             threshold: float = 0.002) -> np.ndarray:
    """Meters until curvature exceeds threshold (reverse scan, O(n))."""
    is_corner = curvature > threshold
    result = np.full(len(curvature), dist[-1] - dist[0])
    last_corner_dist = dist[-1] + 1000
    for i in range(len(curvature) - 1, -1, -1):
        if is_corner[i]:
            last_corner_dist = dist[i]
        result[i] = max(last_corner_dist - dist[i], 0.0)
    return result


def compute_straight_length(curvature: np.ndarray, dist: np.ndarray,
                             threshold: float = 0.002) -> np.ndarray:
    """Length of the contiguous straight segment each point belongs to."""
    is_straight = curvature < threshold
    result = np.zeros(len(curvature))
    i = 0
    while i < len(curvature):
        if is_straight[i]:
            j = i
            while j < len(curvature) and is_straight[j]:
                j += 1
            seg_length = dist[min(j, len(dist) - 1)] - dist[i]
            result[i:j] = seg_length
            i = j
        else:
            i += 1
    return result


def assign_sector_id(dist: np.ndarray, n_sectors: int = 3) -> np.ndarray:
    d_range = dist[-1] - dist[0]
    if d_range < 1:
        return np.zeros(len(dist), dtype=int)
    sector = ((dist - dist[0]) / d_range * n_sectors).astype(int)
    return np.clip(sector, 0, n_sectors - 1)


def build_pair_features(att_tel, def_tel, context: dict) -> pd.DataFrame | None:
    """Merge attacker/defender telemetry onto a common distance grid and
    compute all 45 engineered features.

    Returns None if there is insufficient overlap.
    """
    d_min = max(att_tel["Distance"].min(), def_tel["Distance"].min())
    d_max = min(att_tel["Distance"].max(), def_tel["Distance"].max())
    if d_max - d_min < 200:
        return None

    n_pts = max(int((d_max - d_min) / 5), 200)
    dist = np.linspace(d_min, d_max, n_pts)
    df = pd.DataFrame({"Distance": dist})

    df["att_Speed"] = np.interp(dist, att_tel["Distance"], att_tel["Speed"])
    df["att_Throttle"] = np.interp(dist, att_tel["Distance"], att_tel["Throttle"])
    df["att_Brake"] = np.interp(dist, att_tel["Distance"],
                                att_tel["Brake"].astype(float))
    df["att_Gear"] = np.interp(dist, att_tel["Distance"], att_tel["nGear"])
    drs_a = np.interp(dist, att_tel["Distance"], att_tel["DRS"])
    df["att_DRS"] = (drs_a >= 10).astype(int)
    df["att_X"] = np.interp(dist, att_tel["Distance"], att_tel["X"])
    df["att_Y"] = np.interp(dist, att_tel["Distance"], att_tel["Y"])

    df["def_Speed"] = np.interp(dist, def_tel["Distance"], def_tel["Speed"])
    df["def_Throttle"] = np.interp(dist, def_tel["Distance"], def_tel["Throttle"])
    df["def_Brake"] = np.interp(dist, def_tel["Distance"],
                                def_tel["Brake"].astype(float))
    df["def_Gear"] = np.interp(dist, def_tel["Distance"], def_tel["nGear"])
    drs_d = np.interp(dist, def_tel["Distance"], def_tel["DRS"])
    df["def_DRS"] = (drs_d >= 10).astype(int)
    df["def_X"] = np.interp(dist, def_tel["Distance"], def_tel["X"])
    df["def_Y"] = np.interp(dist, def_tel["Distance"], def_tel["Y"])

    att_time = np.interp(dist, att_tel["Distance"],
                         att_tel["SessionTime"].dt.total_seconds())
    def_time = np.interp(dist, def_tel["Distance"],
                         def_tel["SessionTime"].dt.total_seconds())
    df["DistanceToDriverAhead"] = att_time - def_time

    df["SpatialGap_m"] = np.sqrt(
        (df["att_X"] - df["def_X"]) ** 2 + (df["att_Y"] - df["def_Y"]) ** 2
    )

    df["RelativeVelocity"] = (df["att_Speed"] - df["def_Speed"]) / 3.6

    avg_speed_ms = (df["att_Speed"] + df["def_Speed"]).clip(lower=1) / 2 / 3.6
    gap_m = df["DistanceToDriverAhead"] * avg_speed_ms
    with np.errstate(divide="ignore", invalid="ignore"):
        ttc = np.where(df["RelativeVelocity"] > 0.5,
                       gap_m / df["RelativeVelocity"], np.nan)
    df["TTC"] = np.clip(ttc, a_min=None, a_max=30)

    df["ClosingRate"] = -np.gradient(df["DistanceToDriverAhead"], dist)
    df["SpeedRatio"] = df["att_Speed"] / df["def_Speed"].clip(lower=1)

    df["SlipstreamCoeff"] = slipstream_coefficient(
        df["DistanceToDriverAhead"].values, df["RelativeVelocity"].values)
    df["EnergyDelta"] = energy_delta_proxy(
        df["att_Throttle"].values, df["def_Throttle"].values,
        df["att_DRS"].values, df["def_DRS"].values)

    df["DefenderLineDeviation"] = kalman_line_deviation(
        df["def_X"].values, df["def_Y"].values,
        window=max(int(SAMPLE_HZ * 3), 5))

    df["att_FreshTyre"] = context["att_FreshTyre"]
    df["def_FreshTyre"] = context["def_FreshTyre"]
    df["CompoundAdvantage"] = context["compound_advantage"]
    df["att_MandatoryPitDone"] = context["att_MandatoryPitDone"]
    df["def_MandatoryPitDone"] = context["def_MandatoryPitDone"]
    df["att_TyreLife"] = context["att_TyreLife"]
    df["def_TyreLife"] = context["def_TyreLife"]

    win_1s = max(SAMPLE_HZ, 3)
    df["att_Speed_mean_1s"] = uniform_filter1d(df["att_Speed"].values, size=win_1s)
    df["def_Speed_mean_1s"] = uniform_filter1d(df["def_Speed"].values, size=win_1s)
    df["ClosingRate_mean_1s"] = uniform_filter1d(df["ClosingRate"].values, size=win_1s)
    df["att_Throttle_std_1s"] = (pd.Series(df["att_Throttle"])
                                 .rolling(win_1s, center=True, min_periods=1)
                                 .std().values)
    df["def_Throttle_std_1s"] = (pd.Series(df["def_Throttle"])
                                 .rolling(win_1s, center=True, min_periods=1)
                                 .std().values)

    in_zone = ((df["DistanceToDriverAhead"].values > 0) &
               (df["DistanceToDriverAhead"].values <= CONFRONTATION_GAP_S))
    bp = np.zeros(len(df))
    for idx in range(1, len(df)):
        bp[idx] = (bp[idx - 1] + 1) if in_zone[idx] else 0
    df["BattlePersistence"] = bp

    curvature = compute_track_curvature(df["def_X"].values, df["def_Y"].values)
    df["TrackCurvature"] = curvature
    df["DistanceToNextCorner"] = distance_to_next_corner(curvature, dist)
    df["StraightLength"] = compute_straight_length(curvature, dist)
    df["Sector_ID"] = assign_sector_id(dist)

    speed_ms = np.clip(df["att_Speed"].values / 3.6, 1.0, None)
    d_spacing = np.diff(dist, prepend=dist[0])
    d_spacing[0] = d_spacing[1] if len(d_spacing) > 1 else 5.0
    df["ElapsedTime"] = np.cumsum(d_spacing / speed_ms)

    return df
